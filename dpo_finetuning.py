import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from datasets import Dataset

# Disable WandB logging
os.environ["WANDB_DISABLED"] = "true"

def load_dataset(dataset_path, test_size=0.2, seed=42):
    """Loads the dataset and prepares prompt-preference-rejection triplets."""
    with open(dataset_path, "r") as f:
        instruct_data = json.load(f)

    triplets = [
        {"prompt": row["instruction"] + "\n" + row["input"], 
         "chosen": row["chosen"], 
         "rejected": row["rejected"]}
        for row in instruct_data
    ]
    full_dataset = Dataset.from_list(triplets)
    return full_dataset.train_test_split(test_size=test_size, seed=seed)

def load_models(policy_model_name, ref_model_name, device):
    """Loads both policy and reference models along with their tokenizer."""
    policy_model = AutoModelForCausalLM.from_pretrained(policy_model_name).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name).to(device)

    tokenizer = AutoTokenizer.from_pretrained(policy_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token to EOS token

    return policy_model, ref_model, tokenizer

def collate_fn(batch, tokenizer, max_length=128, device="cuda"):
    """Tokenizes prompts and responses while creating preference pairs."""
    prompts = ["Instruct: " + item["prompt"] + "\n" for item in batch]
    chosen_responses = ["Output: " + item["chosen"] for item in batch]
    rejected_responses = ["Output: " + item["rejected"] for item in batch]

    prompt_ids = tokenizer(prompts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"].to(device)
    prefered_ids = tokenizer(chosen_responses, padding=True, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"].to(device)
    disprefered_ids = tokenizer(rejected_responses, padding=True, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"].to(device)

    prompt_prefered_ids = torch.cat([prompt_ids, prefered_ids], dim=-1)
    prompt_disprefered_ids = torch.cat([prompt_ids, disprefered_ids], dim=-1)

    prompt_prefered_mask = torch.cat([torch.ones_like(prompt_ids), torch.zeros_like(prefered_ids)], dim=-1)
    prompt_disprefered_mask = torch.cat([torch.ones_like(prompt_ids), torch.zeros_like(disprefered_ids)], dim=-1)

    return {
        "prompt_prefered_ids": prompt_prefered_ids,
        "prompt_disprefered_ids": prompt_disprefered_ids,
        "prompt_prefered_mask": prompt_prefered_mask,
        "prompt_disprefered_mask": prompt_disprefered_mask,
    }

def calculate_dpo_loss(
    pi_preferred_logps,
    pi_dispreferred_logps,
    ref_preferred_logps,
    ref_dispreferred_logps,
    beta=0.3
):
    """
    Computes the Direct Preference Optimization (DPO) loss.
    """
    preferred_logratio = pi_preferred_logps - ref_preferred_logps
    dispreferred_logratio = pi_dispreferred_logps - ref_dispreferred_logps

    reward_accuracies = (preferred_logratio > dispreferred_logratio).float().mean(dim=-1)
    reward_margins = (preferred_logratio - dispreferred_logratio).mean(dim=-1)

    loss = -F.logsigmoid(beta * (preferred_logratio - dispreferred_logratio)).mean(dim=-1)

    return loss, preferred_logratio.mean(dim=-1), dispreferred_logratio.mean(dim=-1), reward_accuracies, reward_margins

def get_log_prob(logits, labels):
    """Computes log probabilities for the given logits and labels."""
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1).mean(-1)

def kl_divergence(p_logits, q_logits):
    """
    Computes the token-level KL divergence between two sets of logits.
    """
    p_log_probs = F.log_softmax(p_logits, dim=-1)
    q_log_probs = F.log_softmax(q_logits, dim=-1)
    p_probs = p_log_probs.exp()
    kl = (p_probs * (p_log_probs - q_log_probs)).sum(dim=-1)
    return kl

#this code is partially based on https://allam.vercel.app/post/dpo/
def train_dpo(
    model,
    ref_model,
    tokenizer,
    train_dataloader,
    device="cuda",
    epochs=5,            
    beta=0.3,
    lr=2e-6,
    save_path="dpo-finetuned-model"
):
    """
    Trains the model using Direct Preference Optimization (DPO) while logging and storing per-step metrics:
      - The loss per fine-tuning step.
      - The KL divergence between the policy and reference model (for the "preferred" outputs) per step.
      - The total number of fine-tuning steps (global steps).
    
    Returns:
      A dictionary with keys:
         'step_loss': list of loss values for each fine-tuning step,
         'step_kl': list of KL divergence values for each fine-tuning step,
         'global_steps': list of global step indices.
    """
    model.train()
    ref_model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    step_losses = []
    step_kls = []
    global_steps = []
    
    global_step = 0
    
    for epoch in range(epochs):
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()

            # Unpack the batch.
            prompt_preferred_ids = batch["prompt_prefered_ids"]
            prompt_dispreferred_ids = batch["prompt_disprefered_ids"]
            prompt_preferred_mask = batch["prompt_prefered_mask"]
            prompt_dispreferred_mask = batch["prompt_disprefered_mask"]

            # Forward pass: policy (pi) logits.
            pi_preferred_logits = model(prompt_preferred_ids, attention_mask=prompt_preferred_mask).logits
            pi_dispreferred_logits = model(prompt_dispreferred_ids, attention_mask=prompt_dispreferred_mask).logits

            # Forward pass: reference (ref) logits.
            ref_preferred_logits = ref_model(prompt_preferred_ids, attention_mask=prompt_preferred_mask).logits
            ref_dispreferred_logits = ref_model(prompt_dispreferred_ids, attention_mask=prompt_dispreferred_mask).logits

            # Convert logits to log probabilities (for loss computation).
            pi_preferred_logps = get_log_prob(pi_preferred_logits, prompt_preferred_ids)
            pi_dispreferred_logps = get_log_prob(pi_dispreferred_logits, prompt_dispreferred_ids)
            ref_preferred_logps = get_log_prob(ref_preferred_logits, prompt_preferred_ids)
            ref_dispreferred_logps = get_log_prob(ref_dispreferred_logits, prompt_dispreferred_ids)

            loss, _, _, _, _ = calculate_dpo_loss(
                pi_preferred_logps, pi_dispreferred_logps,
                ref_preferred_logps, ref_dispreferred_logps,
                beta=beta
            )

            # Compute KL divergence on the "preferred" logits.
            kl_values = kl_divergence(pi_preferred_logits, ref_preferred_logits) 
            avg_kl = kl_values.mean().item()  # Average over tokens and batch.

            loss.backward()
            optimizer.step()

            # Increment global step and store step-level metrics.
            global_step += 1
            global_steps.append(global_step)
            step_losses.append(loss.item())
            step_kls.append(avg_kl)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

    metrics = {
        "step_loss": step_losses,
        "step_kl": step_kls,
        "global_steps": global_steps
    }
    return metrics

