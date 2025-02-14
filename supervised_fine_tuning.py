import os
import json
import math
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from datasets import Dataset

def load_dataset(dataset_path, test_size=0.2, seed=42):
    """Loads dataset from JSON file and prepares text samples."""
    with open(dataset_path, "r") as f:
        instruct_data = json.load(f)

    pairs = [
        {"text": f"Instruct: {row['instruction']}\n{row['input']}\nAnswer: {row['chosen']}"}
        for row in instruct_data
    ]
    
    full_dataset = Dataset.from_list(pairs)
    return full_dataset.train_test_split(test_size=test_size, seed=seed)

def load_model(model_name, device):
    """Loads the pre-trained model and tokenizer, ensuring proper padding token handling."""
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  
    return model, tokenizer

def collate_fn(batch, tokenizer, max_length=128, device="cuda"):
    """Tokenizes and pads batch samples properly, ensuring attention masks are set correctly."""
    texts = [item["text"] for item in batch]
    encoding = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    
    return {
        "input_ids": encoding["input_ids"].to(device),
        "attention_mask": encoding["attention_mask"].to(device), 
        "labels": encoding["input_ids"].to(device),
    }

def evaluate(model, eval_dataloader):
    """Computes average loss and perplexity on the evaluation set."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            total_loss += outputs.loss.item()

    avg_loss = total_loss / len(eval_dataloader)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

def train(model, tokenizer, train_dataloader, eval_dataloader, epochs=3, learning_rate=2e-6, save_path="fine_tuned_model"):
    """Trains the model for a given number of epochs."""
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        eval_loss, eval_ppl = evaluate(model, eval_dataloader)

        print(f"\n[Epoch {epoch+1}] Train Loss = {avg_train_loss:.4f}, Eval Loss = {eval_loss:.4f}, Eval PPL = {eval_ppl:.4f}\n")

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

def test_model_generation(model, tokenizer, eval_dataset, num_samples=3, max_new_tokens=50, temperature=0.7, top_k=50, top_p=0.9):
    """Tests the model by generating outputs for random samples."""
    model.eval()
    indices = random.sample(range(len(eval_dataset)), num_samples)
    for i in indices:
        sample = eval_dataset[i]
        text = sample["text"]
        splitted = text.split("Answer:") if "Answer:" in text else [text]

        prompt_part = splitted[0] + "Answer:" if len(splitted) > 1 else text
        input_ids = tokenizer.encode(prompt_part, return_tensors="pt").to(model.device)

        output_ids = model.generate(
            input_ids, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id,
            temperature=temperature, do_sample=True, top_k=top_k, top_p=top_p
        )
        gen_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        print("-------------- EXAMPLE --------------")
        print(f"PROMPT:\n{prompt_part}")
        print(f"GENERATED:\n{gen_text}")
        if len(splitted) > 1:
            print(f"REFERENCE:\n{splitted[1].strip()}")
        print("-------------------------------------\n")
