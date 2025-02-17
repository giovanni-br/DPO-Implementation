import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

def load_models(model_paths, device):
    models = {name: AutoModelForCausalLM.from_pretrained(path).to(device) for name, path in model_paths.items()}
    tokenizer = AutoTokenizer.from_pretrained(list(model_paths.values())[0], padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    return models, tokenizer

def load_data(json_path, subset_size):
    with open(json_path, "r") as f:
        data = json.load(f)
    return list(np.random.choice(data, size=min(subset_size, len(data)), replace=False))

def prepare_eval_pairs(data):
    return [{"prompt": row["instruction"] + row["input"],
             "chosen": row["chosen"],
             "rejected": row["rejected"]} for row in data]

def decode_generated_text(input_ids, output_ids, tokenizer):
    prompt_len = input_ids.shape[0]
    gen_ids = output_ids[prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)

def compute_similarity(generated, chosen, rejected, sbert_model):
    generated_emb = sbert_model.encode(generated, convert_to_tensor=True)
    chosen_emb = sbert_model.encode(chosen, convert_to_tensor=True)
    rejected_emb = sbert_model.encode(rejected, convert_to_tensor=True)
    return util.cos_sim(generated_emb, chosen_emb).item() > util.cos_sim(generated_emb, rejected_emb).item()

def evaluate_model(model, model_name, tokenizer, sbert_model, eval_pairs, device, betas, batch_size):
    win_rates = []
    
    for beta in betas:
        correct, total = 0, 0
        print(f"\nEvaluating {model_name} at β={beta}...\n")
        
        for i in range(0, len(eval_pairs), batch_size):
            batch = eval_pairs[i: i + batch_size]
            batch_prompts = [ex["prompt"] for ex in batch]
            batch_chosen = [ex["chosen"] for ex in batch]
            batch_rejected = [ex["rejected"] for ex in batch]

            inputs = tokenizer(batch_prompts, return_tensors="pt", truncation=True, padding=True).to(device)
            max_new_tokens = 30
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=beta,
                    pad_token_id=tokenizer.eos_token_id
                )

            batch_generated = [decode_generated_text(inputs["input_ids"][i], output_ids, tokenizer)
                               for i, output_ids in enumerate(outputs)]

            # Compare generated responses with chosen/rejected responses
            for j, (generated, chosen, rejected) in enumerate(zip(batch_generated, batch_chosen, batch_rejected)):
                is_correct = compute_similarity(generated, chosen, rejected, sbert_model)
                correct += is_correct
                total += 1

        win_rate = (correct / total) * 100
        win_rates.append(win_rate)
        print(f"\n{model_name} | Beta={beta:.2f} => Win Rate: {win_rate:.2f}%\n{'='*80}")
    
    return win_rates


def plot_results(betas, model_win_rates):
    plt.figure(figsize=(8, 5))
    for model_name, win_rates in model_win_rates.items():
        plt.plot(betas, win_rates, marker="o", linestyle="--", label=model_name)
    plt.xlabel("Beta (Sampling Temperature)")
    plt.ylabel("Win Rate (%)")
    plt.title("Comparison of Model Win Rates")
    plt.legend()
    plt.show()

def run_evaluation(model_paths, json_path, save_path="win_rate_comparison.csv", subset_size=100, batch_size=4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    betas = [0.1, 0.25, 0.5, 0.75, 1.0]
    models, tokenizer = load_models(model_paths, device)
    sbert_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").to(device)
    data = load_data(json_path, subset_size)
    eval_pairs = prepare_eval_pairs(data)
    model_win_rates = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        model_win_rates[model_name] = evaluate_model(model, model_name, tokenizer, sbert_model, eval_pairs, device, betas, batch_size)
    
    df_results = pd.DataFrame({"Beta": betas})
    for model_name, win_rates in model_win_rates.items():
        df_results[model_name + " Win Rate (%)"] = win_rates
    df_results.to_csv(save_path, index=False)
    
    plot_results(betas, model_win_rates)
