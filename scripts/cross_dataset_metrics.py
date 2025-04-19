"""
Calculate and compare metrics for SAEs trained on specific datasets when applied to others.

Example usage:
    python scripts/cross_dataset_metrics.py --sae_paths=./checkpoints --num_sequences=1000
"""

import os
from os.path import join, basename
import fire
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE
from datasets import load_dataset
from transformer_lens.utils import tokenize_and_concatenate, get_act_name
from convert import convert

# Make tokenizer global so it's accessible in all functions
tokenizer = None

def get_sae_folders(sae_paths, llm_id, site, layer, dict_size, topk, lr, seed, seq_len, steps, faithful, tiny, openweb, red, pile):
    dataset_ids["faithful"] = f"seonglae/{faithful}"
    model_name = llm_id.split('/')[-1]
    if model_name == "gpt2":
        model_name = "gpt2-small"
    
    # Define SAE paths directly in main
    sae_folders = {
        "fineweb": f"{sae_paths}/{model_name}_blocks.{layer}.hook_{site}_{dict_size}_topk_{topk}_{lr}_{seed}_fineweb_{seq_len}_{steps}",
        "faithful": f"{sae_paths}/{model_name}_blocks.{layer}.hook_{site}_{dict_size}_topk_{topk}_{lr}_{seed}_{faithful}_{seq_len}_{steps}"
    }
    tiny = True
    openweb = True
    if tiny:
        sae_folders["tinystories"] = f"{sae_paths}/{model_name}_blocks.{layer}.hook_{site}_{dict_size}_topk_{topk}_{lr}_{seed}_TinyStories_{seq_len}_{steps}"
    if openweb:
        sae_folders["openwebtext"] = f"{sae_paths}/{model_name}_blocks.{layer}.hook_{site}_{dict_size}_topk_{topk}_{lr}_{seed}_openwebtext_{seq_len}_{steps}"
    if red:
        sae_folders["redpajama"] = f"{sae_paths}/{model_name}_blocks.{layer}.hook_{site}_{dict_size}_topk_{topk}_{lr}_{seed}_redpajama_{seq_len}_{steps}"
    if pile:
        sae_folders["pile"] = f"{sae_paths}/{model_name}_blocks.{layer}.hook_{site}_{dict_size}_topk_{topk}_{lr}_{seed}_pile-uncopyrighted_{seq_len}_{steps}"
    return sae_folders

# Dataset options with actual paths
dataset_ids = {
    "fineweb": "HuggingFaceFW/fineweb", 
    "tinystories": "roneneldan/TinyStories",
    "pile": "monology/pile-uncopyrighted",
    "openwebtext": "Skylion007/openwebtext",
}

def calculate_explained_variance(original, reconstructed):
    """Calculate explained variance: 1 - (variance of error) / (variance of original)"""
    original_var = torch.var(original)
    error_var = torch.var(original - reconstructed)
    return 1 - (error_var / original_var)

def evaluate_sae_on_dataset(sae, llm, dataset_samples, device, layer_i, max_sequences=50):
    """Evaluate an SAE on tokenized dataset samples"""
    all_l2_losses = []
    all_ce_losses = []
    all_explained_vars = []
    
    for i, item in enumerate(tqdm(dataset_samples[:max_sequences], desc=f"Evaluating")):
        input_ids = item["input_ids"].to(device)
        
        with torch.no_grad():
            # Original forward pass
            orig_outputs = llm(input_ids.unsqueeze(0), output_hidden_states=True)
            orig_hidden_state = orig_outputs.hidden_states[layer_i]
            orig_logits = orig_outputs.logits
            
            # Get SAE reconstruction
            act = sae.encode(orig_hidden_state)
            reconstructed = sae.decode(act)
            
            # Calculate L2 loss
            l2 = F.mse_loss(orig_hidden_state, reconstructed)
            all_l2_losses.append(l2.item())
            
            # Calculate explained variance (1 - error_var/original_var)
            original_var = torch.var(orig_hidden_state)
            error_var = torch.var(orig_hidden_state - reconstructed)
            explained_var = 1 - (error_var / original_var)
            all_explained_vars.append(explained_var.item())
            
            # Run forward pass with reconstructed hidden state
            recon_outputs = apply_reconstruction(llm, input_ids.unsqueeze(0), reconstructed, layer_i)
            recon_logits = recon_outputs.logits
            
            # Calculate CE loss difference (using shift logits like in layer_loss.py)
            # First make sure we have at least 2 tokens
            if input_ids.size(0) > 1:
                # Shift logits and prepare labels
                shift_logits_orig = orig_logits[:, :-1, :]
                shift_logits_recon = recon_logits[:, :-1, :]
                labels = input_ids.unsqueeze(0)[:, 1:]
                
                # Calculate cross entropy loss
                orig_ce = F.cross_entropy(shift_logits_orig.reshape(-1, shift_logits_orig.size(-1)), 
                                         labels.reshape(-1))
                recon_ce = F.cross_entropy(shift_logits_recon.reshape(-1, shift_logits_recon.size(-1)), 
                                          labels.reshape(-1))
                ce_diff = recon_ce - orig_ce
                all_ce_losses.append(ce_diff.item())
    
    return {
        "l2_loss": np.mean(all_l2_losses),
        "explained_variance": np.mean(all_explained_vars),
        "ce_diff": np.mean(all_ce_losses) if all_ce_losses else 0.0
    }

def apply_reconstruction(llm, input_ids, reconstructed, layer_i):
    """Apply the reconstructed hidden state to the model and get outputs"""
    outputs = None
    hook_handle = None
    
    def reconstruction_pre_hook(module, inputs):
        """Pre-hook to replace the module's input with reconstructed activation"""
        # Return the reconstructed tensor to replace the original input
        return (reconstructed,) + inputs[1:] if len(inputs) > 1 else reconstructed
    
    # Register the pre-hook (like in layer_loss.py) instead of a forward hook
    if llm.config.model_type == "gpt2":
        hook_handle = llm.transformer.h[layer_i].register_forward_pre_hook(reconstruction_pre_hook)
    else:
        hook_handle = llm.model.layers[layer_i].register_forward_pre_hook(reconstruction_pre_hook)
    
    # Run the forward pass with use_cache=False to avoid IndexError
    outputs = llm(input_ids, output_hidden_states=True, use_cache=False)
    
    # Remove the hook
    hook_handle.remove()
    
    return outputs

def load_datasets(tokenizer, max_samples=100):
    all_datasets = {}
    
    for name, dataset_id in dataset_ids.items():
        print(f"Loading dataset: {name}")
        if name == "redpajama":
            raw_dataset = load_dataset(dataset_id, 'default', split="train", streaming=True)
        else:
            raw_dataset = load_dataset(path=dataset_id, split="train", streaming=True)
        
        collected_samples = []
        for i, item in enumerate(raw_dataset):
            if i >= max_samples:
                break
                
            if "text" in item and item["text"].strip():  # Check if text exists and is not empty
                # Tokenize each example directly
                tokens = tokenizer(item["text"], return_tensors="pt", truncation=True, max_length=512)["input_ids"][0]
                collected_samples.append({"input_ids": tokens})
        
        all_datasets[name] = collected_samples
        print(f"Successfully loaded {len(collected_samples)} sequences from {name}")
    
    return all_datasets

def main(sae_paths="./checkpoints", llm_id="meta-llama/Llama-3.2-1B", site="resid_pre", layer=12, seed=42, seq_len=512, lr=0.0002, pile=False, tiny=False, openweb=False, red=False,
         topk=48, dict_size=14336, num_sequences=100, results_folder="results_gemma", steps=195311, half_topk=False, faithful="faithful-llama3.2-1b"):
    """
    Main function to evaluate SAEs trained on different datasets.
    
    Args:
        sae_paths: Path to directory containing SAE checkpoints
        llm_id: Model ID for the language model
        site: Site to extract activations from
        layer: Layer to extract activations from
        topk: Top-k value for SAE
        seed: Seed for random operations
        dict_size: Dictionary size for SAE
        num_sequences: Number of sequences to evaluate on
        results_folder: Folder to save results
    """
    global tokenizer
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else device
    print(f"Using device: {device}")
    
    os.makedirs(results_folder, exist_ok=True)
    
    print("Loading LLM and tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(llm_id)
    llm = AutoModelForCausalLM.from_pretrained(llm_id, torch_dtype=torch.bfloat16).to(device)
    llm.eval()
    dataset_ids["faithful"] = f"seonglae/{faithful}"
    model_name = llm_id.split('/')[-1]
    if model_name == "gpt2":
        model_name = "gpt2-small"
    
    # Define SAE paths directly in main
    sae_folders = get_sae_folders(sae_paths, llm_id, site, layer, dict_size, topk, lr, seed, seq_len, steps, faithful, tiny, openweb, red, pile)
    print(f"Found {len(sae_folders)} SAEs to evaluate")
    
    print(f"Loading datasets (up to {num_sequences} sequences each)")
    all_datasets = load_datasets(tokenizer, max_samples=num_sequences)
    
    print(f"Successfully loaded {len(all_datasets)} datasets:")
    for dataset_name, dataset in all_datasets.items():
        print(f"  - {dataset_name}: {len(dataset)} sequences")
    
    results = {
        "train_dataset": [],
        "eval_dataset": [],
        "l2_loss": [],
        "explained_variance": [],
        "ce_diff": []
    }
    
    all_sae_models = {}
    for train_dataset, sae_folder in sae_folders.items():
        print(f"\nLoading SAE trained on {train_dataset}: {sae_folder}")
        
        sae, original = convert(sae_folder)
        llm = llm.to(original.cfg["dtype"])
        sae = sae.to(device)
        all_sae_models[train_dataset] = sae
        print(f"Successfully loaded SAE model from {sae_folder}")
    
    for train_dataset, sae in all_sae_models.items():
        print(f"\nEvaluating SAE trained on {train_dataset}")
        
        for eval_dataset_name, eval_dataset in all_datasets.items():
            print(f"Evaluating on {eval_dataset_name} dataset")
            metrics = evaluate_sae_on_dataset(
                sae, llm, eval_dataset, device, 
                layer_i=layer, max_sequences=num_sequences
            )
            
            results["train_dataset"].append(train_dataset)
            results["eval_dataset"].append(eval_dataset_name)
            results["l2_loss"].append(metrics["l2_loss"])
            results["explained_variance"].append(metrics["explained_variance"])
            results["ce_diff"].append(metrics["ce_diff"])
            
            print(f"Results: {metrics}")
    
    results_df = pd.DataFrame(results)
    
    results_path = os.path.join(results_folder, f"cross_metrics_layer{layer}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")
    
    create_visualizations(results_df, results_folder, layer, dict_size, topk, seed)
    
    print("Evaluation complete!")

def create_visualizations(df, results_folder, layer, dict_size, topk, seed):
    """Create simple visualizations of the cross-dataset metrics"""
    if df.empty:
        print("No data available for visualization")
        return
    
    # Create a single visualization file with all metrics
    plt.figure(figsize=(25, 7))
    
    # Create a matrix for cross-dataset comparisons
    train_datasets = sorted(df['train_dataset'].unique())
    eval_datasets = sorted(df['eval_dataset'].unique())
    
    metrics = ['explained_variance', 'l2_loss', 'ce_diff']
    titles = ['Explained Variance', 'L2 Loss', 'CE Difference']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        plt.subplot(1, 3, i+1)
        
        # Create a pivot table for this metric
        pivot = df.pivot(index='train_dataset', columns='eval_dataset', values=metric)
        
        # Apply appropriate colormaps based on the metric
        if metric == 'explained_variance':
            cmap = 'viridis_r'  # Higher is better
            vmin, vmax = 0.8, 1
        elif metric == 'l2_loss':
            cmap = 'rocket_r'  # Lower is better
            vmin, vmax = None, None
        else:  # ce_diff
            cmap = 'coolwarm'  # Center at 0
            vmin, vmax = None, None
        
        # Create heatmap
        sns.heatmap(pivot, annot=True, cmap=cmap, vmin=vmin, vmax=vmax, 
                    cbar_kws={'label': metric})
        plt.title(title)
        plt.tight_layout()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure
    viz_path = os.path.join(results_folder, f"cross_metrics_layer{layer}.png")
    plt.savefig(viz_path, bbox_inches="tight", dpi=150)
    print(f"Saved visualization to {viz_path}")
    plt.close()

if __name__ == "__main__":
    fire.Fire(main) 