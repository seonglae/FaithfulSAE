import fire
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import torch
from tqdm import tqdm

def visualize_distributions(model_dist, dataset_dist, model_top_tokens, model_top_prob, 
                           dataset_top_tokens, dataset_top_prob, topk, vocab_size):
    plt.figure(figsize=(20, 16))
    
    plt.subplot(2, 2, 1)
    sns.barplot(x=model_top_tokens, y=model_top_prob, palette="magma")
    plt.title(f"Model Top-{topk}")
    plt.xlabel("Tokens")
    plt.ylabel("Probability")
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 2)
    sns.barplot(x=dataset_top_tokens, y=dataset_top_prob, palette="magma")
    plt.title(f"Dataset Top-{topk}")
    plt.xlabel("Tokens")
    plt.ylabel("Probability")
    plt.xticks(rotation=45)
    
    cmap = plt.cm.viridis
    
    ax3 = plt.subplot(2, 2, 3)
    sorted_model = np.sort(model_dist)[::-1]
    nonzero = sorted_model > 0
    x_indices = np.arange(len(sorted_model))[nonzero]
    y_values = sorted_model[nonzero]
    
    max_index = vocab_size
    colors = [cmap(idx/max_index) for idx in x_indices]
    
    for i in range(len(x_indices)-1):
        plt.fill_between(x_indices[i:i+2], y_values[i:i+2], color=colors[i], alpha=0.7)
    plt.plot(x_indices, y_values, color='black', linewidth=0.5, alpha=0.5)
    
    plt.title("Model Full Distribution")
    plt.xlabel("Index (Sorted)")
    plt.ylabel("Probability")
    plt.yscale('log')
    
    sm1 = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_index))
    sm1.set_array([])
    cbar1 = plt.colorbar(sm1, ax=ax3)
    cbar1.set_label(f'Index Position', fontsize=10)
    
    ax4 = plt.subplot(2, 2, 4)
    sorted_dataset = np.sort(dataset_dist)[::-1]
    nonzero = sorted_dataset > 0
    x_indices = np.arange(len(sorted_dataset))[nonzero]
    y_values = sorted_dataset[nonzero]
    
    colors = [cmap(idx/max_index) for idx in x_indices]
    
    for i in range(len(x_indices)-1):
        plt.fill_between(x_indices[i:i+2], y_values[i:i+2], color=colors[i], alpha=0.7)
    plt.plot(x_indices, y_values, color='black', linewidth=0.5, alpha=0.5)
    
    plt.title("Dataset Full Distribution")
    plt.xlabel("Index (Sorted)")
    plt.ylabel("Probability")
    plt.yscale('log')
    
    sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_index))
    sm2.set_array([])
    cbar2 = plt.colorbar(sm2, ax=ax4)
    cbar2.set_label(f'Index Position', fontsize=10)
    
    plt.tight_layout()
    plt.savefig("token_distribution_analysis.png", dpi=300)
    print("Graph saved to 'token_distribution_analysis.png'")


def dist(model_name="google/gemma-2-2b", dataset_name="seonglae/faithful-gemma2-2b", column="text", split="train", topk=10):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    
    ds = load_dataset(dataset_name, split=split)
    
    total_tokens = 0
    total_examples = 0
    mean = 0.0
    m2 = 0.0
    min_tokens = None
    max_tokens = None
    first_tokens = []
    all_tokens = []
    batch_size = 16384
    total_examples_count = len(ds)
    
    print("Analyzing dataset token statistics...")
    for i in tqdm(range(0, total_examples_count, batch_size), desc="Processing dataset"):
        batch_end = min(i + batch_size, total_examples_count)
        batch = ds[i:batch_end]
        texts = batch[column]
        batch_encodings = tokenizer(texts, return_attention_mask=False)
        batch_token_counts = [len(ids) for ids in batch_encodings['input_ids']]
        batch_total_tokens = sum(batch_token_counts)
        total_tokens += batch_total_tokens
        for token_count in batch_token_counts:
            total_examples += 1
            delta = token_count - mean
            mean += delta / total_examples
            delta2 = token_count - mean
            m2 += delta * delta2
        batch_min = min(batch_token_counts) if batch_token_counts else None
        batch_max = max(batch_token_counts) if batch_token_counts else None
        min_tokens = batch_min if min_tokens is None else min(min_tokens, batch_min) if batch_min is not None else min_tokens
        max_tokens = batch_max if max_tokens is None else max(max_tokens, batch_max) if batch_max is not None else max_tokens
        
        for ids in batch_encodings['input_ids']:
            if len(ids) > 0:
                first_tokens.append(ids[0])
                all_tokens.extend(ids)
    
    variance = m2 / (total_examples - 1) if total_examples > 1 else 0
    print("\nToken Statistics:")
    print("Total sequences:", total_examples)
    print("Total tokens:", total_tokens)
    print("Unique tokens used in all positions:", len(set(all_tokens)))
    print("Unique tokens in first position:", len(set(first_tokens)))
    print("Mean token count:", mean)
    print("Token count std:", variance ** 0.5)
    print("Min sequence length:", min_tokens)
    print("Max sequence length:", max_tokens)
    
    print("\nCalculating distributions...")
    
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        bos_token_id = tokenizer.cls_token_id or tokenizer.eos_token_id
    
    input_ids = torch.tensor([[bos_token_id]])
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, 0, :]
    
    logits_np = logits.cpu().numpy()
    exp_logits = np.exp(logits_np - np.max(logits_np))
    model_probs = exp_logits / np.sum(exp_logits)
    
    counter = Counter(first_tokens)
    total = len(first_tokens)
    
    vocab_size = tokenizer.vocab_size
    
    model_dist = np.zeros(vocab_size)
    model_dist[:len(model_probs)] = model_probs[:vocab_size]
    
    dataset_dist = np.zeros(vocab_size)
    for token_id, count in counter.items():
        if token_id < vocab_size:
            dataset_dist[token_id] = count / total
    
    model_top_idx = np.argsort(-model_dist)[:topk]
    model_top_prob = model_dist[model_top_idx]
    model_top_tokens = [tokenizer.decode([idx]) for idx in model_top_idx]
    
    dataset_top_idx = np.argsort(-dataset_dist)[:topk]
    dataset_top_prob = dataset_dist[dataset_top_idx]
    dataset_top_tokens = [tokenizer.decode([idx]) for idx in dataset_top_idx]
    
    eps = 1e-10
    kl1 = np.sum(np.where(model_dist > 0, model_dist * np.log((model_dist + eps) / (dataset_dist + eps)), 0))
    kl2 = np.sum(np.where(dataset_dist > 0, dataset_dist * np.log((dataset_dist + eps) / (model_dist + eps)), 0))
    kl_sym = (kl1 + kl2) / 2
    cross_entropy = -np.sum(model_dist * np.log(dataset_dist + eps))
    
    print("\nDistribution Analysis:")
    print(f"Tokenizer vocabulary size: {vocab_size}")
    print(f"Unique tokens in dataset first position: {len(counter)}")
    print(f"KL(Model -> Dataset): {kl1:.4f}")
    print(f"KL(Dataset -> Model): {kl2:.4f}")
    print(f"Jensen-Shannon divergence: {kl_sym:.4f}")
    print(f"Cross-entropy: {cross_entropy:.4f}")
    
    visualize_distributions(model_dist, dataset_dist, model_top_tokens, model_top_prob, 
                           dataset_top_tokens, dataset_top_prob, topk, vocab_size)

if __name__ == '__main__':
    fire.Fire({
        'dist': dist
    })
