import fire
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import torch
from tqdm import tqdm
import matplotlib as mpl
import statistics
from vllm import LLM, SamplingParams

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

def visualize_perplexity(position_perplexities, position_entropies=None):
    """
    Visualize token-position-wise perplexity and entropy.
    
    Args:
        position_perplexities: Dict with keys for avg, min, max perplexities by position
        position_entropies: Optional dict with keys for avg, min, max entropies by position
    """
    plt.figure(figsize=(15, 10))
    
    # Perplexity plot
    positions = list(position_perplexities['avg'].keys())
    avg_ppl = [position_perplexities['avg'][pos] for pos in positions]
    min_ppl = [position_perplexities['min'][pos] for pos in positions]
    max_ppl = [position_perplexities['max'][pos] for pos in positions]
    
    plt.subplot(2, 1, 1)
    plt.plot(positions, avg_ppl, 'b-', linewidth=2, label='Average Perplexity')
    plt.fill_between(positions, min_ppl, max_ppl, color='blue', alpha=0.2, label='Min-Max Range')
    
    plt.title('Token-Position Perplexity')
    plt.xlabel('Token Position')
    plt.ylabel('Perplexity')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # If entropy data is provided, plot it too
    if position_entropies:
        plt.subplot(2, 1, 2)
        positions = list(position_entropies['avg'].keys())
        avg_ent = [position_entropies['avg'][pos] for pos in positions]
        min_ent = [position_entropies['min'][pos] for pos in positions]
        max_ent = [position_entropies['max'][pos] for pos in positions]
        
        plt.plot(positions, avg_ent, 'r-', linewidth=2, label='Average Entropy')
        plt.fill_between(positions, min_ent, max_ent, color='red', alpha=0.2, label='Min-Max Range')
        
        plt.title('Token-Position Entropy')
        plt.xlabel('Token Position')
        plt.ylabel('Entropy (bits)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig("perplexity_analysis.png", dpi=300)
    print("Perplexity analysis graph saved to 'perplexity_analysis.png'")

def dist(model_name="EleutherAI/pythia-6.9b", dataset_name="seonglae/true-synthetic-pythia-6.9b", column="text", split="train", topk=10):
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

def calculate_entropy(probs):
    """Calculate entropy from a probability distribution"""
    nonzero_probs = probs[probs > 0]
    entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs))
    return entropy

def perplexity(model_name="EleutherAI/pythia-6.9b", dataset_name="seonglae/true-synthetic-pythia-6.9b", 
             column="text", split="train", batch_size=32, max_length=2048, 
             tensor_parallel_size=1, gpu_memory_utilization=0.8):
    """
    Analyze token-position perplexity using vLLM for faster inference
    
    Args:
        model_name: Name of the model to use
        dataset_name: Name of the dataset to analyze
        column: Column name containing text in the dataset
        split: Dataset split to use
        batch_size: Number of sequences to process in parallel
        max_length: Maximum sequence length to consider
        tensor_parallel_size: Number of GPUs to use for tensor parallelism
        gpu_memory_utilization: Fraction of GPU memory to use
    """
    print(f"Loading model {model_name} using vLLM...")
    # Initialize vLLM for faster inference
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_length,
    )
    
    # Load tokenizer separately to calculate token-level statistics
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loading dataset {dataset_name}...")
    ds = load_dataset(dataset_name, split=split)
    
    # Store perplexity statistics by position
    position_perplexities = {
        'avg': defaultdict(list),
        'min': defaultdict(float),
        'max': defaultdict(float),
    }
    
    # Also track entropy as an additional metric
    position_entropies = {
        'avg': defaultdict(list),
        'min': defaultdict(float),
        'max': defaultdict(float),
    }
    
    # Keep track of sequence lengths for overall statistics
    sequence_lengths = []
    total_perplexities = []
    
    # Process the dataset in batches
    total_examples = len(ds)
    
    for i in tqdm(range(0, total_examples, batch_size), desc="Calculating perplexity"):
        batch_end = min(i + batch_size, total_examples)
        batch = ds[i:batch_end]
        texts = batch[column]
        
        # Tokenize texts to get sequence lengths
        tokenized_texts = tokenizer(texts, return_tensors="pt", padding=True)
        input_ids = tokenized_texts.input_ids
        attention_mask = tokenized_texts.attention_mask
        
        # Get log probabilities from vLLM
        prompt_logprobs = []
        for j, text in enumerate(texts):
            # vLLM's logprobs API
            outputs = llm.generate(
                [text],
                SamplingParams(temperature=0.0, logprobs=tokenizer.vocab_size),
                use_tqdm=False,
            )
            
            token_logprobs = []
            for token_info in outputs[0].prompt_logprobs:
                if token_info and token_info.logprob is not None:
                    token_logprobs.append(token_info.logprob)
                else:
                    # Handle cases where logprob might be None for some tokens
                    token_logprobs.append(0.0)
            
            prompt_logprobs.append(token_logprobs)
        
        # Process sequence by sequence
        for j, (ids, mask, logprobs) in enumerate(zip(input_ids, attention_mask, prompt_logprobs)):
            # Get valid token positions (excluding padding)
            valid_positions = mask.bool().tolist()
            valid_ids = ids[valid_positions].tolist()
            
            if len(logprobs) < len(valid_positions):
                # Ensure logprobs has the same length
                logprobs = logprobs + [0.0] * (len(valid_positions) - len(logprobs))
            
            valid_logprobs = [lp for lp, is_valid in zip(logprobs, valid_positions) if is_valid]
            
            # Calculate sequence perplexity
            seq_perplexity = np.exp(-np.mean(valid_logprobs))
            total_perplexities.append(seq_perplexity)
            sequence_lengths.append(len(valid_ids))
            
            # Calculate token-position statistics
            for pos, (token_id, logprob) in enumerate(zip(valid_ids, valid_logprobs)):
                # Skip first token as it doesn't have a meaningful perplexity (no context)
                if pos == 0:
                    continue
                
                # Convert log probability to perplexity for this position
                token_perplexity = np.exp(-logprob)
                
                # Update position-wise statistics
                position_perplexities['avg'][pos].append(token_perplexity)
                position_perplexities['min'][pos] = min(position_perplexities['min'].get(pos, float('inf')), token_perplexity)
                position_perplexities['max'][pos] = max(position_perplexities['max'].get(pos, 0), token_perplexity)
                
                # Calculate and store entropy
                # For entropy, we would ideally use the full probability distribution
                # As an approximation, we'll use exponential of the logprobs to get probabilities
                # This is a simplified approach, as we don't have the full distribution from vLLM here
                token_entropy = -logprob / np.log(2)  # Convert to bits
                position_entropies['avg'][pos].append(token_entropy)
                position_entropies['min'][pos] = min(position_entropies['min'].get(pos, float('inf')), token_entropy)
                position_entropies['max'][pos] = max(position_entropies['max'].get(pos, 0), token_entropy)
    
    # Calculate average perplexity per position
    avg_position_perplexities = {
        'avg': {pos: statistics.mean(values) for pos, values in position_perplexities['avg'].items()},
        'min': position_perplexities['min'],
        'max': position_perplexities['max'],
    }
    
    # Calculate average entropy per position
    avg_position_entropies = {
        'avg': {pos: statistics.mean(values) for pos, values in position_entropies['avg'].items()},
        'min': position_entropies['min'],
        'max': position_entropies['max'],
    }
    
    # Print overall statistics
    print("\nPerplexity Analysis:")
    print(f"Average sequence length: {statistics.mean(sequence_lengths):.2f}")
    print(f"Average sequence perplexity: {statistics.mean(total_perplexities):.4f}")
    print(f"Min sequence perplexity: {min(total_perplexities):.4f}")
    print(f"Max sequence perplexity: {max(total_perplexities):.4f}")
    print(f"Std dev of perplexity: {statistics.stdev(total_perplexities):.4f}")
    
    # Calculate relative changes between consecutive positions
    positions = sorted(avg_position_perplexities['avg'].keys())
    if len(positions) > 1:
        rel_changes = []
        for i in range(1, len(positions)):
            prev_pos = positions[i-1]
            curr_pos = positions[i]
            prev_ppl = avg_position_perplexities['avg'][prev_pos]
            curr_ppl = avg_position_perplexities['avg'][curr_pos]
            rel_change = (curr_ppl - prev_ppl) / prev_ppl * 100  # percentage
            rel_changes.append(rel_change)
        
        print(f"Average relative change between positions: {statistics.mean(rel_changes):.2f}%")
        print(f"Max increase between consecutive positions: {max(rel_changes):.2f}%")
        print(f"Max decrease between consecutive positions: {min(rel_changes):.2f}%")
    
    # Visualize the results
    visualize_perplexity(avg_position_perplexities, avg_position_entropies)

if __name__ == '__main__':
    fire.Fire({
        'dist': dist,
        'perplexity': perplexity
    })
