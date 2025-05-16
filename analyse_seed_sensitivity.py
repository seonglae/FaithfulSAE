import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import pandas as pd
from itertools import combinations
from sparsify.sparsify import Sae

from constants import PYTHIA_1_4B


def get_normalized_weights(sae, use_decoder=True):
    """Extract and normalize weights from SAE model."""
    if use_decoder:
        weights = sae.W_dec.data
    else:
        weights = sae.encoder.weight.data
    
    return weights / weights.norm(dim=1, keepdim=True)


def compute_similarity_matrix(base_weights, other_weights, batch_size=4096):
    """Compute similarity matrix between two sets of weights using batched processing."""
    n_batches = (base_weights.shape[0] + batch_size - 1) // batch_size
    cost_matrix = torch.zeros(base_weights.shape[0], other_weights.shape[0], device="cpu")
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, base_weights.shape[0])
        similarity = base_weights[start_idx:end_idx] @ other_weights.T
        cost_matrix[start_idx:end_idx] = similarity.cpu()
    
    return torch.nan_to_num(cost_matrix, nan=0)


def save_results(scores, indices, run_name, use_decoder=True):
    """Save results to CSV and pickle files."""
    df = pd.DataFrame(scores)
    mode = "decoder" if use_decoder else "encoder"
    file_name_scores = f"scores_{mode}_{run_name}_all_seeds.csv"
    
    df.to_csv(file_name_scores)
    return file_name_scores


def analyze_seed_sensitivity(
    target_models=["pythia-1.4b"],
    source_models=["1.4b"],
    layers=[8, 16],
    seeds=[0, 1],
    device="cuda",
    use_decoder=True,
    batch_size=4096
):
    """
    Analyze sensitivity between different seeds for SAE models.
    
    Args:
        target_models: List of target model names
        source_models: List of source model names
        layers: List of layer indices to analyze
        seeds: List of seeds to compare
        device: Device to use for computation
        use_decoder: Whether to use decoder weights (True) or encoder weights (False)
        batch_size: Batch size for similarity computation
    """
    scores = []
    indices = []
    
    for target_model in target_models:
        for source_model in source_models:
            checkpoint_name = f"{target_model}_from_{source_model}"
            
            for seed1, seed2 in combinations(seeds, 2):
                for layer in tqdm(layers):
                    # Create run name for this comparison
                    run_name = f"{checkpoint_name}_seed_{seed1}_vs_{seed2}_layer_{layer}"
                    print(f"Processing: {run_name}")
                    
                    # Load SAE models
                    sae_dir1 = f"checkpoints/{checkpoint_name}_seed_{seed1}/gpt_neox.layers.{layer}"
                    sae_dir2 = f"checkpoints/{checkpoint_name}_seed_{seed2}/gpt_neox.layers.{layer}"
                    
                    sae1 = Sae.load_from_disk(sae_dir1, device=device)
                    sae2 = Sae.load_from_disk(sae_dir2, device=device)
                    
                    # Get normalized weights
                    sae1_weights = get_normalized_weights(sae1, use_decoder)
                    sae2_weights = get_normalized_weights(sae2, use_decoder)
                    
                    # Compute similarity matrix
                    cost_matrix = compute_similarity_matrix(sae1_weights, sae2_weights, batch_size)
                    
                    # Find optimal assignment
                    row_indices, col_indices = linear_sum_assignment(cost_matrix.numpy(), maximize=True)
                    
                    # Compute and store mean similarity score
                    mean_similarity = cost_matrix[row_indices, col_indices].mean().item()
                    
                    # Save results
                    score_dict = {run_name: mean_similarity}
                    index_dict = {run_name: (row_indices, col_indices)}
                    
                    print(f"{run_name}: {mean_similarity}")
                    scores.append(score_dict)
                    indices.append(index_dict)
                    
                    # Save results for this comparison
                    save_results(scores, indices, run_name, use_decoder)
    
    return scores, indices


if __name__ == "__main__":
    # Configuration
    config = {
        "target_models": [PYTHIA_1_4B],
        "source_models": [PYTHIA_1_4B],
        "layers": [8, 16],
        "seeds": [0, 1],  # Add all your seeds here
        "device": "cuda",
        "use_decoder": True,
        "batch_size": 4096  # Adjust based on available memory
    }
    
    # Run analysis
    scores, indices = analyze_seed_sensitivity(**config)
