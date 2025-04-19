from os import makedirs
from os.path import join, isfile, isdir
import torch
import fire
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from scipy.optimize import linear_sum_assignment
import umap
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE

from convert import convert
from cross_dataset_metrics import get_sae_folders

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else device

def load_sae(sae_id, local, site, device):
    # Load SAE from pretrained hub or local checkpoint
    if not local:
        sae, _, _ = SAE.from_pretrained(sae_id, site, device=device)
    else:
        sae, _ = convert(sae_id)
    return sae

def weight_sim(w1, w2, topk=4):
    return torch.stack([torch.nn.functional.cosine_similarity(w1, w2[i, :], dim=1).topk(topk).values.detach().cpu() for i in range(w2.shape[0])]).to(torch.float32)

def decoder_feature_sim(sae1, sae2, topk=4): 
    return weight_sim(sae1.W_dec, sae2.W_dec, topk)

def decoder_neuron_sim(sae1, sae2, topk=4): 
    return weight_sim(sae1.W_dec.T, sae2.W_dec.T, topk)

def encoder_feature_sim(sae1, sae2, topk=4): 
    return weight_sim(sae1.W_enc.T, sae2.W_enc.T, topk)

def encoder_neuron_sim(sae1, sae2, topk=4): 
    return weight_sim(sae1.W_enc, sae2.W_enc, topk)

def compute_similarity_matrix(w1, w2, batch_size=4096):
    """Compute pairwise cosine similarity matrix between two sets of weights."""
    # Normalize weights for cosine similarity
    w1_norm = w1 / w1.norm(dim=1, keepdim=True)
    w2_norm = w2 / w2.norm(dim=1, keepdim=True)
    
    n, m = w1_norm.shape[0], w2_norm.shape[0]
    sim_matrix = torch.zeros((n, m), device=w1.device)
    
    for i in range(0, n, batch_size):
        batch_end = min(i + batch_size, n)
        batch1 = w1_norm[i:batch_end]
        
        for j in range(0, m, batch_size):
            batch_end2 = min(j + batch_size, m)
            batch2 = w2_norm[j:batch_end2]
            
            # Compute normalized dot product for cosine similarity
            sim_batch = torch.matmul(batch1, batch2.T)
            sim_matrix[i:batch_end, j:batch_end2] = sim_batch
            
    return sim_matrix

def viz_dist(data, title, xlabel, ylabel='Activation', color='blue',
             log_scale=False, size=20, bins=100, save_path=None):
    plt.figure(figsize=(12, 6))
    ax = sns.histplot(data, bins=bins, kde=True, color=color, fill=True)
    for patch in ax.patches:
        patch.set_edgecolor("none")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if log_scale:
        plt.yscale('log')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def viz_umap(data, base_name, results_folder, folder_name='umap'):
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data)
    plt.figure(figsize=(12, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.7)
    plt.title(f'UMAP Projection: {base_name}')
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    save_file = join(results_folder, folder_name, f"{base_name}.png")
    plt.savefig(save_file)
    plt.close()

def viz_tsne(data, base_name, results_folder, perplexity, max_iter=5000, folder_name='tsne'):
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=max_iter, init='random', random_state=42)
    embedding = tsne.fit_transform(data)
    plt.figure(figsize=(12, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.7)
    plt.title(f'TSNE (perplexity={perplexity}): {base_name}')
    plt.xlabel('TSNE-1')
    plt.ylabel('TSNE-2')
    save_file = join(results_folder, folder_name, f"{base_name}-{perplexity}.png")
    plt.savefig(save_file)
    plt.close()

def feat_match(sae_paths="./checkpoints", llm_id="meta-llama/Llama-3.2-1B", site="resid_pre", layer=12, seeds=[42, 49], seq_len=512, lr=0.0002, pile=False, tiny=False, openweb=False, red=False,
               topk=48, dict_size=14336, num_sequences=100, steps=195311, faithful="faithful-llama3.2-1b", results_folder='results', local=True, threshold=0.7, batch_size=4096, k=4):
    """
    Compare SAE models pairwise and generate similarity distribution plots.
    Also computes the ratio of decoder feature top1 activations above the threshold.
    Saves:
      - Distribution plots under results/ef, results/en, results/df, results/dn
      - top1 ratio JSON under results/df/top1.json
      - UMAP plots under results/umap
    sae1_list and sae2_list: Comma-separated lists (or lists) of SAE identifiers.
    """
    sae1_list = get_sae_folders(sae_paths, llm_id, site, layer, dict_size, topk, lr, seeds[0], seq_len, steps, faithful, tiny, openweb, red, pile)
    sae1_list = list(sae1_list.values())
    sae2_list = get_sae_folders(sae_paths, llm_id, site, layer, dict_size, topk, lr, seeds[1], seq_len, steps, faithful, tiny, openweb, red, pile)
    sae2_list = list(sae2_list.values())

    # Create output folders if they don't exist
    makedirs(results_folder, exist_ok=True)
    makedirs(join(results_folder, 'ef'), exist_ok=True)
    makedirs(join(results_folder, 'en'), exist_ok=True)
    makedirs(join(results_folder, 'df'), exist_ok=True)
    makedirs(join(results_folder, 'dn'), exist_ok=True)
    makedirs(join(results_folder, 'umap'), exist_ok=True)
    
    # Process each pair of SAE models
    top1_ratios = {}
    hungarian_ratios = {}
    
    for sae_id1, sae_id2 in zip(sae1_list, sae2_list):
        print(f"Processing pair: {sae_id1} vs {sae_id2}")
        sae1 = load_sae(sae_id1, local, site, device)
        sae2 = load_sae(sae_id2, local, site, device)
        
        # Prepare config for visualization
        columns = [f"Top {i+1}" for i in range(k)]
        base_name = f"{sae_id1.split('/')[-1]}-{sae_id2.split('/')[-1]}"
        
        # Visualize distributions if input dimensions match
        if sae1.cfg.d_in == sae2.cfg.d_in:
            enc_feat = encoder_feature_sim(sae1, sae2)
            enc_feat_df = pd.DataFrame(enc_feat.numpy(), columns=columns)
            viz_dist(enc_feat_df, 'CosSim Distribution of Encoder Features', 
                     'Cosine Similarity', save_path=join(results_folder, 'ef', f"{base_name}.png"))
            
            # Calculate decoder feature similarity
            dec_feat = decoder_feature_sim(sae1, sae2)
            dec_feat_df = pd.DataFrame(dec_feat.numpy(), columns=columns)
            viz_dist(dec_feat_df, 'CosSim Distribution of Decoder Features', 
                     'Cosine Similarity', save_path=join(results_folder, 'df', f"{base_name}.png"))
            
            # Traditional top1 ratio calculation
            top1 = dec_feat[:, 0]
            ratio = (top1 > threshold).float().mean().item()
            top1_ratios[base_name] = ratio
            
            # Hungarian matching for optimal feature pairing
            try:
                # Compute full similarity matrix
                sim_matrix = compute_similarity_matrix(sae1.W_dec, sae2.W_dec, batch_size)
                
                # Find optimal assignment using Hungarian algorithm
                row_indices, col_indices = linear_sum_assignment(-sim_matrix.detach().cpu().numpy())  # Negate for maximization
                
                # Get matched similarities
                matched_sims = sim_matrix[row_indices, col_indices].detach().cpu()
                # Calculate ratio of matches above threshold
                hungarian_ratio = (matched_sims > threshold).float().mean().item()
                hungarian_ratios[base_name] = hungarian_ratio
                
                # Generate visualization of Hungarian matching
                plt.figure(figsize=(10, 6))
                sns.histplot(matched_sims.numpy(), bins=50, kde=True)
                plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
                plt.title(f"Hungarian Matched Feature Similarity: {base_name}")
                plt.xlabel("Cosine Similarity")
                plt.ylabel("Count")
                plt.legend()
                plt.savefig(join(results_folder, 'df', f"{base_name}_hungarian.png"))
                plt.close()
                
                print(f"Top1 ratio: {ratio:.4f}, Hungarian ratio: {hungarian_ratio:.4f}")
            except Exception as e:
                print(f"Error computing Hungarian matching: {e}")
                hungarian_ratios[base_name] = None
            
            # Save both ratio types
            json_path = join(results_folder, 'df', 'top1.json')
            with open(json_path, 'w') as f:
                json.dump(top1_ratios, f, indent=2)
                
            json_path = join(results_folder, 'df', 'hungarian.json')
            with open(json_path, 'w') as f:
                json.dump(hungarian_ratios, f, indent=2)

        if sae1.cfg.d_sae == sae2.cfg.d_sae:
            enc_neur = encoder_neuron_sim(sae1, sae2)
            enc_neur_df = pd.DataFrame(enc_neur.numpy(), columns=columns)
            viz_dist(enc_neur_df, 'CosSim Distribution of Encoder Neurons', 
                     'Cosine Similarity', save_path=join(results_folder, 'en', f"{base_name}.png"))
            dec_neur = decoder_neuron_sim(sae1, sae2)
            dec_neur_df = pd.DataFrame(dec_neur.numpy(), columns=columns)
            viz_dist(dec_neur_df, 'CosSim Distribution of Decoder Neurons', 
                     'Cosine Similarity', save_path=join(results_folder, 'dn', f"{base_name}.png"))

        # UMAP visualization
        sae_id1 = sae_id1.split('/')[-1]
        data = sae1.W_dec.to(torch.float32).detach().cpu().numpy()
        if not isfile(join(results_folder, 'umap', f"{sae_id1}.png")):
            viz_umap(data, sae_id1, results_folder)
        sae_id2 = sae_id2.split('/')[-1]
        data = sae2.W_dec.to(torch.float32).detach().cpu().numpy()
        if not isfile(join(results_folder, 'umap', f"{sae_id2}.png")):
            viz_umap(data, sae_id2, results_folder)

        print(f"Plots saved for pair: {base_name}")

if __name__ == '__main__':
    fire.Fire(feat_match)
