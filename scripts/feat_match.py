import random
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
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score as sklearn_f1_score
from collections import defaultdict, Counter as PythonCounter

from convert import convert
from cross_dataset_metrics import get_sae_folders

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else device

class LogisticProbe(torch.nn.Module):
    """Logistic regression probe for classification tasks"""
    def __init__(self, input_dim, num_classes, dtype=torch.float32):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, num_classes).to(device)
        self.linear = self.linear.to(dtype)
        
    def forward(self, x, labels=None):
        logits = self.linear(x)
        if labels is not None:
            return logits
        return logits

class Counter:
    """Custom counter for tracking predictions"""
    def __init__(self, num_classes):
        self.total = 0
        self.correct = 0
        self.predictions = []
        self.ground_truth = []
        
    def update(self, predictions):
        """Update counter with batch of predictions"""
        self.total += len(predictions[0])
        self.correct += predictions[1]
        self.predictions.extend(predictions[2])
        self.ground_truth.extend(predictions[3])
        
    def most_common(self, n):
        """Return most common elements (mocking PythonCounter's interface)"""
        if n == 1:
            return [("accuracy", self.correct)]
        return [("accuracy", self.correct)]

def loss_func(logits, labels):
    """Cross entropy loss for classification"""
    labels = labels.to(device)
    return torch.nn.functional.cross_entropy(logits, labels)

def correct_func(logits, labels):
    """Calculate prediction accuracy"""
    labels = labels.to(device)
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return (labels.cpu().tolist(), correct, preds.cpu().tolist(), labels.cpu().tolist())

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

def get_fake_features(llm_id, sae: SAE, n_tokens, threshold, dtype=torch.bfloat16, layer=12, batch_size=4):
    """
    Detect fake features that activate on random token sequences.
    
    A feature is considered "fake" if it fires on more than threshold% of out-of-distribution samples.
    
    Args:
        llm_id: ID of the language model
        sae: The SAE model
        n_tokens: Number of tokens to process
        threshold: Threshold for considering a feature as "fake"
        dtype: Data type for model
        layer: Model layer to use
        batch_size: Batch size for processing
        
    Returns:
        List of fake features with their activation frequencies
    """
    print("Detecting fake features...")
    llm = AutoModelForCausalLM.from_pretrained(llm_id, device_map="auto", torch_dtype=dtype).eval()
    tokenizer = AutoTokenizer.from_pretrained(llm_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Calculate sequences needed
    seq_len = sae.cfg.context_size
    n_seqs = max(1, n_tokens // seq_len)
    
    # Initialize counter for feature activations
    feature_counts = torch.zeros(sae.cfg.d_sae, dtype=torch.float32, device=device)
    vocab_list = list(tokenizer.get_vocab().values())
    total_sequences = 0
    
    with torch.no_grad():
        for batch_idx in tqdm(range(0, n_seqs, batch_size), desc="Inferencing OOD samples"):
            current_batch_size = min(batch_size, n_seqs - batch_idx)
            if current_batch_size <= 0:
                break
                
            # Generate random token sequences
            batch_tokens = []
            for _ in range(current_batch_size):
                # Create a sequence of random tokens
                random_tokens = [random.choice(vocab_list) for _ in range(seq_len)]
                random_tokens[0] = tokenizer.encode(tokenizer.bos_token)[0]
                batch_tokens.append(torch.tensor(random_tokens).to(device))
            
            # Tokenize and get model activations
            batch_tokens = torch.stack(batch_tokens)
            outputs = llm(batch_tokens, output_hidden_states=True)
            
            # Get activations from the specified layer
            hidden_states = outputs.hidden_states[layer]
            
            # Process each sequence in the batch
            for seq_idx in range(current_batch_size):
                # Exclude the first meaningless token
                activations = hidden_states[seq_idx, 1:]
                
                # Encode with SAE to get sparse features
                features = sae.encode(activations)
                
                # Binary indicator of feature firing (threshold at 1.0)
                binary_features = (features > 1.0).float()
                sum_binary_features = binary_features.sum(dim=0)

                # Accumulate counts
                feature_counts += sum_binary_features
                total_sequences += 1
    
    # Calculate the proportion of samples where each feature fired
    feature_proportions = feature_counts / n_tokens
        
    # Find fake features (those that fire on more than threshold of samples)
    fake_feature_mask = feature_proportions > threshold
    fake_feature_indices = torch.nonzero(fake_feature_mask).squeeze().cpu().tolist()
    
    # Handle the case of a single fake feature
    if not isinstance(fake_feature_indices, list):
        fake_feature_indices = [fake_feature_indices]
    
    # Create list of (feature_idx, activation_frequency) tuples
    fake_features = [(idx, feature_proportions[idx].item()) 
                   for idx in fake_feature_indices]
    
    # Sort by activation frequency (highest first)
    fake_features.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Found {len(fake_features)} fake features out of {sae.cfg.d_sae} total features")
    return fake_features

def prove_performance(llm_id, sae, layer, dataset_id, dtype=torch.bfloat16, batch_size=4, subset=None, test_split="test", train_split="train", dataset_labels=2, input_col="sentence", label_col="label", lr=0.001, dataset_limit=10000):
    """
    Evaluate performance of different feature representations on downstream classification tasks.
    
    Three approaches are compared:
    1. Baseline: Using the original hidden states
    2. SAE: Using the sparse expanded features
    3. Recon: Using the reconstructed features from SAE
    
    Args:
        llm_id: Language model ID
        sae: SAE model
        dtype: Data type for model
        layer: Model layer to use
        batch_size: Processing batch size
        dataset_id: HuggingFace dataset ID
        subset: Dataset subset name
        test_split: Test split name
        train_split: Train split name
        dataset_labels: Number of labels in dataset
        input_col: Input text column name
        label_col: Label column name
        lr: Learning rate for probing classifier
        
    Returns:
        Evaluation metrics for baseline, SAE, and reconstructed features
    """
    print(f"Evaluating downstream performance on {dataset_id}...")
    
    # Load model
    try:
        llm = AutoModelForCausalLM.from_pretrained(llm_id, device_map="auto", torch_dtype=dtype).eval()
        tokenizer = AutoTokenizer.from_pretrained(llm_id)
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Failed to load model: {e}")
        return 0, 0, 0, 0, 0, 0
        
    # Load dataset
    try:
        if subset:
            train_dataset = load_dataset(dataset_id, subset, split=train_split)
            test_dataset = load_dataset(dataset_id, subset, split=test_split)
        else:
            train_dataset = load_dataset(dataset_id, split=train_split)
            test_dataset = load_dataset(dataset_id, split=test_split)
            
        # Take a limited number of examples to speed up testing
        train_dataset = train_dataset.shuffle(seed=42).select(range(min(dataset_limit, len(train_dataset))))
        test_dataset = test_dataset.shuffle(seed=42).select(range(min(dataset_limit, len(test_dataset))))
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return 0, 0, 0, 0, 0, 0
        
    # Create data loaders
    def collate_fn(batch):
        texts = [item[input_col] for item in batch]
        labels = [item[label_col] for item in batch]
        return texts, torch.tensor(labels, dtype=torch.long)
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Initialize probing classifiers
    baseline_probe = LogisticProbe(sae.cfg.d_in, dataset_labels, dtype=dtype)
    sae_probe = LogisticProbe(sae.cfg.d_sae, dataset_labels, dtype=dtype)
    recon_probe = LogisticProbe(sae.cfg.d_in, dataset_labels, dtype=dtype)
    
    # Initialize optimizers
    baseline_optimizer = torch.optim.Adam(baseline_probe.parameters(), lr=lr)
    sae_optimizer = torch.optim.Adam(sae_probe.parameters(), lr=lr)
    recon_optimizer = torch.optim.Adam(recon_probe.parameters(), lr=lr)
    
    # Training loop
    baseline_probe.train()
    sae_probe.train()
    recon_probe.train()
    for texts, labels in tqdm(train_loader, desc="Training"):
        # Tokenize inputs
        encoded_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=llm.config.max_position_embeddings).to(device)
        
        # Get model activations
        with torch.no_grad():
            outputs = llm(encoded_inputs.input_ids, output_hidden_states=True)
            
            # Get last token representation for each sequence
            seq_lengths = encoded_inputs.attention_mask.sum(dim=1) - 1  # -1 for 0-indexed
            batch_size = encoded_inputs.input_ids.shape[0]
            
            # Extract activations from specified layer
            activations = outputs.hidden_states[layer]
            features = sae.encode(activations)
            reconstructions = sae.decode(features)

            activations = torch.stack([activations[i, -seq_lengths[i]:].mean(dim=0) for i in range(batch_size)])
            features = torch.stack([features[i, -seq_lengths[i]:].mean(dim=0) for i in range(batch_size)])
            reconstructions = torch.stack([reconstructions[i, -seq_lengths[i]:].mean(dim=0) for i in range(batch_size)])
        
        # Baseline probe
        baseline_optimizer.zero_grad()
        baseline_logits = baseline_probe(activations, labels)
        baseline_loss = loss_func(baseline_logits, labels)
        baseline_loss.backward()
        baseline_optimizer.step()
        
        # SAE probe
        sae_optimizer.zero_grad()
        sae_logits = sae_probe(features, labels)
        sae_loss = loss_func(sae_logits, labels)
        sae_loss.backward()
        sae_optimizer.step()
        
        # Reconstruction probe
        recon_optimizer.zero_grad()
        recon_logits = recon_probe(reconstructions, labels)
        recon_loss = loss_func(recon_logits, labels)
        recon_loss.backward()
        recon_optimizer.step()
    
    # Evaluation
    baseline_probe.eval()
    sae_probe.eval()
    recon_probe.eval()
    
    baseline_counter = Counter(dataset_labels)
    sae_counter = Counter(dataset_labels)
    recon_counter = Counter(dataset_labels)
    
    all_baseline_preds = []
    all_sae_preds = []
    all_recon_preds = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels in tqdm(test_loader, desc="Evaluating"):
            # Tokenize inputs
            encoded_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=llm.config.max_position_embeddings).to(device)
            
            # Get model activations
            outputs = llm(encoded_inputs.input_ids, output_hidden_states=True)
            
            # Get sequence and batch length
            seq_lengths = encoded_inputs.attention_mask.sum(dim=1)  # -1 for 0-indexed
            batch_size = encoded_inputs.input_ids.shape[0]
            
            # Extract activations from specified layer
            activations = outputs.hidden_states[layer]
            
            # Get SAE features and reconstructions
            features = sae.encode(activations)
            reconstructions = sae.decode(features)
            
            activations = torch.stack([activations[i, -seq_lengths[i]:].mean(dim=0) for i in range(batch_size)])
            features = torch.stack([features[i, -seq_lengths[i]:].mean(dim=0) for i in range(batch_size)])
            reconstructions = torch.stack([reconstructions[i, -seq_lengths[i]:].mean(dim=0) for i in range(batch_size)])
        
            baseline_logits = baseline_probe(activations, labels)
            sae_logits = sae_probe(features, labels)
            recon_logits = recon_probe(reconstructions, labels)
            
            # Update counters
            baseline_correct = correct_func(baseline_logits, labels)
            sae_correct = correct_func(sae_logits, labels)
            recon_correct = correct_func(recon_logits, labels)
            
            baseline_counter.update(baseline_correct)
            sae_counter.update(sae_correct)
            recon_counter.update(recon_correct)
            
            # Store predictions and labels for F1 score
            all_baseline_preds.extend(baseline_correct[2])
            all_sae_preds.extend(sae_correct[2])
            all_recon_preds.extend(recon_correct[2])
            all_labels.extend(labels.cpu().tolist())
    
    # Calculate accuracy
    baseline_acc = baseline_counter.correct / baseline_counter.total if baseline_counter.total > 0 else 0
    sae_acc = sae_counter.correct / sae_counter.total if sae_counter.total > 0 else 0
    recon_acc = recon_counter.correct / recon_counter.total if recon_counter.total > 0 else 0
    
    # Calculate F1 scores
    baseline_f1 = sklearn_f1_score(all_labels, all_baseline_preds, average='macro')
    sae_f1 = sklearn_f1_score(all_labels, all_sae_preds, average='macro')
    recon_f1 = sklearn_f1_score(all_labels, all_recon_preds, average='macro')
    
    print(f"Evaluation results on {dataset_id}:")
    print(f"  Baseline: Acc={baseline_acc:.4f}, F1={baseline_f1:.4f}")
    print(f"  SAE: Acc={sae_acc:.4f}, F1={sae_f1:.4f}")
    print(f"  Recon: Acc={recon_acc:.4f}, F1={recon_f1:.4f}")
    
    return baseline_acc, sae_acc, recon_acc, baseline_f1, sae_f1, recon_f1

def feat_match(sae_paths="./checkpoints", llm_id="meta-llama/Llama-3.2-1B", site="resid_pre", layer=12, seeds=[42, 49], seq_len=512, lr=0.0002, pile=False, tiny=False, openweb=False, red=False,
               topk=48, dict_size=14336, num_sequences=100, steps=195311, faithful="faithful-llama3.2-1b", results_folder='results', local=True, threshold=0.7, batch_size=4096, k=4,
               supernatural=False, alpaca=False, openinstruct=False, additional=None, flan=False, fine=False,
               match=False, fake=False, downstream=False, viz=False):
    """
    Compare SAE models pairwise and generate similarity distribution plots.
    Also computes the ratio of decoder feature top1 activations above the threshold.
    Saves:
      - Distribution plots under results/ef, results/en, results/df, results/dn
      - top1 ratio JSON under results/df/top1.json
      - UMAP plots under results/umap
    sae1_list and sae2_list: Comma-separated lists (or lists) of SAE identifiers.
    """
    sae1_list = get_sae_folders(sae_paths, llm_id, site, layer, dict_size, topk, lr, seeds[0], seq_len, steps, faithful, tiny, openweb, red, pile, supernatural, alpaca, openinstruct, additional, flan, fine=fine)
    sae1_list = list(sae1_list.values())
    sae2_list = get_sae_folders(sae_paths, llm_id, site, layer, dict_size, topk, lr, seeds[1], seq_len, steps, faithful, tiny, openweb, red, pile, supernatural, alpaca, openinstruct, additional, flan, fine=fine)
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
    fake_feature_ratios = {}
    downstream_prove_acc = {}
    
    for sae_id1, sae_id2 in zip(sae1_list, sae2_list):
        print(f"Processing pair: {sae_id1} vs {sae_id2}")
        sae1 = load_sae(sae_id1, local, site, device)
        sae2 = load_sae(sae_id2, local, site, device)
        
        # Prepare config for visualization
        columns = [f"Top {i+1}" for i in range(k)]
        base_name = f"{sae_id1.split('/')[-1]}-{sae_id2.split('/')[-1]}"
        
        # Visualize distributions if input dimensions match
        if sae1.cfg.d_in == sae2.cfg.d_in and match:
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
                if viz:
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

        if sae1.cfg.d_sae == sae2.cfg.d_sae and viz:
            enc_neur = encoder_neuron_sim(sae1, sae2)
            enc_neur_df = pd.DataFrame(enc_neur.numpy(), columns=columns)
            viz_dist(enc_neur_df, 'CosSim Distribution of Encoder Neurons', 
                     'Cosine Similarity', save_path=join(results_folder, 'en', f"{base_name}.png"))
            dec_neur = decoder_neuron_sim(sae1, sae2)
            dec_neur_df = pd.DataFrame(dec_neur.numpy(), columns=columns)
            viz_dist(dec_neur_df, 'CosSim Distribution of Decoder Neurons', 
                     'Cosine Similarity', save_path=join(results_folder, 'dn', f"{base_name}.png"))

        if viz:
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

        # Fake Feature Analysis
        if fake:
            print(f"\n==== Analyzing Fake Features for {base_name} ====")
            try:
                fake_features_1 = get_fake_features(llm_id, sae1, n_tokens=100_000_0, threshold=0.1, dtype=torch.float32 if "gpt2" in llm_id else torch.bfloat16, layer=layer, batch_size=4)
                fake_features_2 = get_fake_features(llm_id, sae2, n_tokens=100_000_0, threshold=0.1, dtype=torch.float32 if "gpt2" in llm_id else torch.bfloat16, layer=layer, batch_size=4)

                fake_feature_count1 = len(fake_features_1)
                fake_feature_count2 = len(fake_features_2)
                fake_feature_count1_ratio = fake_feature_count1 / sae1.cfg.d_sae
                fake_feature_count2_ratio = fake_feature_count2 / sae2.cfg.d_sae

                print(f"SAE 1 fake features: {fake_feature_count1}/{sae1.cfg.d_sae} ({fake_feature_count1_ratio:.2%})")
                print(f"SAE 2 fake features: {fake_feature_count2}/{sae2.cfg.d_sae} ({fake_feature_count2_ratio:.2%})")

                fake_feature_ratios[base_name] = {
                    "avg_fake_feature_ratio": (fake_feature_count1_ratio + fake_feature_count2_ratio) / 2,
                    "avg_fake_feature_count1": (fake_feature_count1 + fake_feature_count2) / 2,
                    "fake_feature_count1": fake_feature_count1,
                    "fake_feature_count2": fake_feature_count2,
                    "fake_feature_ratio1": fake_feature_count1_ratio,
                    "fake_feature_ratio2": fake_feature_count2_ratio,
                    "d_sae1": sae1.cfg.d_sae,
                    "d_sae2": sae2.cfg.d_sae,
                }

                # Save intermediate results
                json_path = join(results_folder, 'df', 'fake_features.json')
                with open(json_path, 'w') as f:
                    json.dump(fake_feature_ratios, f, indent=2)

            except Exception as e:
                print(f"Error in fake feature analysis: {e}")
                fake_feature_ratios[base_name] = {"error": str(e)}


        # Downstream Prove Performance
        if downstream:
            print(f"\n==== Evaluating Downstream Performance for {base_name} ====")
            downstream_prove_acc[base_name] = {}
            
            classification_datasets = [
                {"dataset_id": "stanfordnlp/sst2", "subset": None, "test_split": "validation", "train_split": "train", "dataset_labels": 2, "input_col": "sentence", "label_col": "label"},
                {"dataset_id": "nyu-mll/glue", "subset": "cola", "test_split": "validation", "train_split": "train", "dataset_labels": 2, "input_col": "sentence", "label_col": "label"},
                {"dataset_id": "yelp_polarity", "subset": None, "test_split": "test", "train_split": "train", "dataset_labels": 2, "input_col": "text", "label_col": "label"},
            ]
            
            for dataset_cfg in classification_datasets:
                dataset_name = f"{dataset_cfg['dataset_id']}"
                if dataset_cfg["subset"]:
                    dataset_name += f"/{dataset_cfg['subset']}"
                    
                print(f"\nEvaluating {dataset_name}...")
                try:
                    baseline_acc1, sae_acc1, recon_acc1, baseline_f11, sae_f11, recon_f11 = prove_performance(
                        llm_id, sae1, dtype=torch.float32 if "gpt2" in llm_id else torch.bfloat16, layer=layer, batch_size=4, **dataset_cfg
                    )
    
                    baseline_acc2, sae_acc2, recon_acc2, baseline_f12, sae_f12, recon_f12 = prove_performance(
                        llm_id, sae2, dtype=torch.float32 if "gpt2" in llm_id else torch.bfloat16, layer=layer, batch_size=4, **dataset_cfg
                    )
                    
                    baseline_acc = (baseline_acc1 + baseline_acc2) / 2
                    sae_acc = (sae_acc1 + sae_acc2) / 2
                    recon_acc = (recon_acc1 + recon_acc2) / 2
                    baseline_f1 = (baseline_f11 + baseline_f12) / 2
                    sae_f1 = (sae_f11 + sae_f12) / 2
                    recon_f1 = (recon_f11 + recon_f12) / 2
                    
                    downstream_prove_acc[base_name][dataset_name] = {
                        "baseline_acc": baseline_acc,
                        "sae_acc": sae_acc,
                        "recon_acc": recon_acc,
                        "baseline_f1": baseline_f1,
                        "sae_f1": sae_f1,
                        "recon_f1": recon_f1,
                        "baseline_acc1": baseline_acc1,
                        "sae_acc1": sae_acc1,
                        "recon_acc1": recon_acc1,
                        "baseline_f11": baseline_f11,
                        "sae_f11": sae_f11,
                        "recon_f11": recon_f11,
                        "baseline_acc2": baseline_acc2,
                    }
                    
                    # Save intermediate results after each dataset
                    json_path = join(results_folder, 'df', 'downstream_prove_acc.json')
                    with open(json_path, 'w') as f:
                        json.dump(downstream_prove_acc, f, indent=2)
                        
                except Exception as e:
                    print(f"Error evaluating {dataset_name}: {e}")
                    downstream_prove_acc[base_name][dataset_name] = {"error": str(e)}

if __name__ == '__main__':
    fire.Fire(feat_match)
