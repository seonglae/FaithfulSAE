import json
import torch
from transformers import AutoTokenizer, AutoModel

# Configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B"
TEXT_KEY = "text"  # JSONL field containing text
BATCH_SIZE = 64

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()  # Set model to evaluation mode

def load_jsonl(file_path):
    """Load entire JSONL dataset"""
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if TEXT_KEY in data:
                texts.append(data[TEXT_KEY])
    return texts

def get_embedding(text):
    """Tokenizes text and extracts sentence embedding using mean pooling"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()  # Mean pooling

def compute_embedding_cosine(texts):
    """Compute cosine similarity for a large dataset efficiently"""
    embeddings = []
    
    # Process texts in batches to handle large dataset
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_embeddings = torch.stack([get_embedding(text) for text in batch])
        embeddings.append(batch_embeddings)
    
    embeddings = torch.cat(embeddings)
    
    # Normalize vectors
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    
    # Compute cosine similarity matrix
    cosine_sim_matrix = torch.mm(embeddings, embeddings.T)
    
    # Compute mean cosine similarity excluding self-comparisons
    num_samples = cosine_sim_matrix.shape[0]
    mean_cosine_sim = (cosine_sim_matrix.sum() - torch.trace(cosine_sim_matrix)) / (num_samples * (num_samples - 1))
    
    return mean_cosine_sim.item()

# Load dataset
dataset_texts = load_jsonl("llm_training_dataset_256ctx.jsonl")

# Compute similarity
embedding_similarity = compute_embedding_cosine(dataset_texts)
print("Embedding-Based Cosine Similarity (Full Dataset):", embedding_similarity)