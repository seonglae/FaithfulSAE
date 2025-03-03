import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from vllm import LLM

# Model Parameters
MODEL_NAME = "meta-llama/Llama-3.1-8B"

# Configuration
TEXT_KEY = "text"  # JSONL field containing text
BATCH_SIZE = 128
SAMPLE_SIZE = 10_000
RESULT_PATH = "results"
S = 5  # Number of seeds
extra_datasets = [
    "roneneldan/TinyStories"
]  # Extra dataset(s) to compare with the synthetic datasets

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def load_jsonl(file_path, sample_size=-1):
    """Load JSONL dataset up to given sample_size"""
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if sample_size != -1 and i >= sample_size:
                break
            data = json.loads(line)
            if TEXT_KEY in data:
                texts.append(data[TEXT_KEY])
    return texts


def get_embedding(dataset):
    """Tokenizes text and extracts sentence embedding using mean pooling"""
    embeddings = []
    for i in range(0, len(dataset), BATCH_SIZE):
        batch = dataset[i : i + BATCH_SIZE]
        outputs = llm.embed(batch)  # Get embeddings for give batch
        embedding = [output.outputs.embedding for output in outputs]  # Extract embedding
        embeddings.append(np.array(embedding))  # Convert to NumPy array

    return np.vstack(embeddings)


def consine_similarity(vec1, vec2):
    """Computes the consine similarity between the given vectors"""
    dot_product = np.dot(vec1.flatten(), vec2.flatten())
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0


def plot_similarity_matrix(similarity_matrix, imagename="cosine_matrix.png"):
    """Plots the similarity matrix."""
    plt.figure(figsize=(8, 6))
    plt.imshow(similarity_matrix, cmap="coolwarm", interpolation="nearest")
    plt.colorbar()

    # Label axes
    dataset_names = [f"SEED={i}" for i in range(S)] + [
        name.split("/")[-1] for name in extra_datasets
    ]
    num_of_datasets = len(dataset_names)
    plt.xticks(range(num_of_datasets), dataset_names, rotation=45, ha="right")
    plt.yticks(range(num_of_datasets), dataset_names)
    plt.title("Dataset-to-Dataset Cosine Similarity Matrix")
    plt.xlabel("Dataset")
    plt.ylabel("Dataset")
    plt.tight_layout()

    # Display similarity values on the heatmap
    for i in range(num_of_datasets):
        for j in range(num_of_datasets):
            plt.text(
                j, i, f"{similarity_matrix[i, j]:.4f}", ha="center", va="center", color="black"
            )

    # Save the image
    path = os.path.join(RESULT_PATH, imagename)
    plt.savefig(path)
    print(f"Saved {path}.")


if __name__ == "__main__":
    # Load model
    llm = LLM(model=MODEL_NAME, gpu_memory_utilization=0.95, task="embed")

    # Iteratively get the embeddings of each synthetic dataset
    mean_embeddings = []
    for i in range(S):
        dataset_path = f"datasets/llm-training-dataset-1024ctx-seed={i}.jsonl"
        print(f"Getting embedding for {dataset_path}")
        dataset = load_jsonl(dataset_path, sample_size=SAMPLE_SIZE)

        # Get embedding
        embedding = get_embedding(dataset)

        # Compute mean embedding
        mean_embedding = np.mean(embedding, axis=0)

        # Store the mean embedding
        mean_embeddings.append(mean_embedding)

    # Get the embeddings of each extra dataset
    for dataset_name in extra_datasets:
        # Load the dataset
        dataset = load_dataset(dataset_name, split="train")
        print(f"Getting embedding for {dataset_name}")

        # Sample and get the list of dataset
        sampled_dataset = dataset[:SAMPLE_SIZE][TEXT_KEY]

        # Get embedding
        embedding = get_embedding(sampled_dataset)

        # Compute mean embedding
        mean_embedding = np.mean(embedding, axis=0)

        # Store the mean embedding
        mean_embeddings.append(mean_embedding)

    # Compute similarity matrix
    num_of_datasets = S + len(extra_datasets)
    similarity_matrix = np.zeros((num_of_datasets, num_of_datasets))

    for i in range(num_of_datasets):
        for j in range(num_of_datasets):
            similarity_matrix[i, j] = consine_similarity(mean_embeddings[i], mean_embeddings[j])

    # Plot similarity matrix
    plot_similarity_matrix(similarity_matrix, "cosine_matrix_llama.png")
