import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from vllm import LLM
from tqdm import tqdm
from datasets import load_dataset

# Model Parameters
MODEL_NAME = "meta-llama/Llama-3.1-8B"

# Configuration
TEXT_KEY = "text"  # JSONL field containing text
BATCH_SIZE = 10_000
SAMPLE_SIZE = 10_000
DATASET_PATH = "datasets"
RESULT_PATH = "results/cosine_matrix"
SEED = 42  # Random generator seed

# Dataset Parameters
SEEDS = [0]  # List of seeds
TEMPERATURES = [0.2, 0.4, 0.6, 0.8, 1.0]  # List of temperatures
TOP_PS = [1.0] # List of top_ps
extra_datasets = [
    ["roneneldan/TinyStories"],
    ["HuggingFaceFW/fineweb", "sample-10BT"],
]  # Extra dataset(s) to compare with the synthetic datasets


def load_jsonl(file_path, sample_size):
    """Load JSONL dataset up to given sample_size"""
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if TEXT_KEY in data:
                texts.append(data[TEXT_KEY])

    # Randomly sample sample_size
    random.seed(SEED)
    sampled_texts = random.sample(texts, sample_size)
    return sampled_texts


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


def plot_similarity_matrix(similarity_matrix, labels, imagename="cosine_matrix.png"):
    """Plots the similarity matrix."""
    plt.figure(figsize=(8, 6))
    plt.imshow(similarity_matrix, cmap="coolwarm", interpolation="nearest")
    plt.colorbar()

    # Label axes
    num_of_datasets = len(labels)
    plt.xticks(range(num_of_datasets), labels, rotation=45, ha="right")
    plt.yticks(range(num_of_datasets), labels)
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

    # Create result directory if not exits
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)

    # Save the image
    path = os.path.join(RESULT_PATH, imagename)
    plt.savefig(path)
    print(f"Saved {path}.")


if __name__ == "__main__":
    # Load model
    llm = LLM(model=MODEL_NAME, gpu_memory_utilization=0.95, max_model_len=2048, task="embed")

    # Initiliase the label
    labels = []

    # Iteratively get the embeddings of each synthetic dataset
    mean_embeddings = []
    # Load dataset with different temperatures
    for seed in SEEDS:
        for temp in TEMPERATURES:
            # Load dataset with different top_ps
            for top_p in TOP_PS:
                # Set the label
                labels.append(f"s-{seed},t={temp},p={top_p}")

                # Construct the dataset path
                filename = f"seed={seed}-temp={temp}-top_p={top_p}.jsonl"
                dataset_path = os.path.join(DATASET_PATH, filename)
                print(f"Getting embeddings for {dataset_path}")

                # Load the dataset
                dataset = load_jsonl(dataset_path, sample_size=SAMPLE_SIZE)

                # Get embedding
                embedding = get_embedding(dataset)

                # Compute mean embedding
                mean_embedding = np.mean(embedding, axis=0)

                # Store the mean embedding
                mean_embeddings.append(mean_embedding)

    # Get the embeddings of each extra dataset
    for dataset_property in extra_datasets:
        # Load the dataset
        dataset_path = dataset_property[0]
        print(f"Getting embeddings for {dataset_path}")
        labels.append(dataset_path.split("/")[-1])
        if len(dataset_property) == 1:
            dataset = load_dataset(dataset_path, split="train", streaming=True)
        else:
            dataset_name = dataset_property[1]
            dataset = load_dataset(dataset_path, name=dataset_name, split="train", streaming=True)

        # Shuffle the dataset
        shuffled_dataset = dataset.shuffle(seed=SEED)

        # Sample and get the list of dataset
        sampled_dataset = []
        for _ in tqdm(range(SAMPLE_SIZE), desc="Sampled data"):
            data = next(iter(shuffled_dataset))
            sampled_dataset.append(data[TEXT_KEY])

        # Get embedding
        embedding = get_embedding(sampled_dataset)

        # Compute mean embedding
        mean_embedding = np.mean(embedding, axis=0)

        # Store the mean embedding
        mean_embeddings.append(mean_embedding)

    # Compute similarity matrix
    num_of_datasets = len(mean_embeddings)
    similarity_matrix = np.zeros((num_of_datasets, num_of_datasets))

    for i in range(num_of_datasets):
        for j in range(num_of_datasets):
            similarity_matrix[i, j] = consine_similarity(mean_embeddings[i], mean_embeddings[j])

    # Plot similarity matrix
    plot_similarity_matrix(similarity_matrix, labels, "cosine_matrix_temp.png")
