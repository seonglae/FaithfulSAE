import json
import os
import umap
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from vllm import LLM

# Model Parameters
MODEL_NAME = "google/gemma-2-2b"

# Configuration
TEXT_KEY = "text"  # JSONL field containing text
BATCH_SIZE = 10_000
SAMPLE_SIZE = 10_000
DATASET_PATH = "datasets"
RESULT_PATH = "results"
TEMPERATURES = [0.2, 0.4, 0.6, 0.8, 1.0] # List of temperatures
# TEMPERATURES = [1.0] # List of temperatures
# SEEDS = [0, 1, 2, 3, 4]  # List of seeds
SEEDS = [0]  # List of seeds
extra_datasets = [
    "roneneldan/TinyStories"
]  # Extra dataset(s) to compare with the synthetic datasets


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


def plot_umap(embeddings, labels, imagename="umap.png"):
    """Plots the umap."""
    reducer = umap.UMAP(n_components=2, random_state=42)  # 2D visualization
    all_embeddings = np.vstack(embeddings)
    reduced_embeddings = reducer.fit_transform(all_embeddings)
    plt.figure(figsize=(8, 6))
    start_index = 0
    for i, label in enumerate(labels):
        end_index = start_index + len(embeddings[i])
        plt.scatter(reduced_embeddings[start_index:end_index, 0], reduced_embeddings[start_index:end_index, 1], label=label, alpha=0.3, s=10)
        start_index = end_index
    plt.legend()
    plt.title("UMAP Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()

    # Save the image
    path = os.path.join(RESULT_PATH, imagename)
    plt.savefig(path)
    print(f"Saved {imagename}.")


if __name__ == "__main__":
    # Load model
    llm = LLM(model=MODEL_NAME, gpu_memory_utilization=0.95, task="embed")

    # Iteratively get the embeddings of each synthetic dataset
    labels = []
    embeddings = []
    for seed in SEEDS:
        for temp in TEMPERATURES:
            labels.append(f"SEED={seed}, TEMP={temp}")
            filename = f"llm-training-dataset-seed={seed}-temp={temp}.jsonl"
            dataset_path = os.path.join(DATASET_PATH, filename)
            print(f"Getting embedding for {dataset_path}")
            dataset = load_jsonl(dataset_path, sample_size=SAMPLE_SIZE)

            # Get embedding
            embedding = get_embedding(dataset)

            # Store the mean embedding
            embeddings.append(embedding)

    # Get the embeddings of each extra dataset
    for dataset_name in extra_datasets:
        labels.append(dataset_name.split("/")[-1])
        # Load the dataset
        dataset = load_dataset(dataset_name, split="train")
        print(f"Getting embedding for {dataset_name}")

        # Sample and get the list of dataset
        sampled_dataset = dataset[:SAMPLE_SIZE][TEXT_KEY]

        # Get embedding
        embedding = get_embedding(sampled_dataset)

        # Store the mean embedding
        embeddings.append(embedding)

    # Plot umap
    plot_umap(embeddings, labels, imagename="umap_temp_gemma-2-2b_10k.png")
