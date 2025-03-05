import os
import json
import umap
import pacmap
import numpy as np
import matplotlib.pyplot as plt
from vllm import LLM
from tqdm import tqdm
from datasets import load_dataset
from sklearn.decomposition import PCA

# Model Parameters
MODEL_NAME = "meta-llama/Llama-3.1-8B"

# Configuration
TEXT_KEY = "text"  # JSONL field containing text
BATCH_SIZE = 1_000
SAMPLE_SIZE = 10_000
REDUCE_METHOD = "pca"
DATASET_PATH = "datasets"
RESULT_PATH = f"results/embeddings/{REDUCE_METHOD}"
SEED = 42  # Random generator seed

# Dataset Parameters
SEEDS = [0]  # List of seeds
TEMPERATURES = [0.2, 0.4, 0.6, 0.8, 1.0]  # List of temperatures
TOP_PS = [1.0]  # List of top_ps
extra_datasets = [
    ["roneneldan/TinyStories"],
    ["HuggingFaceFW/fineweb", "sample-10BT"],
]  # Extra dataset(s) to compare with the synthetic datasets


def load_jsonl(file_path, sample_size):
    """Load JSONL dataset up to given sample_size"""
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for _ in tqdm(range(sample_size), desc="Sampled data"):
            line = f.readline()
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


def reduce_embeddings(embeddings, type="pca"):
    if type == "pca":
        reducer = PCA(n_components=2, random_state=SEED)
    elif type == "pacmap":
        reducer = pacmap.PaCMAP(n_components=2, random_state=SEED)
    elif type == "umap":
        reducer = umap.UMAP(n_components=2, random_state=SEED)
    else:
        raise ValueError("Invalid option.")
    all_embeddings = np.vstack(embeddings)
    reduced_embeddings = reducer.fit_transform(all_embeddings)

    return reduced_embeddings


def plot(reduced_embeddings, labels, imagename="embeddings.png"):
    """Plots the embeddings"""
    plt.figure(figsize=(8, 6))
    start_index = 0
    for i, label in enumerate(labels):
        end_index = start_index + len(embeddings[i])
        plt.scatter(
            reduced_embeddings[start_index:end_index, 0],
            reduced_embeddings[start_index:end_index, 1],
            label=label,
            alpha=0.3,
            s=10,
        )
        start_index = end_index
    plt.legend()
    plt.title("Embeddings Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()

    # Create result directory if not exits
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)

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
            for top_p in TOP_PS:
                labels.append(f"seed={seed},temp={top_p},top_p={top_p}")
                filename = f"seed={seed}-temp={temp}-top_p={top_p}.jsonl"
                dataset_path = os.path.join(DATASET_PATH, filename)
                print(f"Getting embeddings for {dataset_path}")
                dataset = load_jsonl(dataset_path, sample_size=SAMPLE_SIZE)

                # Get embedding
                embedding = get_embedding(dataset)

                # Store the embedding
                embeddings.append(embedding)

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

        # Store the embedding
        embeddings.append(embedding)

    reduced_embeddings = reduce_embeddings(embeddings, type=REDUCE_METHOD)

    # Plot the embeddings
    plot(reduced_embeddings, labels, imagename=f"{REDUCE_METHOD}_temp.png")
