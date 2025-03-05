import os
import json
import umap
import pacmap
import numpy as np
import matplotlib.pyplot as plt
from vllm import LLM
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Model Parameters
MODEL_NAME = "meta-llama/Llama-3.1-8B"

# Configuration
TEXT_KEY = "text"  # JSONL field containing text
BATCH_SIZE = 1_000
SAMPLE_SIZE = 10_000
REDUCE_METHOD = "pca"
DATASET_PATH = "datasets"
RESULT_PATH = f"results/kmeans/{REDUCE_METHOD}"
SEED = 42  # Random generator seed


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


def apply_kmeans(embeddings, k=5):
    kmeans = KMeans(n_clusters=k, random_state=SEED)
    labels = kmeans.fit_predict(embeddings)

    return labels


def plot_kmeans(reduced_embeddings, labels, imagename="kmeans.png"):
    """Plots kmeans"""

    # Define markers and colors
    markers = ["o", "s", "D", "^", "X"]  # Different markers for each cluster
    colors = ["red", "blue", "green", "purple", "orange"]

    # Plot
    plt.figure(figsize=(8, 6))
    for cluster in range(len(labels)):
        points = reduced_embeddings[labels == cluster]
        plt.scatter(
            points[:, 0],
            points[:, 1],
            label=f"Cluster {cluster}",
            marker=markers[cluster % len(markers)],
            color=colors[cluster % len(colors)],
            alpha=0.7,
        )
    plt.title("K-Means Clustering of LLM Data")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Clusters")
    plt.tight_layout()

    # Create result directory if not exits
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)

    # Save the image
    path = os.path.join(RESULT_PATH, imagename)
    plt.savefig(path)
    print(f"Saved {imagename}")


if __name__ == "__main__":
    # Load model
    llm = LLM(model=MODEL_NAME, gpu_memory_utilization=0.95, task="embed")

    # Fet the embeddings of the dataset
    filename = f"merged_temp_dataset.jsonl"
    dataset_path = os.path.join(DATASET_PATH, filename)
    print(f"Getting embeddings for {dataset_path}")
    dataset = load_jsonl(dataset_path, sample_size=SAMPLE_SIZE)

    # Get embedding
    embeddings = get_embedding(dataset)

    # Apply k-means
    labels = apply_kmeans(embeddings, k=5)

    # Reduce the embeddings
    reduced_embeddings = reduce_embeddings(embeddings, REDUCE_METHOD)

    # Plot kmeans
    plot_kmeans(reduced_embeddings, labels, imagename="kmeans.png")
