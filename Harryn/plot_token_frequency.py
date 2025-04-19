# Import Packages
import os
import json
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# Parameters
MODEL_NAME = "gpt2"
DATASET_PATH = "datasets/llm-training-dataset-1024ctx-seed=0.jsonl"
RESULT_PATH = "results"
TEXT_KEY = "text"  # JSONL field containing text

# Define jsonl loading function
def load_jsonl(path):
    """
    Load entire JSONL dataset
    """
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if TEXT_KEY in data:
                texts.append(data[TEXT_KEY])

    return texts

# Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Define distribution function
def compute_token_frequency(dataset):
    counts = {}
    for data in dataset:
        tokens = tokenizer.encode(data, add_special_tokens=False)
        for token in tokens:
            if token in counts:
                counts[token] += 1
            else:
                counts[token] = 1
    return counts

# Define plotting function
def plot(frequency, top_n=-1):
    """
    Plot a bar chart based on thier token frequency with given top_n

    Params:
        top_n (default: -1):
            - Number of tokens to display. When top_p=-1, it displays all tokens.
    """
    # Sort the token by their counts
    sorted_tokens = sorted(frequency.items(), key=lambda x: x[1], reverse=True)

    # Get only top_n number of results
    if top_n != -1:
        sorted_tokens = sorted_tokens[:top_n]

    # Extract token_id and counts
    token_id, counts = zip(*sorted_tokens)

    # Decode each token ids
    decoded_token = [tokenizer.decode(token) for token in token_id]

    # Plot the result
    plt.figure(figsize=(10, 5))
    plt.bar(decoded_token, counts, alpha=0.7, edgecolor="black", width=0.8)
    plt.xlabel("Token")
    plt.ylabel("Frequency")
    plt.title(f"Top {top_n} Most Frequent Tokens")
    plt.xticks(rotation=45)
    plt.grid(axis="y")
    imagename = "".join(DATASET_PATH.split(".")[:-1]) + ".png"
    path = os.path.join(RESULT_PATH, imagename)
    plt.savefig(path)
    print(f"Saved {path}.")

# Load the dataset
dataset = load_jsonl(DATASET_PATH)

# Run the token freqeuncy function on the loaded dataset
token_frequency = compute_token_frequency(dataset)

# Plot the distribution
plot(token_frequency, top_n=20)