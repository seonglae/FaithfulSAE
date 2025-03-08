import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer

# Model Hyperparameters
MODEL_NAME = "meta-llama/Llama-3.1-8B"

# Configuration
TEXT_KEY = "text"  # JSONL field containing text
SAMPLE_SIZE = 10_000
DATASET_PATH = "datasets"


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


def get_tokens(dataset):
    tokens = []
    for data in tqdm(dataset, desc="Processed data"):
        token = tokenizer(data)["input_ids"]
        tokens.append(token)

    return tokens


if __name__ == "__main__":
    # Get the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Set the dataset path
    filename = f"seed=0-temp=1.0-top_p=1.0.jsonl"
    dataset_path = os.path.join(DATASET_PATH, filename)

    # Load the dataset
    dataset = load_jsonl(dataset_path, sample_size=SAMPLE_SIZE)

    # Initialise the stats
    total_tokens = 0
    total_samples = 0
    mean = 0.0
    m2 = 0.0
    min_tokens = None
    max_tokens = None

    # Get tokens
    tokens = get_tokens(dataset)

    # Loop through tokens
    for i, token in enumerate(tqdm(tokens, desc="Processed tokens")):
        token_count = len(token)
        total_tokens += token_count
        total_samples += 1
        delta = token_count - mean
        mean += delta / total_samples
        delta2 = token_count - mean
        m2 += delta * delta2
        min_tokens = token_count if min_tokens is None or token_count < min_tokens else min_tokens
        max_tokens = token_count if max_tokens is None or token_count > max_tokens else max_tokens
    variance = m2 / (total_samples - 1) if total_samples > 1 else 0
    print("Total samples:", total_samples)
    print("Total tokens:", total_tokens)
    print("Mean token count:", mean)
    print("Token count std:", variance**0.5)
    print("Min token length:", min_tokens)
    print("Max token length:", max_tokens)
