import os
import json
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer


def download_dataset(file, output_file, target_tokens):
    with open(output_file, "w", encoding="utf-8") as outfile:
        print(f"Downloading {file}")
        # Load a dataset
        dataset = load_dataset(file, split="train", streaming=True)

        # Shuffle the dataset
        dataset = dataset.shuffle(seed=SHUFFLE_SEED)

        # Loop through the dataset
        counter = 0
        tokens = 0
        for line in tqdm(dataset, desc="Sampled lines", unit=" lines"):
            if file == "aifeifei798/merged_uncensored_alpaca":
                text = get_harmful_text(line)
            elif file == "Muennighoff/natural-instructions":
                text = get_natural_instruct_text(line)
            else:
                text = line["text"]

            # Create a data
            data = {"id": counter, "text": text}
            counter += 1

            # Dump it into the outfile
            json.dump(data, outfile, ensure_ascii=False)
            outfile.write("\n")

            # Compute the number of tokens
            tokens += len(tokenizer(text)["input_ids"])

            # Stop downloading data if the tokens is at least target_tokens
            if tokens >= target_tokens:
                break

    print(f"File saved as {output_file} with {tokens} tokens")

def get_natural_instruct_text(line):
    definition = line["definition"]
    inputs = line["inputs"]
    targets = line["targets"]

    text = f"Q:\n\n{definition}\n\n{inputs}\n\nA:\n\n{targets}"
    return text


def get_harmful_text(line):
    # Get each values for all keys and preprocess
    instruction = preprocess_text(line["instruction"])
    input = preprocess_text(line["input"])
    output = preprocess_text(line["output"])

    # Construct the text in the format of "Q:\n\nQustion?\n\nA:\n\nAnswer"

    # Use instruction for Question of input is empty or instruction start with "Jenna, "
    if input == "" or "Jenna, " in line["instruction"]:
        text = f"Q:\n\n{instruction}\n\nA:\n\n{output}"
    # Use input otherwise
    else:
        text = f"Q:\n\n{input}\n\nA:\n\n{output}"
    return text


def preprocess_text(text):
    # Remove "Jenna, "
    if text.startswith("Jenna, "):
        text = text.replace("Jenna, ", "", 1)

    # Remove "Q: "
    if text.startswith("Q: "):
        text = text.replace("Q: ", "", 1)

    # Remove "Explanation: "
    if text.startswith("Explanation: "):
        text = text.replace("Explanation: ", "", 1)

    # Capitalise the first letter
    text = text.title()

    return text


if __name__ == "__main__":
    # Model Hyperparameters
    MODEL_NAME = "EleutherAI/pythia-2.8b"

    # Dataset Parameters
    DATASET_PATH = "datasets"
    SHUFFLE_SEED = 42
    TARGET_TOKENS = 100_000_000

    # Define the output filename
    output_filename = "better-natural-instructions_dataset.jsonl"
    output_path = os.path.join(DATASET_PATH, output_filename)

    # Path of the dataset
    dataset = "Muennighoff/natural-instructions"

    # Get the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Download target_token amount of dataset
    download_dataset(dataset, output_path, TARGET_TOKENS)
