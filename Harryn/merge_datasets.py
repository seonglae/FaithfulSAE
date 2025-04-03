import os
import json
from tqdm import tqdm
from datasets import load_dataset

# Dataset Parameters
DATASET_PATH = "datasets"
RESULT_PATH = "results"
SEEDS = [0]  # List of seeds to generate
TEMPERATURES = [0.2, 0.5, 1.0]  # List of temperatures
TOP_PS = [1.0]  # List of top_ps


def merge_datasets(files, output_file):
    counter = 0
    with open(output_file, "w", encoding="utf-8") as outfile:
        for file in files:
            print(f"Merging {file}")
            if file.endswith(".jsonl"):
                with open(file, "r", encoding="utf-8") as infile:
                    for line in tqdm(infile, desc="Sampled lines", unit=" lines"):
                        data = json.loads(line)
                        data["id"] = counter
                        counter += 1
                        json.dump(data, outfile, ensure_ascii=False)
                        outfile.write("\n")
            else:
                # Load a dataset
                dataset = load_dataset(file, split="train", streaming=True)

                # Loop through the dataset
                for line in tqdm(dataset, desc="Sampled lines", unit=" lines"):
                    # Reformat the prompt and harmful answer in a form below:
                    # Q:\n\nQustion?\n\nA:\n\nAnswer
                    question = (
                        line["prompt"].replace("[INST]\n\n ", "Q:\n\n").replace(" [/INST]", "")
                    )
                    answer = line["chosen"].lstrip().replace("</s>", "")

                    # Merge it
                    text = f"{question}\n\nA:\n\n{answer}"

                    # Create a data
                    data = {"id": counter, "text": text}
                    counter += 1

                    # Dump it into the outfile
                    json.dump(data, outfile, ensure_ascii=False)
                    outfile.write("\n")

    print(f"Merged file saved as {output_file}")


if __name__ == "__main__":
    # Define the output filename
    output_filename = "merged_harmful_dataset.jsonl"
    output_path = os.path.join(DATASET_PATH, output_filename)

    # Get paths of datasets
    datasets = ["datasets/pythia-2.8b_synthetic_180k.jsonl", "Baidicoot/helpful-harmful-rlhf"]

    # Merge given dataset paths into a single dataset
    merge_datasets(datasets, output_path)
