import os
import json
from tqdm import tqdm


def merge_datasets(files, output_file):
    counter = 0
    with open(output_file, "w", encoding="utf-8") as outfile:
        for file in files:
            print(f"Merging {file}")
            with open(file, "r", encoding="utf-8") as infile:
                lines = infile.readlines()
                for line in tqdm(lines, desc="Sampled lines", unit=" lines"):
                    data = json.loads(line)
                    data["id"] = counter
                    counter += 1
                    json.dump(data, outfile, ensure_ascii=False)
                    outfile.write("\n")

    print(f"Merged file saved as {output_file}")


if __name__ == "__main__":
    # Dataset Parameters
    DATASET_PATH = "datasets"

    # Define the output filename
    output_filename = "pythia-2.8b-harmful_dataset.jsonl"
    output_path = os.path.join(DATASET_PATH, output_filename)

    # Get paths of datasets
    datasets = ["datasets/pythia-2.8b_synthetic_180k.jsonl", "datasets/more-harmful_dataset.jsonl"]

    # Merge given dataset paths into a single dataset
    merge_datasets(datasets, output_path)
