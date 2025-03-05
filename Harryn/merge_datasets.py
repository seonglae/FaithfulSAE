import os
import json

# Dataset Parameters
DATASET_PATH = "datasets"
RESULT_PATH = "results"
SEEDS = [0]  # List of seeds to generate
TEMPERATURES = [0.2, 0.5, 1.0]  # List of temperatures
TOP_PS = [1.0]  # List of top_ps


def merge_jsonl_files(files, output_file):
    counter = 0
    with open(output_file, "w", encoding="utf-8") as outfile:
        for file in files:
            print(f"Merging {file}")
            with open(file, "r", encoding="utf-8") as infile:
                for line in infile:
                    data = json.loads(line)
                    data["id"] = counter
                    counter += 1
                    json.dump(data, outfile, ensure_ascii=False)
                    outfile.write("\n")

    print(f"Merged file as {output_file}")


if __name__ == "__main__":
    # Define the output filename
    output_filename = "merged_temp_dataset.jsonl"
    output_path = os.path.join(DATASET_PATH, output_filename)

    # Get paths of datasets
    datasets = []
    for seed in SEEDS:
        for temp in TEMPERATURES:
            for top_p in TOP_PS:
                filename = f"seed={seed}-temp={temp}-top_p{top_p}.jsonl"
                dataset_path = os.path.join(DATASET_PATH, filename)
                datasets.append(dataset_path)

    # Merge given dataset paths into a single dataset
    merge_jsonl_files(datasets, output_path)
