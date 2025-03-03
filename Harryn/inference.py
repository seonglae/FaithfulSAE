import os
import gc
import json
import torch
import numpy as np
from vllm import LLM, SamplingParams

# Model Hyperparameters
TEMPERATURES = [0.2, 0.4, 0.6, 0.8, 1.0]  # List of temperatures
CTX = 1024
MAX_TOKENS = 1024
TOP_PS = [1.0]

# Inference Parameters
N = 10_000  # Number of data to generate
SEEDS = [0, 1, 2, 3, 4]  # List of seeds to generate
SAVE_PATH = "datasets"

# Create N number of empty prompts
prompts = [""] * N

# Create different training dataset with different seeds
for seed in SEEDS:
    # Create different training dataset with different temperatures
    for temp in TEMPERATURES:
        # Create different training dataset with different top_ps
        for top_p in TOP_PS:
            output_file = f"llm-training-dataset-seed={seed}-temp={temp}-top_p={top_p}.jsonl"

            # Create a sampling params object with the temp
            sampling_params = SamplingParams(temperature=temp, top_p=top_p, max_tokens=MAX_TOKENS)

            # Create an LLM using llama 3.1 8B model
            llm = LLM(
                model="meta-llama/Llama-3.1-8B",
                gpu_memory_utilization=0.95,
                max_model_len=CTX,
                seed=seed,
            )

            # Generate and save the outputs.
            output_path = os.path.join(SAVE_PATH, output_file)
            with open(output_path, "w", encoding="utf-8") as f:
                for i, output in enumerate(llm.generate(prompts, sampling_params)):
                    generated_text = output.outputs[0].text

                    # Write output in JSONL format
                    json.dump({"id": i, "text": generated_text}, f, ensure_ascii=False)
                    f.write("\n")

            print(f"Unconditional LLM output saved as {output_path}")

            # Delete the loaded model
            del llm  # Delete model reference
            gc.collect()  # Force garbage collection
            torch.cuda.empty_cache()  # Free GPU memory
