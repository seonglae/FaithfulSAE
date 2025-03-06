import os
import gc
import json
import torch
from vllm import LLM, SamplingParams

# Model Hyperparameters
MODEL_NAME = "meta-llama/Llama-3.1-8B"
CTX = 1024
MAX_TOKENS = 1024

# Dataset Parameters
SAVE_PATH = "datasets"
SEEDS = [0]  # List of seeds to generate
N = 10_000  # Number of data to generate
TEMPERATURES = [0.2, 0.5, 1.0]  # List of temperatures
TOP_PS = [0.5, 0.8, 0.9, 1.0]  # List of top ps

# Create N number of empty prompts
prompts = [""] * N

# Create different training dataset with different seeds
for seed in SEEDS:
    # Create different training dataset with different temperatures
    for temp in TEMPERATURES:
        # Create different training dataset with different top_ps
        for top_p in TOP_PS:
            output_file = f"seed={seed}-temp={temp}-top_p={top_p}.jsonl"

            # Create a sampling params object with the temp
            sampling_params = SamplingParams(temperature=temp, top_p=top_p, max_tokens=MAX_TOKENS)

            # Create an LLM using llama 3.1 8B model
            llm = LLM(
                model=MODEL_NAME,
                gpu_memory_utilization=0.95,
                max_model_len=CTX,
                seed=seed,
                task="generate",
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
