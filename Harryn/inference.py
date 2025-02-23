import gc
import json
import torch
from vllm import LLM, SamplingParams

# Model Hyperparameters
TEMPERATURE = 1.0
MAX_TOKENS = 1024
TOP_P = 1.0

# Inference Parameters
N = 10_000  # Number of data to generate
S = 5  # Number of seeds to generate

# Create a sampling params object with the SEED
sampling_params = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS)

# Create different training dataset with S different seeds
for seed in range(S):
    output_file = f"llm-training-dataset-{MAX_TOKENS}ctx-seed={seed}.jsonl"
    # Create an LLM using llama 3.1 8B model
    llm = LLM(
        model="meta-llama/Llama-3.1-8B", gpu_memory_utilization=0.95, max_model_len=1024, seed=seed
    )

    # Create N number of empty prompts
    prompts = [""] * N

    # Generate and save the outputs.
    with open(output_file, "w", encoding="utf-8") as f:
        for i, output in enumerate(llm.generate(prompts, sampling_params)):
            generated_text = output.outputs[0].text

            # Write output in JSONL format
            json.dump({"id": i, "text": generated_text}, f, ensure_ascii=False)
            f.write("\n")

    print(f"Unconditional LLM output saved as {output_file}")

    # Delete the loaded model
    del llm # Delete model reference
    gc.collect() # Force garbage collection 
    torch.cuda.empty_cache() # Free GPU memory
