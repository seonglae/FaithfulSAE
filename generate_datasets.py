import os
import gc
import json
import torch
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from constants import ELEUTHERAI, PYTHIA_1_4B

# Model Hyperparameters
CTX = 1024
MAX_TOKENS = 1024
SAVE_PATH = "datasets"

def parse_args():
    parser = argparse.ArgumentParser(description="Generate datasets using language models")
    parser.add_argument("--models", nargs="+", default=[f"{ELEUTHERAI}/{PYTHIA_1_4B}"], 
                        help="List of models to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for generation")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p for generation")
    parser.add_argument("--n", type=int, default=400_000,
                        help="Number of samples to generate")
    return parser.parse_args()

def colored_print(text, index):
    colors = ["\033[91m", "\033[92m", "\033[93m", "\033[94m", "\033[95m", "\033[96m"]
    color = colors[index % len(colors)]
    print(f"{color}{text}\033[0m")


def get_tokenizer_and_prompt(model, num_samples):
    """Get tokenizer and create prompt token IDs"""
    tokenizer = AutoTokenizer.from_pretrained(model)
    bos_token_id = tokenizer.bos_token_id
    prompt_token_ids = [[bos_token_id]] * num_samples
    return tokenizer, prompt_token_ids

def create_model(model_name, seed):
    """Create and return the LLM model"""
    return LLM(
        model=model_name,
        gpu_memory_utilization=0.95,
        max_model_len=CTX,
        seed=seed,
        task="generate",
    )

def generate_batch(llm, batch_prompt_token_ids, sampling_params, start_idx, f):
    """Generate text for a batch of prompts and write to file"""
    for i, output in enumerate(
        llm.generate(prompt_token_ids=batch_prompt_token_ids, sampling_params=sampling_params)
    ):
        generated_text = output.outputs[0].text
        # Calculate the global index
        global_idx = start_idx + i
        # Write output in JSONL format
        json.dump({"id": global_idx, "text": generated_text}, f, ensure_ascii=False)
        f.write("\n")
    
    # Clear cache between batches
    torch.cuda.empty_cache()

def generate_dataset(model, args):
    """Process a single model with given args"""
    model_name = model.split("/")[-1]
    output_file = f"{model_name}_{args.seed}_{args.temperature}_{args.top_p}.jsonl"
    output_path = os.path.join(SAVE_PATH, output_file)
    
    # Get tokenizer and prompt tokens
    tokenizer, prompt_token_ids = get_tokenizer_and_prompt(model, args.n)
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature, 
        top_p=args.top_p, 
        max_tokens=MAX_TOKENS, 
        repetition_penalty=1.1
    )
    
    # Create LLM model
    llm = create_model(model, args.seed)
    
    # Split into batches
    num_batches = 10
    batch_size = args.n // num_batches
    
    # Generate and save outputs
    with open(output_path, "w", encoding="utf-8") as f:
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size if batch < num_batches - 1 else args.n
            current_batch_size = end_idx - start_idx
            
            # Create batch of prompts
            batch_prompt_token_ids = [[tokenizer.bos_token_id]] * current_batch_size
            
            print(f"Processing batch {batch+1}/{num_batches} ({start_idx} to {end_idx-1})")
            generate_batch(llm, batch_prompt_token_ids, sampling_params, start_idx, f)
    
    print(f"Unconditional LLM output saved as {output_path}")
    
    # Clean up resources
    del llm
    gc.collect()
    torch.cuda.empty_cache()

def main():
    args = parse_args()
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    
    for model in args.models:
        generate_dataset(model, args)

if __name__ == "__main__":
    main()
