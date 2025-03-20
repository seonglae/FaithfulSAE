import os
import gc
import json
import torch
import fire
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

from datasets import Dataset, DatasetDict
from sglang.utils import launch_server_cmd, wait_for_server, terminate_process

from transformers import AutoTokenizer


def vllm(model_name="EleutherAI/pythia-1b", ctx=1024, max_tokens=1024, seed=42, temperature=1.0, top_p=1.0, total_tokens=4e8, repo="seonglae/faithful-pythia-1b"):
    from vllm import LLM, SamplingParams
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bos_token_id = tokenizer.bos_token_id
    batch_size = 1000
    
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    llm = LLM(model=model_name, max_model_len=ctx, seed=seed, task="generate")
    
    all_data = []
    tokens_generated = 0
    sample_id = 0
    
    pbar = tqdm(total=total_tokens, unit='tokens')
    
    while tokens_generated < total_tokens:
        prompt_token_ids = [[bos_token_id]] * batch_size
        outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
        
        texts = [output.outputs[0].text for output in outputs]
        encodings = tokenizer.batch_encode_plus(texts, return_length=True)
        token_counts = encodings["length"]
        
        for i, output in enumerate(outputs):
            text = texts[i]
            num_tokens = token_counts[i]
            
            all_data.append({
                "id": sample_id, 
                "seed": seed, 
                "temp": temperature, 
                "top_p": top_p, 
                "text": text,
                "tokens": num_tokens
            })
            
            tokens_generated += num_tokens
            sample_id += 1
            pbar.update(num_tokens)
    
    pbar.close()
    
    dataset = Dataset.from_dict({
        "id": [d["id"] for d in all_data],
        "seed": [d["seed"] for d in all_data],
        "temp": [d["temp"] for d in all_data],
        "top_p": [d["top_p"] for d in all_data],
        "text": [d["text"] for d in all_data],
        "tokens": [d["tokens"] for d in all_data]
    })
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dict.push_to_hub(repo)
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def sglang(
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    max_tokens=1024,
    seed=0,
    temperature=1.0,
    top_p=1.0,
    prompt="The capital of France is",
    total_tokens=4e8,
    repo="seonglae/faithful-llama3.1-8b",
    workers=10
):
    server_cmd = f"python -m sglang.launch_server --model-path {model_name} --host 0.0.0.0"
    server_process, port = launch_server_cmd(server_cmd)
    wait_for_server(f"http://localhost:{port}")
    print(f"Server started on http://localhost:{port}")
    
    base_url = f"http://localhost:{port}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    all_data = []
    tokens_generated = 0
    sample_id = 0
    
    pbar = tqdm(total=total_tokens, unit='tokens')
    
    def call_generate():
        nonlocal tokens_generated, sample_id
        
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "top_p": top_p
            }
        }
        response = requests.post(f"{base_url}/generate", json=payload)
        result = response.json()
        generated_text = result.get("text", "")
        num_tokens = len(tokenizer.encode(generated_text))
        
        return {"id": sample_id, "seed": seed, "temp": temperature, "top_p": top_p, "text": generated_text, "tokens": num_tokens}
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        
        for _ in range(workers):
            if tokens_generated < total_tokens:
                futures.append(executor.submit(call_generate))
            
        while futures:
            done, futures = wait_on_first_completed(futures)
            
            for future in done:
                result = future.result()
                all_data.append(result)
                num_tokens = result["tokens"]
                tokens_generated += num_tokens
                sample_id += 1
                pbar.update(num_tokens)
                
                if tokens_generated < total_tokens:
                    futures.append(executor.submit(call_generate))
                
                if tokens_generated >= total_tokens:
                    break
    
    pbar.close()
    
    dataset = Dataset.from_dict({
        "id": [d["id"] for d in all_data],
        "seed": [d["seed"] for d in all_data],
        "temp": [d["temp"] for d in all_data],
        "top_p": [d["top_p"] for d in all_data],
        "text": [d["text"] for d in all_data],
        "tokens": [d["tokens"] for d in all_data]
    })
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dict.push_to_hub(repo)
    
    terminate_process(server_process)
    gc.collect()
    torch.cuda.empty_cache()

def wait_on_first_completed(futures):
    """Wait for the first future to complete and return done and not done futures."""
    done, not_done = concurrent.futures.wait(
        futures, 
        return_when=concurrent.futures.FIRST_COMPLETED
    )
    return done, list(not_done)

if __name__ == "__main__":
    fire.Fire({
        "sglang": sglang,
        "vllm": vllm,
    })
