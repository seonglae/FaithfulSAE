import fire
from datasets import load_dataset
from transformers import AutoTokenizer

def main(tokenizer_name="meta-llama/Llama-3.1-8B", dataset_name="seonglae/true-synthetic-llama3.1-8b", column="text", split="train"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    ds = load_dataset(dataset_name, split=split, streaming=True)
    total_tokens = 0
    total_examples = 0
    mean = 0.0
    m2 = 0.0
    min_tokens = None
    max_tokens = None
    for example in ds:
        token_count = len(tokenizer(example[column])["input_ids"])
        total_tokens += token_count
        total_examples += 1
        delta = token_count - mean
        mean += delta / total_examples
        delta2 = token_count - mean
        m2 += delta * delta2
        min_tokens = token_count if min_tokens is None or token_count < min_tokens else min_tokens
        max_tokens = token_count if max_tokens is None or token_count > max_tokens else max_tokens
    variance = m2 / (total_examples - 1) if total_examples > 1 else 0
    print("Total sequences:", total_examples)
    print("Total tokens:", total_tokens)
    print("Mean token count:", mean)
    print("Token count std:", variance ** 0.5)
    print("Min seqence length:", min_tokens)
    print("Max sequence length:", max_tokens)

if __name__ == '__main__':
    fire.Fire(main)
