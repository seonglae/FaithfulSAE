import os, re, fire
from datasets import load_dataset, concatenate_datasets, DatasetDict

def detect(remote_repo="seonglae/true-synthetic-llama3.1-8b", directory="."):
    files = [f for f in os.listdir(directory) if f.endswith('.jsonl')]
    dss = []
    subsets = {}
    for f in files:
        m = re.search(r'seed=(\d+)-temp=([\d\.]+)-top_p=([\d\.]+)\.jsonl$', f)
        if not m:
            continue
        seed, temp, top_p = m.groups()
        key = f"seed{seed}_temp{temp.replace('.', '_')}_top_p{top_p.replace('.', '_')}"
        file_path = os.path.join(directory, f)
        ds = load_dataset("json", data_files=file_path, split="train").map(lambda x, seed=seed, temp=temp, top_p=top_p: {"seed": int(seed), "temp": float(temp), "top_p": float(top_p), "text": x["text"]})
        dss.append(ds)
        subsets[key] = ds
    combined = concatenate_datasets(dss)
    dd = DatasetDict({"train": combined})
    for key, ds in subsets.items():
        dd[key] = ds
    dd.push_to_hub(remote_repo)

def manual(remote_repo="seonglae/true-synthetic-pythia-6.9b", directory=".", seed=0, temperature=1.0, top_p=1.0):
    files = [f for f in os.listdir(directory) if f.endswith('.jsonl')]
    dss = []
    subsets = {}
    for f in files:
        key = f"seed{seed}_temp{temperature}_top_p{top_p}"
        file_path = os.path.join(directory, f)
        ds = load_dataset("json", data_files=file_path, split="train").map(lambda x: {"seed": seed, "temp": temperature, "top_p": top_p, "text": x["text"]})
        dss.append(ds)
        subsets[key] = ds
    combined = concatenate_datasets(dss)
    dd = DatasetDict({"train": combined})
    for key, ds in subsets.items():
        dd[key] = ds
    dd.push_to_hub(remote_repo)

if __name__ == '__main__':
    fire.Fire({
        'detect': detect,
        'manual': manual
    })
