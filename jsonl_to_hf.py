import os, re, fire
from datasets import load_dataset, concatenate_datasets, DatasetDict

def main(remote_repo="seonglae/true-synthetic-llama3.1-8b", directory="."):
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
        ds = load_dataset("json", data_files=file_path, split="train").map(lambda x, seed=seed, temp=temp, top_p=top_p: {"seed": int(seed), "temp": f>        dss.append(ds)
        subsets[key] = ds
    combined = concatenate_datasets(dss)
    dd = DatasetDict({"train": combined})
    for key, ds in subsets.items():
        dd[key] = ds
    dd.push_to_hub(remote_repo)

if __name__ == '__main__':
    fire.Fire(main)
