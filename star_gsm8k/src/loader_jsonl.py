# src/loader_jsonl.py
# small loader for datasets formatted as {"id":..., "text": "...###Input...###Output...####..."}
from datasets import load_dataset

def load_jsonl_dataset(path, split_ratio=0.0, seed=42):
    # returns HF dataset or splitted dict if split_ratio>0
    data = load_dataset("json", data_files=path, split="train")
    if split_ratio and split_ratio > 0.0:
        return data.train_test_split(test_size=split_ratio, seed=seed)
    return data
