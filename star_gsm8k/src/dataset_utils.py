# src/dataset_utils.py
# Utilities to load GSM8K from HF if local JSONL not provided, and to build class-format JSONL.
import json
from datasets import load_dataset

def load_gsm8k_split(split="train"):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return ds

def write_class_jsonl_from_hf(split, out_path, limit=None):
    """
    Convert HF GSM8K split to lines {"id", "question", "gold_answer", "gold_rationale", "text"}
    where text is the class-format "###Input:/###Output:...#### final"
    """
    ds = load_gsm8k_split(split=split)
    out = []
    cnt = 0
    with open(out_path, "w", encoding="utf-8") as w:
        for i, ex in enumerate(ds):
            if limit and i >= limit: break
            q = ex["question"]
            # GSM8K 'answer' field often is like "X\n\nThe final answer is 123."
            # If dataset provides 'answer' or 'answer', try to extract numeric final (but we keep gold answer field).
            ga = ex.get("answer", "").strip()
            # paper uses human rationale for Vanilla SFT; some versions of GSM8K include 'rationale' or 'solution'
            gr = ex.get("rationale", "")
            if not gr:
                # try 'explanation' or 'solution' keys
                gr = ex.get("solution", "") or ex.get("explanation", "")
            text = f"###Input:\n{q}\n\n###Output:\n{gr}\n#### {ga}"
            j = {"id": f"{split}_{i}", "question": q, "gold_answer": ga, "gold_rationale": gr, "text": text}
            w.write(json.dumps(j, ensure_ascii=False) + "\n")
            cnt += 1
    return out_path, cnt
