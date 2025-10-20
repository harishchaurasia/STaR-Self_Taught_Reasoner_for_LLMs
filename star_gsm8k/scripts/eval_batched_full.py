#!/usr/bin/env python3
import os, json, re, math, argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch

DEF_TMPL = """You are a helpful math tutor. Solve the problem step by step, showing clear reasoning.
At the end, output only the final numeric answer on a new line as: #### <number>
Question:
{question}
"""

def extract_final(x: str):
    m = re.findall(r"####\s*([-\d][\d,\.]*)", x or "")
    return m[-1].replace(",", "") if m else None

def exact_match_from_text(gold: str, text: str):
    pg = extract_final(gold) or gold
    pt = extract_final(text) or ""
    return (pg is not None) and (pt is not None) and (pt.strip() == pg.strip())

def shard_indices(n: int, k: int, i: int):
    per = math.ceil(n / k); s = i * per; e = min(n, s + per); return s, e

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--tokenizer_id", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--forward_tmpl", default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tok_id = args.tokenizer_id or args.model_dir
    tok = AutoTokenizer.from_pretrained(tok_id)
    tok.pad_token = tok.eos_token; tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype="auto", device_map="auto")
    model.eval()

    ds = load_dataset("openai/gsm8k", "main", split="test")
    total = len(ds)
    s, e = shard_indices(total, args.num_shards, args.shard_idx)
    idxs = list(range(s, e))
    if args.limit: idxs = idxs[:args.limit]

    tmpl = open(args.forward_tmpl).read() if (args.forward_tmpl and os.path.exists(args.forward_tmpl)) else DEF_TMPL
    def make_prompt(q: str): return tmpl.format(question=q)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ok = 0; written = 0

    with open(args.out, "w") as f:
        for i in tqdm(range(0, len(idxs), args.batch_size), desc=f"Eval shard {args.shard_idx}/{args.num_shards}", unit="batch"):
            batch_ids = idxs[i:i+args.batch_size]
            qs = [ds[j]["question"] for j in batch_ids]
            golds = [ds[j]["answer"] for j in batch_ids]
            prompts = [make_prompt(q) for q in qs]

            inputs = tok(prompts, return_tensors="pt", padding="longest").to(model.device)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            texts = tok.batch_decode(out_ids, skip_special_tokens=True)[:len(batch_ids)]

            for j, text in enumerate(texts):
                em = exact_match_from_text(golds[j], text)
                row = {"id": f"hf_test_{batch_ids[j]}", "em": bool(em), "gold_answer": golds[j], "output": text}
                f.write(json.dumps(row) + "\n"); ok += int(bool(em)); written += 1

    em_final = ok / max(1, written)
    print(f"DONE {written} EM={em_final:.3f} bs={args.batch_size} max_new={args.max_new_tokens}")

if __name__ == "__main__":
    main()
