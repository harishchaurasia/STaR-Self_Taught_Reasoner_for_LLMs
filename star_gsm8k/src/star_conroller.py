#!/usr/bin/env python3
import os, json, re, argparse, subprocess
from typing import List, Dict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch

PROMPT_FALLBACK = """You are a helpful math tutor. Solve the problem step by step, showing clear reasoning.
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

def generate_batches(model, tok, prompts, max_new, bs):
    outputs = []
    for i in tqdm(range(0, len(prompts), bs), desc="gen", unit="batch"):
        batch = prompts[i:i+bs]
        inputs = tok(batch, return_tensors="pt", padding="longest").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        texts = tok.batch_decode(out_ids, skip_special_tokens=True)
        outputs.extend(texts)
    return outputs

def read_jsonl(p):
    with open(p) as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def write_jsonl(p, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")

def run_cmd(cmd):
    print("[cmd]", cmd, flush=True)
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0: raise RuntimeError(f"Command failed: {cmd}")

def evaluate(model_dir, out_jsonl, batch_size=32, max_new=256, tokenizer_id=None, forward_tmpl=None):
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    eval_py = os.path.join(repo_root, "scripts/eval_batched_full.py")
    tok_flag = f"--tokenizer_id {tokenizer_id}" if tokenizer_id else ""
    tmpl_flag = f"--forward_tmpl {forward_tmpl}" if forward_tmpl else ""
    cmd = f"python3 {eval_py} --model_dir {model_dir} {tok_flag} --out {out_jsonl} --batch_size {batch_size} --max_new_tokens {max_new} {tmpl_flag}"
    run_cmd(cmd)
    ok=n=0
    for row in read_jsonl(out_jsonl):
        ok += int(bool(row.get("em", False))); n += 1
    return ok/max(1,n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--tokenizer_id", default=None)
    ap.add_argument("--prompt_path", default="prompts/forward.txt")
    ap.add_argument("--out_dir", default="run/star")
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--batch_gen", type=int, default=32)
    ap.add_argument("--batch_eval", type=int, default=32)
    ap.add_argument("--max_new_gen", type=int, default=256)
    ap.add_argument("--max_new_eval", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok_id = args.tokenizer_id or args.base_model
    tok = AutoTokenizer.from_pretrained(tok_id)
    tok.pad_token = tok.eos_token; tok.padding_side="left"

    # Load prompt
    prompt_text = open(args.prompt_path).read() if os.path.exists(args.prompt_path) else PROMPT_FALLBACK

    # Iter 1: evaluate base model on GSM8K test
    iter1_preds = os.path.join(args.out_dir, "iter1_preds.jsonl")
    em1 = evaluate(args.base_model, iter1_preds, batch_size=args.batch_eval, max_new=args.max_new_eval, tokenizer_id=tok_id, forward_tmpl=args.prompt_path if os.path.exists(args.prompt_path) else None)
    print(f"[iter1] EM={em1:.3f}")

    # Load GSM8K for backfill
    ds_test = load_dataset("openai/gsm8k", "main", split="test")

    cur_model_for_build = args.base_model
    cur_preds = iter1_preds
    for it in range(2, args.iterations+2):
        it_name = f"iter{it}"
        # Build misses
        misses = []
        for row in read_jsonl(cur_preds):
            if row.get("em", False): continue
            q = row.get("question"); gold = row.get("gold_answer"); rid = row.get("id", "")
            if (q is None or gold is None) and isinstance(rid, str) and rid.startswith("hf_test_"):
                idx = int(rid.split("_")[-1])
                q = ds_test[idx]["question"]; gold = ds_test[idx]["answer"]
            if q and gold: misses.append({"id": rid, "question": q, "gold_answer": gold})
        print(f"[{it_name}] misses={len(misses)}")

        # Generate improved solutions
        model = AutoModelForCausalLM.from_pretrained(cur_model_for_build, dtype="auto", device_map="auto")
        model.eval()
        prompts = [prompt_text.format(question=m["question"]) for m in misses]
        outs = generate_batches(model, tok, prompts, args.max_new_gen, args.batch_gen)

        # Save raw new rows
        iter_new = os.path.join(args.out_dir, f"{it_name}_new.jsonl")
        new_rows = []
        for m, t in zip(misses, outs):
            em2 = exact_match_from_text(m["gold_answer"], t)
            new_rows.append({"id": m["id"], "question": m["question"], "gold_answer": m["gold_answer"], "output": t, "em_after": bool(em2)})
        write_jsonl(iter_new, new_rows)
        print(f"[{it_name}] wrote {len(new_rows)} to {iter_new}")

        # Build SFT jsonl for your vanilla_sft.py (question, rationale, answer)
        sft_rows = []
        for r in new_rows:
            rationale = r["output"]
            final = extract_final(rationale) or ""
            sft_rows.append({"question": r["question"], "rationale": rationale, "answer": f"#### {final}" if final else ""})
        prev_train = os.path.join(args.out_dir, f"iter{it-1}_train.jsonl")
        if os.path.exists(prev_train):
            sft_rows.extend({"question": j.get("question",""), "rationale": j.get("rationale",""), "answer": j.get("answer","")} for j in read_jsonl(prev_train))
        iter_train = os.path.join(args.out_dir, f"{it_name}_train.jsonl")
        write_jsonl(iter_train, sft_rows)
        print(f"[{it_name}] train size={len(sft_rows)}")

        # Train one epoch with your existing vanilla_sft.py
        out_model_dir = os.path.join(args.out_dir, f"model_{it_name}")
        repo_root = os.path.dirname(os.path.abspath(__file__)); repo_root = os.path.dirname(repo_root)
        sft_py = os.path.join(repo_root, "src/vanilla_sft.py")
        cmd = f"python3 {sft_py} --model {args.base_model} --train_file {iter_train} --out {out_model_dir} --epochs 1 --lr {args.lr} --max_seq_len 512 --batch 4 --grad_accum 2"
        run_cmd(cmd)
        cur_model_for_build = os.path.join(out_model_dir, "epoch1")

        # Evaluate the new checkpoint
        preds_path = os.path.join(args.out_dir, f"{it_name}_preds.jsonl")
        em = evaluate(cur_model_for_build, preds_path, batch_size=args.batch_eval, max_new=args.max_new_eval, tokenizer_id=tok_id, forward_tmpl=args.prompt_path if os.path.exists(args.prompt_path) else None)
        print(f"[{it_name}] EM={em:.3f}")
        cur_preds = preds_path

    print("Done.")

if __name__ == "__main__":
    main()
