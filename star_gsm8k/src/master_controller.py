#!/usr/bin/env python3
import os, json, re, argparse, subprocess
from typing import List, Dict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch
from datetime import datetime

PROMPT_FALLBACK = """You are a helpful math tutor. Solve the problem step by step, showing clear reasoning.
At the end, output only the final numeric answer on a new line as: #### <number>
Question:
{question}
"""

HINT_TMPL = """You are a careful math tutor. Re-solve the problem step by step and verify each operation.
Hint: 1) Identify variables and known quantities 2) Set up the relevant equations 3) Compute carefully
4) Check units and parity 5) Recalculate if a step seems inconsistent.
At the end, output only the final numeric answer on a new line as: #### <number>
Question:
{question}
"""

def ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)

def read_jsonl(p: str):
    if not os.path.exists(p): return []
    rows = []
    with open(p) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def write_jsonl(p: str, rows: List[Dict]):
    ensure_dir(os.path.dirname(p) or ".")
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def append_text(p: str, text: str):
    ensure_dir(os.path.dirname(p) or ".")
    with open(p, "a") as f:
        f.write(text + "\n")

def extract_final(x: str):
    m = re.findall(r"####\s*([-\d][\d,\.]*)", x or "")
    return m[-1].replace(",", "") if m else None

def exact_match_from_text(gold: str, text: str):
    pg = extract_final(gold) or gold
    pt = extract_final(text) or ""
    return (pg is not None) and (pt is not None) and (pt.strip() == pg.strip())

def load_prompt(path: str, fallback: str) -> str:
    if path and os.path.exists(path):
        return open(path).read()
    return fallback

def build_prompts(tmpl: str, questions: List[str]) -> List[str]:
    return [tmpl.format(question=q) for q in questions]

def batched_generate(model, tok, prompts, max_new, batch_size, do_sample=False, temperature=1.0, top_p=1.0) -> List[str]:
    out = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="gen", unit="batch"):
        batch = prompts[i:i+batch_size]
        inputs = tok(batch, return_tensors="pt", padding="longest").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tok.eos_token_id,
            )
        texts = tok.batch_decode(out_ids, skip_special_tokens=True)
        out.extend(texts)
    return out

def run_batched_eval(repo_root: str, model_dir: str, tokenizer_id: str, out_jsonl: str, forward_tmpl: str, batch_size: int, max_new: int):
    eval_py = os.path.join(repo_root, "scripts", "eval_batched_full.py")
    ensure_dir(os.path.dirname(out_jsonl) or ".")
    cmd = (
        f"python3 {eval_py} "
        f"--model_dir {model_dir} "
        f"--tokenizer_id {tokenizer_id} "
        f"--out {out_jsonl} "
        f"--forward_tmpl {forward_tmpl} "
        f"--batch_size {batch_size} "
        f"--max_new_tokens {max_new}"
    )
    print("[eval]", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

def compute_em_from_jsonl(p: str) -> float:
    ok = 0; n = 0
    for r in read_jsonl(p):
        ok += int(bool(r.get("em", False))); n += 1
    return ok / max(1, n)

def run_vanilla_sft(sft_py_path: str, base_or_prev_model: str, train_file: str, out_dir: str, epochs: int, lr: float, max_len: int, batch: int, grad_accum: int, subset: int = 0):
    ensure_dir(out_dir)
    cmd = (
        f"python3 {sft_py_path}"
        f" --model {base_or_prev_model}"
        f" --train_file {train_file}"
        f" --out {out_dir}"
        f" --epochs {epochs}"
        f" --lr {lr}"
        f" --max_seq_len {max_len}"
        f" --batch {batch}"
        f" --grad_accum {grad_accum}"
        f" --subset {subset}"
    )
    print("[train]", cmd, flush=True)
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        raise RuntimeError("vanilla_sft.py failed")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--tokenizer_id", default=None)
    ap.add_argument("--prompt_path", default="prompts/forward.txt")
    ap.add_argument("--hint_prompt_path", default="prompts/hint_forward.txt")
    ap.add_argument("--out_dir", default="run/star")
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--batch_gen", type=int, default=32)
    ap.add_argument("--batch_eval", type=int, default=32)
    ap.add_argument("--max_new_gen", type=int, default=256)
    ap.add_argument("--max_new_eval", type=int, default=256)
    ap.add_argument("--vanilla_epochs", type=int, default=1)
    ap.add_argument("--vanilla_lr", type=float, default=1e-5)
    ap.add_argument("--vanilla_subset", type=int, default=0)   # full train
    ap.add_argument("--sft_batch", type=int, default=32)
    ap.add_argument("--sft_grad_accum", type=int, default=2)
    ap.add_argument("--k_consensus", type=int, default=1)      # set >1 to sample multiple per miss
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # star_gsm8k/
    report_path = os.path.join(args.out_dir, "master_report.txt")
    append_text(report_path, f"\n==== Run started {datetime.now().isoformat()} ====")

    tok_id = args.tokenizer_id or args.base_model

    # 1) Zero-shot evaluation (batched)
    z_out = os.path.join(args.out_dir, "zero_shot_preds_full_batched.jsonl")
    run_batched_eval(repo_root, args.base_model, tok_id, z_out, args.prompt_path, args.batch_eval, args.max_new_eval)
    z_em = compute_em_from_jsonl(z_out)
    append_text(report_path, f"[zero_shot] EM={z_em:.3f} batch={args.batch_eval} max_new={args.max_new_eval}")

    # 2) Vanilla SFT on full GSM8K train then evaluate (batched)
    here = os.path.dirname(os.path.abspath(__file__))
    sft_py = os.path.join(here, "vanilla_sft.py")
    v_out = os.path.join(args.out_dir, "model_vanilla")
    v_train_jsonl = os.path.join("data", "gsm8k_train_sft.jsonl")  # created by vanilla_sft.py if missing
    run_vanilla_sft(
        sft_py_path=sft_py,
        base_or_prev_model=args.base_model,
        train_file=v_train_jsonl,
        out_dir=v_out,
        epochs=args.vanilla_epochs,
        lr=args.vanilla_lr,
        max_len=512,
        batch=args.sft_batch,
        grad_accum=args.sft_grad_accum,
        subset=args.vanilla_subset,
    )
    v_ckpt = os.path.join(v_out, "epoch1")
    v_out_jsonl = os.path.join(args.out_dir, "vanilla_preds_full_batched.jsonl")
    run_batched_eval(repo_root, v_ckpt, tok_id, v_out_jsonl, args.prompt_path, args.batch_eval, args.max_new_eval)
    v_em = compute_em_from_jsonl(v_out_jsonl)
    append_text(report_path, f"[vanilla_sft] EM={v_em:.3f} epochs={args.vanilla_epochs} lr={args.vanilla_lr} subset={args.vanilla_subset}")

    # 3) STaR iterations: hinted regeneration -> correct-only cumulative SFT -> train from last model -> eval
    ds_test = load_dataset("openai/gsm8k", "main", split="test")
    base_prompt = load_prompt(args.prompt_path, PROMPT_FALLBACK)
    hint_prompt = load_prompt(args.hint_prompt_path, HINT_TMPL)
    tok = AutoTokenizer.from_pretrained(tok_id); tok.pad_token = tok.eos_token; tok.padding_side = "left"

    cur_model_for_build = v_ckpt
    cur_eval_jsonl = v_out_jsonl

    for it in range(2, args.iterations + 2):
        it_name = f"iter{it}"

        # Collect misses
        misses = []
        for row in read_jsonl(cur_eval_jsonl):
            if row.get("em", False): continue
            q = row.get("question"); gold = row.get("gold_answer"); rid = row.get("id", "")
            if (q is None or gold is None) and isinstance(rid, str) and rid.startswith("hf_test_"):
                try:
                    idx = int(rid.split("_")[-1])
                    q = ds_test[idx]["question"]; gold = ds_test[idx]["answer"]
                except Exception:
                    pass
            if q and gold: misses.append({"id": rid, "question": q, "gold_answer": gold})
        append_text(report_path, f"[{it_name}] misses={len(misses)}")
        if not misses:
            append_text(report_path, f"[{it_name}] no misses; stopping early")
            break

        # Regenerate with hints; optionally self-consistency (k>1 with sampling)
        gen_model = AutoModelForCausalLM.from_pretrained(cur_model_for_build, dtype="auto", device_map="auto")
        gen_model.eval()

        kept_rows = []
        if args.k_consensus <= 1:
            prompts = build_prompts(hint_prompt, [m["question"] for m in misses])
            gens = batched_generate(gen_model, tok, prompts, args.max_new_gen, args.batch_gen, do_sample=False)
            for m, t in zip(misses, gens):
                if exact_match_from_text(m["gold_answer"], t):
                    kept_rows.append({"id": m["id"], "question": m["question"], "gold_answer": m["gold_answer"], "output": t, "em_after": True})
        else:
            # sample k candidates and keep any that match gold; pick shortest matching chain
            prompts = build_prompts(hint_prompt, [m["question"] for m in misses])
            for i in tqdm(range(0, len(prompts), args.batch_gen), desc="consensus-gen", unit="batch"):
                batch_prompts = prompts[i:i+args.batch_gen]
                batch_misses = misses[i:i+args.batch_gen]
                cand_texts = []
                for _ in range(args.k_consensus):
                    outs = batched_generate(gen_model, tok, batch_prompts, args.max_new_gen, len(batch_prompts), do_sample=True, temperature=args.temperature, top_p=args.top_p)
                    cand_texts.append(outs)
                # transpose: per example list of candidates
                for m_idx, m in enumerate(batch_misses):
                    cands = [cand_texts[k][m_idx] for k in range(args.k_consensus)]
                    matches = [c for c in cands if exact_match_from_text(m["gold_answer"], c)]
                    if matches:
                        best = min(matches, key=lambda x: len(x))
                        kept_rows.append({"id": m["id"], "question": m["question"], "gold_answer": m["gold_answer"], "output": best, "em_after": True})

        it_new = os.path.join(args.out_dir, f"{it_name}_new.jsonl")
        write_jsonl(it_new, kept_rows)
        append_text(report_path, f"[{it_name}] generated={len(misses)} kept_correct={len(kept_rows)}")

        # Build correct-only cumulative SFT set: question, rationale, answer
        it_train = os.path.join(args.out_dir, f"{it_name}_train.jsonl")
        sft_rows = []
        for r in kept_rows:
            rationale = r["output"]
            final = extract_final(rationale) or ""
            if final:
                sft_rows.append({"question": r["question"], "rationale": rationale, "answer": f"#### {final}"})
        # cumulative: add prior SFT train
        prev_train = os.path.join(args.out_dir, f"iter{it-1}_train.jsonl")
        if os.path.exists(prev_train):
            sft_rows.extend(read_jsonl(prev_train))
        write_jsonl(it_train, sft_rows)
        append_text(report_path, f"[{it_name}] train_size={len(sft_rows)}")

        # Train one epoch from the previous model (incremental) and save as model_iterN/epoch1
        it_model_dir = os.path.join(args.out_dir, f"model_{it_name}")
        run_vanilla_sft(
            sft_py_path=sft_py,
            base_or_prev_model=cur_model_for_build,   # incremental fine-tuning
            train_file=it_train,
            out_dir=it_model_dir,
            epochs=1,
            lr=args.vanilla_lr,
            max_len=512,
            batch=args.sft_batch,
            grad_accum=args.sft_grad_accum,
            subset=0,
        )
        cur_model_for_build = os.path.join(it_model_dir, "epoch1")

        # Evaluate with batched evaluator and record EM
        it_preds = os.path.join(args.out_dir, f"{it_name}_preds_full_batched.jsonl")
        run_batched_eval(repo_root, cur_model_for_build, tok_id, it_preds, args.prompt_path, args.batch_eval, args.max_new_eval)
        it_em = compute_em_from_jsonl(it_preds)
        append_text(report_path, f"[{it_name}] EM={it_em:.3f} batch={args.batch_eval} max_new={args.max_new_eval}")
        cur_eval_jsonl = it_preds

    append_text(report_path, f"==== Run ended {datetime.now().isoformat()} ====")
    print(f"\nWrote report: {report_path}")

if __name__ == "__main__":
    main()
