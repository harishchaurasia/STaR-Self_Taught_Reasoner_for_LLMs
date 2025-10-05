#!/usr/bin/env python3

import argparse, json, re, sys, math, os
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Prompt template for Zero-shot CoT
DEFAULT_PROMPT_TMPL = "Q: {q}\nLet's think step by step.\nA:"

def extract_final_answer(text):
    if text is None:
        return ""
    txt = str(text).strip()
    lower = txt.lower()
    patterns = [
        r"final answer(?: is|:)?\s*([+-]?\d+\.?\d*)\b",
        r"answer(?: is|:)?\s*([+-]?\d+\.?\d*)\b",
        r"####\s*([+-]?\d+\.?\d*)\b",
        r"=>\s*([+-]?\d+\.?\d*)\b",
        r"=\s*([+-]?\d+\.?\d*)\b",
        r"result(?: is|:)?\s*([+-]?\d+\.?\d*)\b",
    ]
    for p in patterns:
        matches = re.findall(p, lower)
        if matches:
            return matches[-1].replace(",", "")
    nums = re.findall(r"[+-]?\d+\.?\d*", txt.replace(",", ""))
    return nums[-1] if nums else (txt.splitlines()[-1].strip() if txt.splitlines() else "")

def normalize_answer_for_em(ans):
    if ans is None:
        return ""
    s = str(ans).strip().rstrip(".")
    try:
        f = float(s.replace(",", ""))
        if f.is_integer():
            return str(int(f))
        return s.lower()
    except Exception:
        return s.lower()

def exact_match(gold, pred):
    if gold is None:
        return 0
    g = normalize_answer_for_em(gold)
    p = normalize_answer_for_em(pred)
    return 1 if g == p else 0

def load_test_lines(path):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            j = json.loads(ln)
            lines.append(j)
    return lines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local model path or HF id")
    ap.add_argument("--test", required=True, help="input JSONL with questions (id + question fields)")
    ap.add_argument("--out", required=True, help="output JSONL predictions")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--device", default="auto", help="'cuda' or 'cpu' or 'auto'")
    ap.add_argument("--prompt_template", default=None, help="Override default prompt template (use {q} for question)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Resolve prompt template without scoping issues
    prompt_tmpl = args.prompt_template or DEFAULT_PROMPT_TMPL

    torch.manual_seed(args.seed)

    # load tokenizer + model
    print("Loading tokenizer and model from", args.model, file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=True)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype="auto", token=True
    )

    device = next(model.parameters()).device
    print("Model parameters on", device, file=sys.stderr)

    print("Model loaded", file=sys.stderr)

    test_lines = load_test_lines(args.test)
    print(f"Loaded {len(test_lines)} test examples", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    total_em = 0
    pbar = tqdm(test_lines, desc="ZeroShotCoT")
    for ex in pbar:
        qid = ex.get("id", None)
        question = ex.get("question", ex.get("Q", ex.get("q", "")))
        prompt = prompt_tmpl.format(q=question)

        # Encode on CPU; device_map='auto' will handle execution placement
        # inputs = tokenizer(prompt, return_tensors="pt")
        inputs = tokenizer(prompt, return_tensors="pt")
        # move all input tensors to the model device (handles cpu/cuda/multi-device)
        inputs = {k: v.to(device) for k, v in inputs.items()}


        # Deterministic greedy decoding; temperature is ignored when do_sample=False
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Slice continuation by input length to exclude the prompt
        input_ids = inputs["input_ids"]
        cont_ids = gen_ids[:, input_ids.shape[1]:]
        generated = tokenizer.batch_decode(cont_ids, skip_special_tokens=True)[0].strip()

        # Split rationale vs final line and extract numeric
        lines = [l.strip() for l in generated.splitlines() if l.strip()]
        rationale = "\n".join(lines[:-1]) if len(lines) > 1 else generated
        final_answer = extract_final_answer(generated)

        gold_raw = (
            ex.get("answer")
            or ex.get("gold_answer")
            or ex.get("final_answer")
            or ex.get("answer_text")
            or ex.get("A")
            or ""
        )
        gold_extracted = extract_final_answer(gold_raw)
        em = exact_match(gold_extracted, final_answer) if gold_extracted != "" else None
        if em is not None:
            total_em += em

        outj = {
            "id": qid,
            "question": question,
            "generated": generated,
            "rationale": rationale,
            "answer": final_answer,
            "gold": gold_raw,
            "gold_extracted": gold_extracted,
            "em": em,
        }
        results.append(outj)

    with open(out_path, "w", encoding="utf-8") as w:
        for r in results:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_with_gold = sum(1 for r in results if r["em"] is not None)
    if n_with_gold:
        acc = total_em / n_with_gold
        print(f"Zero-shot EM on {n_with_gold} = {acc:.4%}", file=sys.stderr)
    else:
        print("No gold labels found to compute EM.", file=sys.stderr)

if __name__ == "__main__":
    main()
