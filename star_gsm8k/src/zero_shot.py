#!/usr/bin/env python3

import argparse, json, re, sys, math, os
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Prompt template for Zero-shot CoT
PROMPT_TMPL = "Q: {q}\nLet's think step by step.\nA:"

# simple function to extract the last numeric answer from generated text
def extract_final_answer(text):

    if text is None:
        return ""
    # common patterns:
    patterns = [
        r"answer(?: is|:)?\s*([-\d,\.]+)\b",
        r"final answer(?: is|:)?\s*([-\d,\.]+)\b",
        r"=\s*([-\d,\.]+)\b",
        r"=>\s*([-\d,\.]+)\b",
    ]
    lower = text.lower()
    for p in patterns:
        m = re.search(p, lower)
        if m:
            s = m.group(1)
            s = s.replace(',', '')
            return s.strip()
    # fallback: find last integer/float token
    nums = re.findall(r"[-]?\d+\.?\d*", text.replace(',', ''))
    if nums:
        return nums[-1]
    # nothing numeric found: return last non-empty line as answer (trim)
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""

def normalize_answer_for_em(ans):
    """Normalize extracted answer to an integer string if possible, else strip whitespace/lower."""
    if ans is None:
        return ""
    s = str(ans).strip()
    # strip trailing periods
    s = s.rstrip('.')
    # try int
    try:
        # handle floats that are integral like "5.0"
        f = float(s)
        i = int(round(f))
        return str(i)
    except Exception:
        pass
    # fallback normalize whitespace/case
    return s.lower()

def exact_match(gold, pred):
    # gold and pred are strings; try numeric match first
    if gold is None:
        return 0
    g = normalize_answer_for_em(gold)
    p = normalize_answer_for_em(pred)
    return 1 if g == p else 0

def load_test_lines(path):
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            if not ln.strip(): continue
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
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # load tokenizer + model
    print("Loading tokenizer and model from", args.model, file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_auth_token=True)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype="auto")
    device = next(model.parameters()).device
    print("Model parameters on", device, file=sys.stderr)

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
        prompt = PROMPT_TMPL.format(q=question)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # generate
        gen = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=False,  # deterministic greedy by default for EM evaluation; set True for creative outputs
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            output_scores=False,
            return_dict_in_generate=False,
        )
        text = tokenizer.decode(gen[0], skip_special_tokens=True)

        # ASSUME the model echoes the prompt; find the generated part after the prompt
        if text.startswith(prompt):
            generated = text[len(prompt):].strip()
        else:
            # fallback
            generated = text.strip()

        # try to split rationale vs final answer: naive heuristic:
        # - rationale = all but last line; final = last line or numeric extraction
        lines = [l.strip() for l in generated.splitlines() if l.strip()]
        rationale = "\n".join(lines[:-1]) if len(lines) > 1 else generated
        final_answer = extract_final_answer(generated)

        # gold answer: try fields gold_answer, answer, final_answer, or 'answer' inside ex
        gold = ex.get("answer") or ex.get("gold_answer") or ex.get("final_answer") or ex.get("answer_text") or ex.get("A") or None

        em = exact_match(gold, final_answer) if gold is not None else None
        if em is not None:
            total_em += em

        outj = {
            "id": qid,
            "question": question,
            "generated": generated,
            "rationale": rationale,
            "answer": final_answer,
            "gold": gold,
            "em": em,
        }
        results.append(outj)

    # write results
    with open(out_path, "w", encoding="utf-8") as w:
        for r in results:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    # summary
    n_with_gold = sum(1 for r in results if r["em"] is not None)
    if n_with_gold:
        acc = total_em / n_with_gold
        print(f"Zero-shot EM on {n_with_gold} = {acc:.4%}", file=sys.stderr)
    else:
        print("No gold labels found to compute EM.", file=sys.stderr)

if __name__ == "__main__":
    main()
