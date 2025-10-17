#!/usr/bin/env python3

import argparse, json, re, sys, os
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# System/env speedups
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cuda.matmul.allow_tf32 = True

# Prompt template for Zero-shot CoT with explicit final line
DEFAULT_PROMPT_TMPL = (
    "Q: {q}\n"
    "Let's think step by step.\n"
    "Please give the final answer as '#### <number>' on the last line.\n"
    "A:"
)

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
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--device", default="auto", help="'cuda' or 'cpu' or 'auto'")
    ap.add_argument("--prompt_template", default=None, help="Override default prompt template (use {q} for question)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_compile", action="store_true", help="disable torch.compile if set")
    args = ap.parse_args()

    # Resolve prompt template
    prompt_tmpl = args.prompt_template or DEFAULT_PROMPT_TMPL

    # Reproducibility
    torch.manual_seed(args.seed)

    print("Loading tokenizer and model from", args.model, file=sys.stderr)

    # Load tokenizer from the same source as model for alignment
    tok_src = args.model
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    attn_impl = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except Exception:
        attn_impl = "sdpa" 

    # Load model on A100 with bf16 + FlashAttention-2
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )

    # before compiling
    try:
        import torch._inductor.config as inductor_config
        inductor_config.triton.cudagraphs = False
    except Exception:
        pass

    if not args.no_compile:
        try:
            model.forward = torch.compile(
                model.forward,
                mode="reduce-overhead",
                fullgraph=True,
                options={"triton.cudagraphs": False},  # disable CUDA Graphs
            )
        except Exception:
            pass


    # Prefer static cache for repeated batched decoding
    # try:
    #     model.generation_config.cache_implementation = "static"
    # except Exception:
    #     pass

    model.eval()
    device = next(model.parameters()).device
    print("Model parameters on", device, file=sys.stderr)
    print("Model loaded", file=sys.stderr)

    test_lines = load_test_lines(args.test)
    print(f"Loaded {len(test_lines)} test examples", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    total_em = 0
    n_with_gold = 0

    # Batched generation loop
    pbar = tqdm(range(0, len(test_lines), args.batch), desc="ZeroShotCoT")
    with torch.no_grad():
        for start in pbar:
            chunk = test_lines[start:start + args.batch]
            questions = [
                ex.get("question", ex.get("Q", ex.get("q", ""))) for ex in chunk
            ]
            prompts = [prompt_tmpl.format(q=q) for q in questions]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                pad_to_multiple_of=8,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            gen_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # deterministic greedy decoding
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            input_len = inputs["input_ids"].shape[1]
            cont_ids = gen_ids[:, input_len:]
            generations = tokenizer.batch_decode(cont_ids, skip_special_tokens=True)

            for ex, generated in zip(chunk, generations):
                generated = generated.strip()
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
                    n_with_gold += 1

                results.append({
                    "id": ex.get("id", None),
                    "question": ex.get("question", ex.get("Q", ex.get("q", ""))),
                    "generated": generated,
                    "rationale": rationale,
                    "answer": final_answer,
                    "gold": gold_raw,
                    "gold_extracted": gold_extracted,
                    "em": em,
                })

    with open(out_path, "w", encoding="utf-8") as w:
        for r in results:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    if n_with_gold:
        acc = total_em / n_with_gold
        print(f"Zero-shot EM on {n_with_gold} = {acc:.4%}", file=sys.stderr)
    else:
        print("No gold labels found to compute EM.", file=sys.stderr)

if __name__ == "__main__":
    main()
