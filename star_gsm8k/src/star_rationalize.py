# src/star_rationalize.py
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils import read_jsonl, write_jsonl, extract_final, exact_match
import torch

RAT_TMPL = open("prompts/rationalize.txt").read()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)      # full train jsonl
    ap.add_argument("--missed", required=True)     # forward kept file path (so we know misses)
    ap.add_argument("--out", default="run/star_rationalized.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")

    kept_ids = {r["id"] for r in read_jsonl(args.missed)}
    pool = [r for r in read_jsonl(args.train) if r["id"] not in kept_ids]

    kept = []
    for ex in pool:
        prompt = RAT_TMPL.format(question=ex["question"], gold_final=ex["gold_answer"])
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                             do_sample=True, temperature=args.temperature, top_p=args.top_p)
        text = tok.decode(out[0], skip_special_tokens=True)
        pred = extract_final(text)
        if exact_match(pred, ex["gold_answer"]):
            # Save rationale text (but NOT the hint); we save model output as rationale, but
            # when building SFT targets we'll only store the rationale lines (no hint).
            kept.append({"id": ex.get("id"), "question": ex.get("question"),
                         "rationale": text, "final": ex.get("gold_answer")})
    print(f"[star_rationalize] kept {len(kept)} items")
    write_jsonl(args.out, kept)

if __name__ == "__main__":
    main()
