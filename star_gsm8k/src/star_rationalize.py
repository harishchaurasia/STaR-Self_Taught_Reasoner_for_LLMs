# src/star_rationalize.py
# For items missed by forward generation, ask the model (with hint) to produce a rationale that ends at the gold final.
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils import read_jsonl, write_jsonl, extract_final, exact_match
import torch

DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
RAT_TMPL = open("prompts/rationalize.txt").read()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--train", required=True)
    ap.add_argument("--missed", required=True)
    ap.add_argument("--out", default="run/star_rationalized.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto", use_auth_token=True)

    missed_ids = {r["id"] for r in read_jsonl(args.missed)}
    pool = [r for r in read_jsonl(args.train) if r["id"] not in missed_ids]

    kept=[]
    for ex in pool:
        prompt = RAT_TMPL.format(question=ex["question"], gold_final=ex["gold_answer"])
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=True, temperature=args.temperature, top_p=args.top_p)
        text = tok.decode(out_ids[0], skip_special_tokens=True)
        pred = extract_final(text)
        if exact_match(pred, ex["gold_answer"]):
            kept.append({"id": ex["id"], "question": ex["question"], "rationale": text, "final": ex["gold_answer"]})
    write_jsonl(args.out, kept)
    print(f"[star_rationalize] kept {len(kept)} items")
if __name__ == "__main__":
    main()
