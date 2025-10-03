# src/star_generate.py
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils import read_jsonl, write_jsonl, extract_final, exact_match
import torch

FORWARD_TMPL = open("prompts/forward.txt").read()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", default="run/star_forward.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")

    kept = []
    for ex in read_jsonl(args.train):
        prompt = FORWARD_TMPL.format(question=ex["question"])
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                             do_sample=True, temperature=args.temperature, top_p=args.top_p)
        text = tok.decode(out[0], skip_special_tokens=True)
        pred = extract_final(text)
        if exact_match(pred, ex["gold_answer"]):
            kept.append({"id": ex.get("id"), "question": ex.get("question"),
                         "rationale": text, "final": ex.get("gold_answer")})
    print(f"[star_generate] kept {len(kept)} items (out of attempted)")
    write_jsonl(args.out, kept)

if __name__ == "__main__":
    main()
