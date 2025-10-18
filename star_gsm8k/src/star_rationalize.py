import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils import read_jsonl, write_jsonl, extract_final, exact_match
import torch
from tqdm import tqdm

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
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, token=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left" 
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", device_map="auto", token=True)

    missed_ids = {r["id"] for r in read_jsonl(args.missed)}
    pool = [r for r in read_jsonl(args.train) if r["id"] not in missed_ids]

    kept = []
    batch_size = args.batch_size

    for i in tqdm(range(0, len(pool), batch_size), desc="Rationalizing", ncols=80):
        batch = pool[i:i+batch_size]
        prompts = [RAT_TMPL.format(question=ex["question"], gold_final=ex["answer"]) for ex in batch]
        inputs = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        out_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tok.eos_token_id
        )
        texts = tok.batch_decode(out_ids, skip_special_tokens=True)
        for ex, text in zip(batch, texts):
            pred = extract_final(text)
            if exact_match(pred, ex["answer"]):
                kept.append({"id": ex["id"], "question": ex["question"], "rationale": text, "final": ex["answer"]})
    write_jsonl(args.out, kept)
    print(f"[star_rationalize] kept {len(kept)} items")

if __name__ == "__main__":
    main()
