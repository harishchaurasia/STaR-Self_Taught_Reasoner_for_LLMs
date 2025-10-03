# src/zero_shot.py
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils import read_jsonl, write_jsonl, extract_final, exact_match
import torch

FORWARD_TMPL = open("prompts/forward.txt").read()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")  # change on Sol to meta-llama/Llama-3.2-3B-Instruct
    ap.add_argument("--test", default="data/gsm8k_test_small.jsonl")
    ap.add_argument("--out", default="run/zero_shot_preds.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")

    rows, correct, total = [], 0, 0
    for ex in read_jsonl(args.test):
        prompt = FORWARD_TMPL.format(question=ex["question"])
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                             do_sample=True, temperature=args.temperature, top_p=args.top_p)
        text = tok.decode(out[0], skip_special_tokens=True)
        pred = extract_final(text)
        ok = exact_match(pred, ex["gold_answer"])
        total += 1
        correct += int(ok)
        rows.append({
            "id": ex.get("id"),
            "question": ex.get("question"),
            "output": text,
            "pred_final": pred,
            "gold_answer": ex.get("gold_answer"),
            "em": ok
        })

    acc = 100.0 * correct / max(1, total)
    print(f"Zero-shot EM on {total} = {acc:.2f}%")
    write_jsonl(args.out, rows)

if __name__ == "__main__":
    main()
