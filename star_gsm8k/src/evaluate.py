# src/evaluate.py
# Run forward prompt and compute EM. If test file is missing, you can pass --hf_test to load HF GSM8K test split.
import argparse, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils import read_jsonl, write_jsonl, extract_final, exact_match
import torch
from src.dataset_utils import load_gsm8k_split, write_class_jsonl_from_hf

DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
FORWARD_TMPL = open("prompts/forward.txt").read()

def load_test(test_path=None, use_hf=False, hf_limit=None):
    if test_path and os.path.exists(test_path):
        return list(read_jsonl(test_path))
    if use_hf:
        # build a temp HF test jsonl in memory
        ds = load_gsm8k_split("test")
        rows = []
        for i, ex in enumerate(ds):
            if hf_limit and i>=hf_limit: break
            rows.append({"id": f"hf_test_{i}", "question": ex["question"], "gold_answer": ex.get("answer","")})
        return rows
    raise FileNotFoundError("Test file not found and use_hf not set")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--test", default=None)
    ap.add_argument("--use_hf_test", action="store_true", help="Load GSM8K test from HF")
    ap.add_argument("--hf_limit", type=int, default=None)
    ap.add_argument("--out", default="run/eval_preds.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)
    args = ap.parse_args()

    # get test examples
    tests = load_test(args.test, use_hf=args.use_hf_test, hf_limit=args.hf_limit)

    tok = AutoTokenizer.from_pretrained(args.model, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto", use_auth_token=True)

    out_rows = []
    correct = 0
    for ex in tests:
        prompt = FORWARD_TMPL.format(question=ex["question"])
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=True, temperature=args.temperature, top_p=args.top_p)
        text = tok.decode(out_ids[0], skip_special_tokens=True)
        pred = extract_final(text)
        em = exact_match(pred, ex["gold_answer"])
        out_rows.append({"id": ex.get("id"), "question": ex.get("question"), "output": text, "pred_final": pred, "gold_answer": ex.get("gold_answer"), "em": em})
        correct += int(em)

    write_jsonl(args.out, out_rows)
    N = len(out_rows)
    print(f"Eval EM on {N} = {100.0*correct/max(1,N):.2f}% ({correct}/{N})")

if __name__ == "__main__":
    main()
