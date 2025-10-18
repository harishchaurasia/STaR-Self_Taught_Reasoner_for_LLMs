# src/star_batch_driver.py
import argparse, os, json, subprocess
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils import read_jsonl, normalize_answer, exact_match_from_text

DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
FORWARD_TMPL = open("prompts/forward.txt").read()
RAT_TMPL = open("prompts/rationalize.txt").read()

def write_jsonl_append(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def forward_batch(model, tok, batch, max_new, n_candidates):
    prompts = [FORWARD_TMPL.format(question=ex["question"]) for ex in batch]
    inputs = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=False,
        num_beams=n_candidates,
        num_return_sequences=n_candidates,
        pad_token_id=tok.eos_token_id,
    )
    texts = tok.batch_decode(out_ids, skip_special_tokens=True)
    kept, missed = [], []
    K = n_candidates
    for j, ex in enumerate(batch):
        candidates = texts[j*K:(j+1)*K]
        kept_one = False
        for text in candidates:
            if exact_match_from_text(ex["answer"], text):
                kept.append({"id": ex["id"], "question": ex["question"], "rationale": text, "final": ex["answer"]})
                kept_one = True
                break
        if not kept_one:
            missed.append(ex)
    return kept, missed

def rationalize_batch(model, tok, missed_batch, max_new, n_candidates):
    prompts = [RAT_TMPL.format(question=ex["question"], gold_final=ex["answer"]) for ex in missed_batch]
    inputs = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=False,
        num_beams=n_candidates,
        num_return_sequences=n_candidates,
        pad_token_id=tok.eos_token_id,
    )
    texts = tok.batch_decode(out_ids, skip_special_tokens=True)
    kept = []
    K = n_candidates
    for j, ex in enumerate(missed_batch):
        candidates = texts[j*K:(j+1)*K]
        for text in candidates:
            if exact_match_from_text(ex["answer"], text):
                kept.append({"id": ex["id"], "question": ex["question"], "rationale": text, "final": ex["answer"]})
                break
    return kept

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--base_model", default=DEFAULT_MODEL)
    ap.add_argument("--workdir", default="run/star")
    ap.add_argument("--iter", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_candidates", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--preview_every", type=int, default=10, help="How many micro-batches before printing a summary")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    k = args.iter
    fw = f"{args.workdir}/iter{k}_forward.jsonl"
    rt = f"{args.workdir}/iter{k}_rationalized.jsonl"

    # fresh start
    for p in [fw, rt]:
        if os.path.exists(p):
            os.remove(p)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype="auto", device_map="auto")

    examples = list(read_jsonl(args.train))
    B, K = args.batch_size, args.n_candidates

    total_fwd_kept = 0
    total_rat_kept = 0
    for bidx in tqdm(range(0, len(examples), B), desc="STaR per-batch", ncols=80):
        batch = examples[bidx:bidx+B]

        fwd_kept, missed = forward_batch(model, tok, batch, args.max_new_tokens, K)
        write_jsonl_append(fw, fwd_kept)
        total_fwd_kept += len(fwd_kept)

        if missed:
            rat_kept = rationalize_batch(model, tok, missed, args.max_new_tokens, K)
            write_jsonl_append(rt, rat_kept)
            total_rat_kept += len(rat_kept)

        if ((bidx // B) + 1) % args.preview_every == 0:
            print(f"[progress] batches={(bidx // B)+1}  fwd_kept={total_fwd_kept}  rat_kept={total_rat_kept}  missed_last_batch={len(missed)}")

    # Build SFT data at the end (or call periodically if desired)
    subprocess.check_call([
        "python3", "-m", "src.star_build",
        "--forward", fw,
        "--rationalized", rt,
        "--out", f"{args.workdir}/iter{k}_train.jsonl"
    ])

    # Then call your class SFT script and evaluation just like in your main driver, if desired.

if __name__ == "__main__":
    main()
