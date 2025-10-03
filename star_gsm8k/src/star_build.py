# src/star_build.py
# Merge forward-kept and rationalized-kept into class-format SFT JSONL
import argparse
from src.utils import read_jsonl, write_jsonl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward", required=True)
    ap.add_argument("--rationalized", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    keep={}
    for path in [args.forward, args.rationalized]:
        for r in read_jsonl(path):
            keep[r["id"]] = r

    rows=[]
    for r in keep.values():
        text = f"###Input:\n{r['question']}\n\n###Output:\n{r['rationale']}\n#### {r['final']}"
        rows.append({"id": r["id"], "text": text})
    write_jsonl(args.out, rows)
    print(f"[star_build] wrote {len(rows)} lines to {args.out}")

if __name__ == "__main__":
    main()
