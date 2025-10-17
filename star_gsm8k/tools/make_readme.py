#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

def load_results(path):
    """
    Load a JSONL of per-example predictions.
    Expects each line to optionally contain: 'em' (0/1), 'generated', etc.
    Returns summary metrics and counters for parseability.
    """
    n_gold, correct = 0, 0
    total, n_parse_ok = 0, 0
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            j = json.loads(ln)
            total += 1
            em = j.get("em")
            if em is not None:
                n_gold += 1
                if em == 1:
                    correct += 1
            gen = (j.get("generated") or "").strip()
            if "####" in gen:
                n_parse_ok += 1
    acc = (correct / n_gold) if n_gold > 0 else 0.0
    return {
        "path": str(path),
        "total": total,
        "n_gold": n_gold,
        "correct": correct,
        "accuracy": acc,
        "parse_ok": n_parse_ok,
    }

def percent(x, d=2):
    return f"{100.0 * x:.{d}f}%"

def add_fenced_block(lines, content, lang=""):
    """
    Append a Markdown fenced code block safely without embedding raw backticks
    inside Python string delimiters.
    """
    fence_open = "```
    lines.append(fence_open)
    for ln in str(content).splitlines():
        lines.append(ln)
    lines.append("```")

def main():
    ap = argparse.ArgumentParser(description="Generate README.md with results table from JSONL outputs.")
    ap.add_argument("--exp", action="append", default=[],
                    help='Experiment spec: "Label:path/to/results.jsonl" (repeatable)')
    ap.add_argument("--out", default="README.md", help="Output README markdown path")
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct", help="Model identifier/checkpoint")
    ap.add_argument("--prompt_text", default=None, help="Inline prompt text to include in README")
    ap.add_argument("--prompt_file", default=None, help="Path to a file containing the prompt text")
    ap.add_argument("--gen_config", default="greedy (do_sample=False), max_new_tokens=512, bf16, left-padding, batch=16",
                    help="Short description of generation settings")
    ap.add_argument("--train_config", default=None,
                    help="Short description of SFT/STaR training settings (optional)")
    args = ap.parse_args()

    if not args.exp:
        print("Provide at least one --exp 'Label:path.jsonl'", file=sys.stderr)
        sys.exit(1)

    rows = []
    for spec in args.exp:
        if ":" not in spec:
            print(f"Bad --exp format: {spec}. Use 'Label:path.jsonl'", file=sys.stderr)
            sys.exit(2)
        label, path = spec.split(":", 1)
        path = path.strip()
        if not os.path.exists(path):
            print(f"Missing results file: {path}", file=sys.stderr)
            sys.exit(3)
        r = load_results(path)
        rows.append({
            "label": label.strip(),
            "path": path,
            "accuracy": r["accuracy"],
            "n_gold": r["n_gold"],
            "total": r["total"],
            "parse_ok": r["parse_ok"],
        })

    # Load prompt text
    prompt_txt = ""
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt_txt = f.read().strip()
    elif args.prompt_text:
        prompt_txt = args.prompt_text.strip()

    # Write README
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# GSM8K Results ({now})")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- Model: `{args.model}`")
    lines.append(f"- Generation: {args.gen_config}")
    if args.train_config:
        lines.append(f"- Training: {args.train_config}")
    lines.append("")

    if prompt_txt:
        lines.append("## Prompt")
        lines.append("")
        add_fenced_block(lines, prompt_txt, lang="")
        lines.append("")

    lines.append("## Results (Exact Match)")
    lines.append("")
    lines.append("| Method | EM | Evaluated | Parseable Final Line | File |")
    lines.append("|:--|--:|--:|--:|:--|")
    for r in rows:
        lines.append(
            f"| {r['label']} | {percent(r['accuracy'])} | {r['n_gold']} / {r['total']} | {r['parse_ok']} | `{r['path']}` |"
        )
    lines.append("")
    lines.append("## How to Reproduce")
    lines.append("")
    lines.append("- Use the official GSM8K test split (1,319 problems) and the same prompt/decoding settings across methods.")
    lines.append("- Ensure the final line is formatted as `#### <number>` so exact-match extraction is reliable.")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- EM is computed from the per-example `em` field written by the inference script; if missing, examples are excluded from the EM denominator.")
    lines.append("- The 'Parseable Final Line' column counts generations containing `####`, useful for diagnosing extraction issues.")

    with open(out_path, "w", encoding="utf-8") as w:
        w.write("\n".join(lines))

    print(f"Wrote {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
