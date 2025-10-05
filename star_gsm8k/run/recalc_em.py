#!/usr/bin/env python3
import json, re, sys
from pathlib import Path

def extract_num_from_text(s):
    if s is None: return ""
    s = str(s).strip()
    low = s.lower()
    # Try explicit phrases first
    patterns = [
        r"final answer(?: is|:)?\s*([+-]?\d+\.?\d*)\b",
        r"answer(?: is|:)?\s*([+-]?\d+\.?\d*)\b",
        r"=\s*([+-]?\d+\.?\d*)\b",
        r"=>\s*([+-]?\d+\.?\d*)\b",
        r"####\s*([+-]?\d+\.?\d*)\b",   # handle "#### 3" style
        r"result(?: is|:)?\s*([+-]?\d+\.?\d*)\b",
    ]
    for p in patterns:
        m = re.search(p, low)
        if m:
            return m.group(1).replace(',', '')
    # fallback: last number token
    nums = re.findall(r"[+-]?\d+\.?\d*", s.replace(',', ''))
    if nums:
        return nums[-1]
    # fallback: last non-empty line
    lines = [l.strip() for l in s.splitlines() if l.strip()]
    if lines:
        return lines[-1]
    return ""

def normalize_numeric(a):
    if a is None: return ""
    a = str(a).strip().rstrip('.')
    # try numeric -> integer string
    try:
        f = float(a)
        i = int(round(f))
        return str(i)
    except:
        return a.lower()

preds_file = Path("run/zero_shot_preds_small.jsonl")
if not preds_file.exists():
    print("File not found:", preds_file)
    sys.exit(1)

total = 0
correct = 0
mismatches = []
with preds_file.open('r', encoding='utf-8') as f:
    for ln in f:
        j = json.loads(ln)
        gold_raw = j.get("gold") or j.get("answer") or j.get("A") or j.get("gold_answer") or ""
        gen_raw = j.get("generated", "")
        gnum = extract_num_from_text(gold_raw)
        pnum = extract_num_from_text(gen_raw)
        ng = normalize_numeric(gnum)
        npred = normalize_numeric(pnum)
        em = 1 if (ng != "" and npred != "" and ng == npred) else 0
        total += 1
        correct += em
        if em == 0:
            mismatches.append({
                "id": j.get("id"),
                "gold_raw": gold_raw,
                "gold_extracted": gnum,
                "gen_raw": gen_raw,
                "gen_extracted": pnum,
                "norm_gold": ng,
                "norm_pred": npred
            })

print(f"Recomputed EM on {total} = {correct}/{total} = {correct/total:.4%}")
print()
print("Showing up to 10 mismatches:")
import itertools, json as _json
for x in itertools.islice(mismatches, 10):
    print(_json.dumps(x, ensure_ascii=False))
