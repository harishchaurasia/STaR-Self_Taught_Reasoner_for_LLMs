#!/usr/bin/env bash
set -euo pipefail
ROOT="$(pwd)"
mkdir -p run data

# ensure small dev file (20 examples)
if [ ! -f data/gsm8k_test_small.jsonl ]; then
  if [ -f data/gsm8k_test.jsonl ]; then
    head -n 20 data/gsm8k_test.jsonl > data/gsm8k_test_small.jsonl
    echo "Created data/gsm8k_test_small.jsonl from full test file."
  else
    echo "ERROR: data/gsm8k_test.jsonl not found. Please create it first." >&2
    exit 1
  fi
fi

MODEL="meta-llama/Llama-3.2-3B-Instruct"
MAXTOK=160
OUT_PREFIX="run/zero_shot_v"

# write prompts to temp files to avoid quoting issues
cat > /tmp/prompt1.txt <<'PROMPT'
Q: {q}
Let's think step by step.
A:
PROMPT

cat > /tmp/prompt2.txt <<'PROMPT'
Q: {q}
Let's think step by step.
A:
(Please finish with a final line starting with "Final answer:" followed by the numeric answer.)
PROMPT

cat > /tmp/prompt3.txt <<'PROMPT'
Q: {q}
Let's think step by step.
A:
(Show your steps and then on the last line write EXACTLY: Final answer: <number>)
PROMPT

FILES=( "run/zero_shot_v1.jsonl" "run/zero_shot_v2.jsonl" "run/zero_shot_v3.jsonl" )
TMPS=( "/tmp/prompt1.txt" "/tmp/prompt2.txt" "/tmp/prompt3.txt" )

for i in 0 1 2; do
  idx=$((i+1))
  outf="${FILES[i]}"
  prompt_file="${TMPS[i]}"
  echo "=== Running variant $idx -> ${outf} ==="
  python3 src/zero_shot_cot.py \
    --model "$MODEL" \
    --test data/gsm8k_test_small.jsonl \
    --out "$outf" \
    --max_new_tokens "$MAXTOK" \
    --prompt_template "$(cat "$prompt_file")"
  echo "Finished variant $idx."
done

# compute EM + show mismatches
python3 - <<'PY'
import json,re
files=["run/zero_shot_v1.jsonl","run/zero_shot_v2.jsonl","run/zero_shot_v3.jsonl"]
def extract_num(s):
    if s is None: return ""
    s=str(s)
    low=s.lower()
    patterns=[r"final answer(?: is|:)?\s*([+-]?\d+\.?\d*)\b",
              r"answer(?: is|:)?\s*([+-]?\d+\.?\d*)\b",
              r"=\s*([+-]?\d+\.?\d*)\b",
              r"=>\s*([+-]?\d+\.?\d*)\b",
              r"####\s*([+-]?\d+\.?\d*)\b"]
    for p in patterns:
        m=re.search(p, low)
        if m: return m.group(1)
    nums=re.findall(r"[+-]?\d+\.?\d*", s.replace(',',''))
    if nums: return nums[-1]
    lines=[l.strip() for l in s.splitlines() if l.strip()]
    return lines[-1] if lines else ""
def norm(x):
    if x is None: return ""
    x=str(x).strip().rstrip('.')
    try:
        f=float(x); return str(int(round(f)))
    except:
        return x.lower()
for fn in files:
    total=0; corr=0; mism=[]
    with open(fn,'r',encoding='utf-8') as f:
        for ln in f:
            j=json.loads(ln)
            gold=j.get("gold") or j.get("answer") or ""
            generated=j.get("generated","")
            g=norm(extract_num(gold))
            p=norm(extract_num(generated))
            total+=1
            if g!="" and p!="" and g==p:
                corr+=1
            else:
                mism.append({"id":j.get("id"), "gold_raw": gold, "gold_ex": g, "pred_raw": generated, "pred_ex": p})
    print("="*70)
    print(f"File: {fn}")
    print(f"EM: {corr}/{total} = {corr/total:.4%}")
    print("Showing up to 6 mismatches (gold_raw | gold_extracted => pred_extracted | pred_raw):")
    import itertools, json as _json
    for x in itertools.islice(mism,6):
        print(_json.dumps(x, ensure_ascii=False))
PY

echo "DONE."
