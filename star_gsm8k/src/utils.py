# src/utils.py
import re, json, os

FINAL_RE = re.compile(r"####\s*([\-]?\d+(?:\.\d+)?)")

def extract_final(text: str):
    if not text:
        return None
    m = FINAL_RE.findall(text)
    return m[-1].strip() if m else None

def normalize_answer(s: str):
    if s is None:
        return None
    s = s.strip().replace(",", "")
    if s.startswith("$"):
        s = s[1:]
    return s

def exact_match(pred: str | None, gold: str):
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    return (p is not None) and (p == g)

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
