# # utils.py
# import re, json, os
# FINAL_RE = re.compile(r"####\s*([\-]?\d+(?:\.\d+)?)")
# NUM_RE   = re.compile(r"[\-]?\d+(?:\.\d+)?")

# def extract_final(text: str):
#     if not text:
#         return None
#     m = FINAL_RE.findall(text)
#     return m[-1].strip() if m else None

# def flexible_extract(text: str):
#     if not text:
#         return None
#     nums = NUM_RE.findall(text)
#     return nums[-1].strip() if nums else None

# def normalize_answer(s: str):
#     if s is None:
#         return None
#     s = s.strip().replace(",", "")
#     if s.startswith("$"):
#         s = s[1:]
#     return s

# def exact_match_from_text(gold: str, text: str):
#     # Try strict #### first, then flexible fallback (SFT-style)
#     pred = extract_final(text)
#     if pred is None:
#         pred = flexible_extract(text)
#     p = normalize_answer(pred)
#     g = normalize_answer(gold)
#     return (p is not None) and (p == g)

# src/utils.py
import re, json, os
FINAL_RE = re.compile(r"####\s*([\-]?\d+(?:\.\d+)?)")
NUM_RE   = re.compile(r"[\-]?\d+(?:\.\d+)?")

def extract_final(text: str):
    if not text:
        return None
    m = FINAL_RE.findall(text)
    return m[-1].strip() if m else None

def flexible_extract(text: str):
    if not text:
        return None
    nums = NUM_RE.findall(text)
    return nums[-1].strip() if nums else None

def normalize_answer(s: str):
    if s is None:
        return None
    s = s.strip().replace(",", "")
    if s.startswith("$"):
        s = s[1:]
    return s

def exact_match_from_text(gold: str, text: str):
    pred = extract_final(text)
    if pred is None:
        pred = flexible_extract(text)
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    return (p is not None) and (p == g)

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
