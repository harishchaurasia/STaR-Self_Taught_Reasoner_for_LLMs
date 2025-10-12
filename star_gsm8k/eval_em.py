#!/usr/bin/env python3
# eval_em.py
import json, re, sys
from pathlib import Path

# --- utility functions (same logic as your scripts) ---
def extract_final_answer(text):
    if text is None:
        return ""
    txt = str(text).strip()
    lower = txt.lower()
    patterns = [
        r"final answer(?: is|:)?\s*([+-]?\d+\.?\d*)\b",
        r"answer(?: is|:)?\s*([+-]?\d+\.?\d*)\b",
        r"####\s*([+-]?\d+\.?\d*)\b",
        r"=>\s*([+-]?\d+\.?\d*)\b",
        r"=\s*([+-]?\d+\.?\d*)\b",
        r"result(?: is|:)?\s*([+-]?\d+\.?\d*)\b",
    ]
    for p in patterns:
        matches = re.findall(p, lower)
        if matches:
            return matches[-1].replace(",", "")
    nums = re.findall(r"[+-]?\d+\.?\d*", txt.replace(",", ""))
    if nums:
        return nums[-1]
    # fallback: last non-empty line
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    return lines[-1] if lines else ""

def normalize_answer_for_em(ans):
    if ans is None:
        return ""
    s = str(ans).strip().rstrip(".")
    try:
        f = float(s.replace(",", ""))
        if f.is_integer():
            return str(int(round(f)))
        return s.lower()
    except Exception:
        return s.lower()

def exact_match(gold, pred):
    if gold is None:
        return 0
    g = normalize_answer_for_em(gold)
    p = normalize_answer_for_em(pred)
    return 1 if g == p else 0

# --- main ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 eval_em.py predictions.jsonl [--save-mismatches out.jsonl] [--top N]")
        sys.exit(1)

    preds_path = Path(sys.argv[1])
    save_mismatches = None
    top_n = 20
    if "--save-mismatches" in sys.argv:
        i = sys.argv.index("--save-mismatches")
        save_mismatches = Path(sys.argv[i+1])
    if "--top" in sys.argv:
        i = sys.argv.index("--top")
        top_n = int(sys.argv[i+1])

    results = []
    with preds_path.open("r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip(): continue
            j = json.loads(ln)
            # adjust these keys if your pred file uses different names
            gold_raw = j.get("gold") or j.get("answer") or j.get("final_answer") or j.get("answer_text") or ""
            pred_raw = j.get("answer") or j.get("generated") or j.get("final_answer") or ""
            # if your script already extracted 'answer' and 'gold_extracted' use those instead:
            # gold_raw = j.get("gold_extracted", gold_raw)
            # pred_raw = j.get("answer", pred_raw)

            gold_ex = extract_final_answer(gold_raw) if gold_raw else gold_raw
            pred_ex = extract_final_answer(pred_raw) if pred_raw else pred_raw

            em = None
            if gold_raw is not None and gold_raw != "":
                em = exact_match(gold_ex, pred_ex)

            results.append({
                "id": j.get("id"),
                "question": j.get("question"),
                "gold_raw": gold_raw,
                "gold_extracted": gold_ex,
                "pred_raw": pred_raw,
                "pred_extracted": pred_ex,
                "em": em,
                "generated": j.get("generated", j.get("rationale", "") + "\n" + str(pred_raw))
            })

    n_with_gold = sum(1 for r in results if r["em"] is not None)
    total_em = sum(r["em"] for r in results if r["em"] is not None)
    acc = (total_em / n_with_gold) if n_with_gold>0 else 0.0
    print(f"Total examples: {len(results)}; with gold: {n_with_gold}; EM = {total_em}/{n_with_gold} = {acc:.4%}")

    # show top mismatches
    mismatches = [r for r in results if r["em"] == 0]
    print(f"Mismatches: {len(mismatches)} (showing up to {top_n})")
    for i, r in enumerate(mismatches[:top_n]):
        print("-----")
        print("id:", r["id"])
        print("q:", r["question"])
        print("gold_extracted:", r["gold_extracted"], "gold_raw:", r["gold_raw"])
        print("pred_extracted:", r["pred_extracted"], "pred_raw (short):", (r["pred_raw"][:300] + "...") if len(r["pred_raw"])>300 else r["pred_raw"])
        print()

    if save_mismatches:
        with save_mismatches.open("w", encoding="utf-8") as out:
            for r in mismatches:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("Saved mismatches to", save_mismatches)
