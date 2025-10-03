# src/sft_train_wrapper.py
import argparse, subprocess, sys
from src.utils import read_jsonl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--per_device_batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=2)
    args = ap.parse_args()

    # quick sanity check
    n = 0
    for r in read_jsonl(args.train):
        t = r.get("text", "")
        assert "###Input:" in t and "###Output:" in t and "#### " in t, "Bad training line format"
        n += 1
    print(f"[sft_wrapper] training file OK with {n} lines")

    # TODO: replace the example below with the exact class script command and flags you used in class.
    # Example placeholder:
    cmd = [
        "python", "class_sft_train.py",
        "--base_model", args.base_model,
        "--train_file", args.train,
        "--save_dir", args.save_dir,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--seq_len", str(args.seq_len),
        "--per_device_batch", str(args.per_device_batch),
        "--grad_accum", str(args.grad_accum),
    ]
    print("[sft_wrapper] running:", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        print("[sft_wrapper] ERROR: class trainer returned non-zero", rc)
        sys.exit(rc)
    print("[sft_wrapper] done")

if __name__ == "__main__":
    main()
