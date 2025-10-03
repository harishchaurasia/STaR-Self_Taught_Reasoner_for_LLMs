# src/sft_train_wrapper.py
# Sanity-check SFT input and call the class SFT trainer script.
# WARNING: adjust the "cmd" list to your class script's exact flags if they differ.
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

    # sanity-check
    n=0
    for r in read_jsonl(args.train):
        t=r.get("text","")
        assert "###Input:" in t and "###Output:" in t and "#### " in t, "Bad training line format"
        n+=1
    print(f"[sft_wrapper] training file OK with {n} lines")

    # TODO: adapt this command if your class trainer expects different flags.
    # This example assumes your class script is cse_576_inference_and_training.py and uses flags:
    # --train_file, --output_dir, --pretrained, --num_epochs, --learning_rate, --max_seq_len, --batch_size, --grad_accum
    cmd = [
      "python", "cse_576_inference_and_training.py",
      "--train_file", args.train,
      "--output_dir", args.save_dir,
      "--pretrained", args.base_model,
      "--num_epochs", str(args.epochs),
      "--learning_rate", str(args.lr),
      "--max_seq_len", str(args.seq_len),
      "--batch_size", str(args.per_device_batch),
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
