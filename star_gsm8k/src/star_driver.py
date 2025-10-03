# src/star_driver.py
import argparse, os, subprocess

def sh(*args):
    print("[run]", " ".join(args))
    return subprocess.check_call(list(args))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)    # gsm8k train jsonl with gold answers (id,question,gold_answer)
    ap.add_argument("--test", required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--workdir", default="run/star")
    ap.add_argument("--iter", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    k = args.iter
    fw = f"{args.workdir}/iter{k}_forward.jsonl"
    rt = f"{args.workdir}/iter{k}_rationalized.jsonl"
    tr = f"{args.workdir}/iter{k}_train.jsonl"
    ck = f"{args.workdir}/model_iter_{k}"

    # 1) forward (use base model for generation at iter 1; paper uses cur model but restarts training from base)
    sh("python", "-m", "src.star_generate", "--model", args.base_model, "--train", args.train, "--out", fw)

    # 2) rationalize on missed
    sh("python", "-m", "src.star_rationalize", "--model", args.base_model, "--train", args.train, "--missed", fw, "--out", rt)

    # 3) build SFT data
    sh("python", "-m", "src.star_build", "--forward", fw, "--rationalized", rt, "--out", tr)

    # 4) fine-tune FROM BASE
    sh("python", "-m", "src.sft_train_wrapper", "--train", tr, "--save_dir", ck, "--base_model", args.base_model)

    # 5) evaluate
    sh("python", "-m", "src.evaluate", "--model", ck, "--test", args.test, "--out", f"{args.workdir}/iter{k}_preds.jsonl")

if __name__ == "__main__":
    main()
