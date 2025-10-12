#!/bin/bash
# Example: run vanilla SFT wrapper (replace class_sft_train CLI inside sft_train_wrapper.py first)
# python -m src.sft_train_wrapper --train data/gsm8k_train_sample.jsonl --save_dir run/model_vanilla --base_model meta-llama/Llama-3.2-3B-Instruct --epochs 1

# python3 src/vanilla_sft.py --subset 500 --epochs 1 --batch 1 --grad_accum 8 --out checkpoints/vanilla_sft_direct_500
# python3 src/vanilla_sft.py --subset 500 --epochs 1 --batch 1 --grad_accum 8 --max_seq_len 256 --fp16 --out checkpoints/test_shorten_seq
# python3 src/vanilla_sft.py --subset 500 --epochs 1 --batch 1 --grad_accum 2 --max_seq_len 256 --fp16 --out checkpoints/gc_run

python3 src/vanilla_sft.py --subset 0 --epochs 1 --batch 1 --grad_accum 4 --lr 1e-5 --fp16 --out checkpoints/sft_test_run
