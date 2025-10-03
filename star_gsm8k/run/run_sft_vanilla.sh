#!/bin/bash
# Example: run vanilla SFT wrapper (replace class_sft_train CLI inside sft_train_wrapper.py first)
python -m src.sft_train_wrapper --train data/gsm8k_train_sample.jsonl --save_dir run/model_vanilla --base_model meta-llama/Llama-3.2-3B-Instruct --epochs 1
