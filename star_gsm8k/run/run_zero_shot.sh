#!/usr/bin/env bash
# run_zero_shot.sh
# Run Zero-Shot CoT evaluation

python3 ../src/zero_shot_cot.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --test ../data/gsm8k_test_small.jsonl \
  --out zero_shot_preds_small.jsonl \
  --max_new_tokens 160 \
  --temperature 0.0
