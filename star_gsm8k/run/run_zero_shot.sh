#!/bin/bash
# run locally or on Sol (change --model)
python -m src.zero_shot --model gpt2 --test data/gsm8k_test_small.jsonl --out run/zero_shot_preds.jsonl
