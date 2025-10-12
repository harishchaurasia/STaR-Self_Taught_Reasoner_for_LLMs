#!/usr/bin/env bash
# run_zero_shot.sh
# Run Zero-Shot CoT evaluation

# python3 ../src/zero_shot_cot.py \
#   --model meta-llama/Llama-3.2-3B-Instruct \
#   --test ../data/gsm8k_test_small.jsonl \
#   --out zero_shot_preds_small.jsonl \
#   --max_new_tokens 160 \
#   --temperature 0.0


# python3 ../src/zero_shot_cot.py \
#   --model meta-llama/Llama-3.2-3B-Instruct \
#   --test data/gsm8k_test.jsonl \
#   --out run/zero_shot_full_greedy.jsonl \
#   --max_new_tokens 256 \
#   --prompt_template "Q: {q}\nLet's think step by step.\nA:\n(Please finish with a final line starting with 'Final answer:' followed by the numeric answer.)"


  # python3 src/zero_shot_cot.py \
  # --model meta-llama/Llama-3.2-3B-Instruct \
  # --test data/gsm8k_test.jsonl \
  # --out run/zero_shot_full_greedy.jsonl \
  # --max_new_tokens 256 \
  # --prompt_template "Q: {q}\nLet's think step by step.\nA:\n(Please finish with a final line starting with 'Final answer:' followed by the numeric answer.)"

  python3 src/vanilla_sft.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --subset 100 \
  --epochs 1 \
  --batch 1 \
  --grad_accum 4 \
  --lr 1e-5 \
  --fp16 \
  --out checkpoints/sft_test_run


