#!/bin/bash
python -m src.star_driver --train data/gsm8k_train_sample.jsonl --test data/gsm8k_test_small.jsonl --base_model meta-llama/Llama-3.2-3B-Instruct --workdir run/star --iter 1
