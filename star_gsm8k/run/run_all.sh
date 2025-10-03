#!/usr/bin/env bash
# run_all.sh - runs Zero-shot, Vanilla SFT, and STaR on GSM8K (full if SUBSET_SIZE=0)
set -euo pipefail

BASE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
TRAIN_JSON="data/gsm8k_train.jsonl"   # optional: if absent script will try to build from HF
TEST_JSON="data/gsm8k_test.jsonl"     # optional
WORKDIR="run"
mkdir -p "${WORKDIR}"

# safe defaults (edit as needed)
SUBSET_SIZE=0           # 0 => use full train from TRAIN_JSON; >0 => subsample for fast run
ITERATIONS=2
GEN_MAX_TOK=256
GEN_TEMP=0.2
GEN_TOPP=0.95
SFT_EPOCHS=1
SFT_LR=1e-5
SFT_SEQ_LEN=512
SFT_BATCH=4
SFT_GRAD_ACC=2

read -p "Proceed with run_all using SUBSET_SIZE=${SUBSET_SIZE}? (y/n) " ok
if [[ "${ok}" != "y" ]]; then echo "Aborting"; exit 1; fi

# If TRAIN_JSON not provided, build from HF GSM8K
if [[ ! -f "${TRAIN_JSON}" ]]; then
  echo "[run_all] building full train file from HF GSM8K -> ${TRAIN_JSON}"
  python - <<PY
from src.dataset_utils import write_class_jsonl_from_hf
write_class_jsonl_from_hf("train", "${TRAIN_JSON}", limit=None)
print("done")
PY
fi

if [[ ! -f "${TEST_JSON}" ]]; then
  echo "[run_all] building test jsonl from HF GSM8K -> ${TEST_JSON}"
  python - <<PY
from src.dataset_utils import write_class_jsonl_from_hf
write_class_jsonl_from_hf("test", "${TEST_JSON}", limit=None)
print("done")
PY
fi

# subset handling
if [[ ${SUBSET_SIZE} -gt 0 ]]; then
  echo "[run_all] creating subset ${WORKDIR}/train_subset.jsonl"
  python - <<PY
import json
n=int(${SUBSET_SIZE}); i=0
with open("${TRAIN_JSON}",'r',encoding='utf-8') as r, open("${WORKDIR}/train_subset.jsonl",'w',encoding='utf-8') as w:
    for line in r:
        if not line.strip(): continue
        if i>=n: break
        w.write(line); i+=1
print("wrote", i)
PY
  TRAIN_USE="${WORKDIR}/train_subset.jsonl"
else
  TRAIN_USE="${TRAIN_JSON}"
fi

# 1) Zero-shot evaluation
echo "[run_all] STEP 1: Zero-shot CoT (base model)"
python -m src.evaluate --model "${BASE_MODEL}" --test "${TEST_JSON}" --out "${WORKDIR}/zero_shot.jsonl" --max_new_tokens ${GEN_MAX_TOK} --temperature ${GEN_TEMP} --top_p ${GEN_TOPP}

# 2) Vanilla SFT
echo "[run_all] STEP 2: Vanilla SFT (train on gold rationales)"
VAN="${WORKDIR}/vanilla_train.jsonl"
python - <<PY
import json
src="${TRAIN_USE}"; dst="${VAN}"
with open(dst,'w',encoding='utf-8') as w:
    for line in open(src,'r',encoding='utf-8'):
        if not line.strip(): continue
        j=json.loads(line)
        q=j.get("question")
        gr=j.get("gold_rationale", j.get("rationale",""))
        ga=j.get("gold_answer", j.get("final",""))
        text=f"###Input:\\n{q}\\n\\n###Output:\\n{gr}\\n#### {ga}"
        w.write(json.dumps({"id":j.get("id"), "text":text, "question":q, "gold_answer":ga}, ensure_ascii=False)+"\\n")
print("wrote", dst)
PY

python -m src.sft_train_wrapper --train "${VAN}" --save_dir "${WORKDIR}/model_vanilla" --base_model "${BASE_MODEL}" --epochs ${SFT_EPOCHS} --lr ${SFT_LR} --seq_len ${SFT_SEQ_LEN} --per_device_batch ${SFT_BATCH} --grad_accum ${SFT_GRAD_ACC}
python -m src.evaluate --model "${WORKDIR}/model_vanilla" --test "${TEST_JSON}" --out "${WORKDIR}/vanilla_preds.jsonl" --max_new_tokens ${GEN_MAX_TOK} --temperature ${GEN_TEMP} --top_p ${GEN_TOPP}

# 3) STaR loop
echo "[run_all] STEP 3: STaR (K=${ITERATIONS})"
for k in $(seq 1 ${ITERATIONS}); do
  echo "[run_all] STaR iter ${k} forward gen"
  python -m src.star_generate --model "${BASE_MODEL}" --train "${TRAIN_USE}" --out "${WORKDIR}/star_iter${k}_forward.jsonl" --max_new_tokens ${GEN_MAX_TOK} --temperature ${GEN_TEMP} --top_p ${GEN_TOPP}
  echo "[run_all] STaR iter ${k} rationalize"
  python -m src.star_rationalize --model "${BASE_MODEL}" --train "${TRAIN_USE}" --missed "${WORKDIR}/star_iter${k}_forward.jsonl" --out "${WORKDIR}/star_iter${k}_rationalized.jsonl" --max_new_tokens ${GEN_MAX_TOK} --temperature ${GEN_TEMP} --top_p ${GEN_TOPP}
  echo "[run_all] STaR iter ${k} build"
  python -m src.star_build --forward "${WORKDIR}/star_iter${k}_forward.jsonl" --rationalized "${WORKDIR}/star_iter${k}_rationalized.jsonl" --out "${WORKDIR}/star_iter${k}_train.jsonl"
  echo "[run_all] STaR iter ${k} SFT from base"
  python -m src.sft_train_wrapper --train "${WORKDIR}/star_iter${k}_train.jsonl" --save_dir "${WORKDIR}/star_model_iter${k}" --base_model "${BASE_MODEL}" --epochs ${SFT_EPOCHS} --lr ${SFT_LR} --seq_len ${SFT_SEQ_LEN} --per_device_batch ${SFT_BATCH} --grad_accum ${SFT_GRAD_ACC}
  echo "[run_all] STaR iter ${k} eval"
  python -m src.evaluate --model "${WORKDIR}/star_model_iter${k}" --test "${TEST_JSON}" --out "${WORKDIR}/star_iter${k}_preds.jsonl" --max_new_tokens ${GEN_MAX_TOK} --temperature ${GEN_TEMP} --top_p ${GEN_TOPP}
done

echo "[run_all] DONE. Inspect ${WORKDIR} for outputs and models."
