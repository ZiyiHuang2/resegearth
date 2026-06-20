#!/usr/bin/env bash
set -euo pipefail

########################################
# TG-Swin-WTI v1.5: Warm-start Train → Merge → LaSeRS Test Eval
#
# 官方 LaSeRS 只有 train + test（无 val）：
#   - 训练：train_data.json
#   - 训练期 validation：train holdout 5%（非 test，避免泄漏）
#   - 最终评估：test 子集（eval.py / taxonomy）
#
# Override examples:
#   GPU_ID=0 MAX_STEPS=100 bash run_train_merge_test_tgswin.sh
#   GPU_ID=0 MAX_STEPS=1000 bash run_train_merge_test_tgswin.sh
#   RUN_TRAIN=0 RUN_EVAL=1 bash run_train_merge_test_tgswin.sh
#   MASK_CONFIG=segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml  # baseline ablation
#
# Short-run gates (see plans/tgswin_wti/experiment_gate.md):
#   MAX_STEPS=1   engineering smoke
#   MAX_STEPS=100 short convergence probe
#   MAX_STEPS=1000 mechanism / taxonomy gate
########################################

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT="${WANDB_PROJECT:-segearth-tgswin}"
export WANDB_NAME="${WANDB_NAME:-tgswin-wti-v15-lasers-warmstart-8w-gd4}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"

RESEG_ROOT="/root/rivermind-data/huangziyi/reseg"
CONDA_ENV_DIR="/root/rivermind-data/miniconda3/envs/reseg"
PYTHON="${CONDA_ENV_DIR}/bin/python"
DEEPSPEED="${CONDA_ENV_DIR}/bin/deepspeed"
export PATH="${CONDA_ENV_DIR}/bin:${PATH}"

GPU_ID="${GPU_ID:-0}"
GPU_SLOT="${GPU_SLOT:-localhost:${GPU_ID}}"
MASTER_PORT="${MASTER_PORT:-29641}"

REPO_DIR="${REPO_DIR:-${RESEG_ROOT}/segearth+tgswin}"
cd "${REPO_DIR}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-${RESEG_ROOT}/output/base/standard-base-lasers-siglip1-8w-gd4/merged_model}"
MODEL_PATH="${MODEL_PATH:-${BASE_MODEL_PATH}}"
VISION_TOWER="${RESEG_ROOT}/pretrained_model/CLIP/siglip-so400m-patch14-384"
VISION_TOWER_MASK="${RESEG_ROOT}/pretrained_model/mask2former/model_final_54b88a.pkl"
MASK_CONFIG="${MASK_CONFIG:-segearth_r2/model/mask_decoder/mask_config/maskformer2_tgswin_v15.yaml}"

BASE_DATA_PATH="${BASE_DATA_PATH:-/root/rivermind-data/huangziyi/data/LaSeRS}"
DATASET_NAME="lasers"
EVAL_SPLIT="test"
LASERS_BENCHMARK="${LASERS_BENCHMARK:-all}"
LASERS_HOLDOUT_RATIO="${LASERS_HOLDOUT_RATIO:-0.05}"

OUTPUT_DIR="${OUTPUT_DIR:-${RESEG_ROOT}/output/tgswin/tgswin-wti-v15-lasers-warmstart-8w-gd4}"
MERGED_DIR="${MERGED_DIR:-${OUTPUT_DIR}/merged_model}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_DIR}/test_results}"

EVAL_METRICS_SCRIPT="${RESEG_ROOT}/eval_val_metrics.py"
EVAL_USE_WANDB="True"
EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-segearth-eval-tgswin}"
EVAL_WANDB_RUN_NAME="${EVAL_WANDB_RUN_NAME:-tgswin-wti-v15-lasers-warmstart-8w-gd4}"

MAX_STEPS="${MAX_STEPS:-100000}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="0.0"
WARMUP_RATIO="0.03"
LR_SCHEDULER_TYPE="cosine"
LOGGING_STEPS="10"
BF16="True"
TF32="False"
MODEL_MAX_LENGTH="2048"
GRADIENT_CHECKPOINTING="False"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
LORA_R="8"
LORA_ALPHA="16"
LORA_DROPOUT="0.05"
DATA_RATIO="1"
SWITCH_BS="4"
SEED="42"
DATA_SEED="42"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_MERGE="${RUN_MERGE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_TAXONOMY="${RUN_TAXONOMY:-1}"
MERGE_CHECKPOINT="${MERGE_CHECKPOINT:-}"

TAXONOMY_SCRIPT="${RESEG_ROOT}/segearth+base/tools/diagnostics/audit_lasers_error_taxonomy.py"
TAXONOMY_ANN_FILE="${BASE_DATA_PATH}/test/annotations/test_multi_cate.json"
TAXONOMY_OUT_JSON="${RESEG_ROOT}/plans/baseline_diagnostics/tgswin_wti_multi_cate_error_taxonomy.json"
TAXONOMY_OUT_MD="${RESEG_ROOT}/plans/baseline_diagnostics/tgswin_wti_multi_cate_error_taxonomy.md"

read_best_checkpoint () {
  local out_dir="$1"
  OUTPUT_DIR_FOR_READ="${out_dir}" "${PYTHON}" - <<'PY'
import json, os, sys
output_dir = os.environ["OUTPUT_DIR_FOR_READ"]
trainer_state = os.path.join(output_dir, "trainer_state.json")
if not os.path.exists(trainer_state):
    print(""); sys.exit(0)
with open(trainer_state) as f:
    print(json.load(f).get("best_model_checkpoint", ""))
PY
}

read_last_checkpoint () {
  local out_dir="$1"
  OUTPUT_DIR_FOR_READ="${out_dir}" "${PYTHON}" - <<'PY'
import os, re
output_dir = os.environ["OUTPUT_DIR_FOR_READ"]
if not os.path.isdir(output_dir):
    print(""); raise SystemExit
candidates = []
for name in os.listdir(output_dir):
    m = re.match(r"checkpoint-(\d+)$", name)
    if m:
        candidates.append((int(m.group(1)), os.path.join(output_dir, name)))
print("" if not candidates else sorted(candidates)[-1][1])
PY
}

merge_ckpt () {
  local ckpt="$1"
  local save_dir="$2"
  rm -rf "${save_dir}"
  mkdir -p "${save_dir}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" segearth_r2/train/merge_lora_weights_and_save_hf_model.py \
    --model_path "${ckpt}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --mask_config "${MASK_CONFIG}" \
    --save_path "${save_dir}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}"
}

eval_model () {
  local model_dir="$1"
  local out_dir="$2"
  mkdir -p "${out_dir}"
  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${PYTHON}" segearth_r2/eval/eval.py \
    --base_data_path "${BASE_DATA_PATH}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --mask_config "${MASK_CONFIG}" \
    --model_path "${model_dir}" \
    --output_dir "${out_dir}" \
    --dataset_name "${DATASET_NAME}" \
    --split "${EVAL_SPLIT}" \
    --eval_batch_size 1 \
    --dataloader_num_workers 0 \
    --skip_existing True \
    --zip_results False
}

run_eval_metrics () {
  local pred_dir="$1"
  local run_name="$2"

  USE_WANDB="${EVAL_USE_WANDB}" \
  WANDB_PROJECT="${EVAL_WANDB_PROJECT}" \
  WANDB_RUN_NAME="${run_name}" \
  DATASET_TYPE="${DATASET_NAME}" \
  BASE_DATA_PATH="${BASE_DATA_PATH}" \
  SPLIT="${EVAL_SPLIT}" \
  LASERS_BENCHMARK="${LASERS_BENCHMARK}" \
  PRED_DIR="${pred_dir}" \
  "${PYTHON}" "${EVAL_METRICS_SCRIPT}"
}

run_taxonomy () {
  local pred_dir="$1"

  if [[ ! -f "${TAXONOMY_SCRIPT}" ]]; then
    echo "[ERROR] taxonomy script not found: ${TAXONOMY_SCRIPT}"
    exit 1
  fi
  if [[ ! -d "${pred_dir}" ]]; then
    echo "[ERROR] pred-dir not found for taxonomy: ${pred_dir}"
    exit 1
  fi
  if [[ ! -f "${TAXONOMY_ANN_FILE}" ]]; then
    echo "[ERROR] ann-file not found for taxonomy: ${TAXONOMY_ANN_FILE}"
    exit 1
  fi

  mkdir -p "$(dirname "${TAXONOMY_OUT_JSON}")"

  echo "[INFO] Running multi_cate error taxonomy..."
  "${PYTHON}" "${TAXONOMY_SCRIPT}" \
    --pred-dir "${pred_dir}" \
    --ann-file "${TAXONOMY_ANN_FILE}" \
    --split-name test_multi_cate \
    --out-json "${TAXONOMY_OUT_JSON}" \
    --out-md "${TAXONOMY_OUT_MD}"
  echo "[OK] taxonomy written: ${TAXONOMY_OUT_JSON}"
}

check_merged_tgswin_model () {
  MERGED_MODEL_DIR="${MERGED_DIR}" "${PYTHON}" - <<'PY'
import json, os, sys
model_dir = os.environ["MERGED_MODEL_DIR"]
index_path = os.path.join(model_dir, "model.safetensors.index.json")
if not os.path.isfile(index_path):
    print(f"[WARN] missing weight index: {index_path}")
    sys.exit(0)
with open(index_path) as f:
    keys = list(json.load(f).get("weight_map", {}).keys())
required = ("tg_swin_tcf", "tg_swin_controller")
missing = [r for r in required if not any(r in k for k in keys)]
if missing:
    print(f"[WARN] merged model may miss TG-Swin weights: {missing}")
else:
    for r in required:
        print(f"[OK] TG-Swin weight: {[k for k in keys if r in k][0]}")
PY
}

echo "========================================"
echo "[0/6] Preflight — TG-Swin-WTI v1.5"
echo "========================================"
echo "[INFO] REPO_DIR=${REPO_DIR}"
echo "[INFO] TG_SWIN config: ${MASK_CONFIG}"
echo "[INFO] BASE checkpoint: ${BASE_MODEL_PATH}"
echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] GPU_ID=${GPU_ID}"
echo "[INFO] MAX_STEPS=${MAX_STEPS}"
echo "[INFO] RUN_TRAIN=${RUN_TRAIN} RUN_MERGE=${RUN_MERGE} RUN_EVAL=${RUN_EVAL}"

MASK_CONFIG="${MASK_CONFIG}" "${PYTHON}" - <<'PY'
import os
from segearth_r2.datasets.dataset import get_mask_config
cfg = get_mask_config(os.environ["MASK_CONFIG"])
tg = getattr(cfg, "TG_SWIN", None)
if tg:
    print(f"[INFO] TG_SWIN VERSION={getattr(tg, 'VERSION', 'v1')}")
    print(f"[INFO] TG_SWIN HEAD_AWARE={getattr(tg, 'HEAD_AWARE', False)}")
    print(f"[INFO] TG_SWIN LOW_RANK_QK={getattr(tg, 'LOW_RANK_QK', False)}")
    print(f"[INFO] TG_SWIN WTI_RANK={getattr(tg, 'WTI_RANK', 16)}")
    print(f"[INFO] TG_SWIN WTI_STAGES={list(getattr(tg, 'WTI_STAGES', []))}")
    print(f"[INFO] TG_SWIN STAGE_ROUTER={getattr(tg, 'STAGE_ROUTER', False)}")
print("[INFO] TG_SWIN block:", dict(tg) if tg else "MISSING")
PY

echo "[INFO] Swin repeat multiplier estimate: ~mean(mask_num) per batch (per-target encoding)"
echo "[INFO] Running trainable param summary requires model init (see train.py logs)"

[[ -d "${MODEL_PATH}" ]] || { echo "[ERROR] model not found: ${MODEL_PATH}"; exit 1; }
[[ -d "${VISION_TOWER}" ]] || { echo "[ERROR] vision tower not found: ${VISION_TOWER}"; exit 1; }
[[ -f "${VISION_TOWER_MASK}" ]] || { echo "[ERROR] vision tower mask not found: ${VISION_TOWER_MASK}"; exit 1; }
[[ -f "${MASK_CONFIG}" ]] || { echo "[ERROR] mask config not found: ${MASK_CONFIG}"; exit 1; }

if [[ ! -f "${BASE_DATA_PATH}/train/annotations/train_data.json" ]]; then
  echo "[ERROR] train annotation not found: ${BASE_DATA_PATH}/train/annotations/train_data.json"
  echo "[HINT] 解压: tar -xzf ${BASE_DATA_PATH}/train.tar.gz -C ${BASE_DATA_PATH}"
  exit 1
fi

if [[ ! -d "${BASE_DATA_PATH}/train/images" ]]; then
  echo "[ERROR] train images not found: ${BASE_DATA_PATH}/train/images"
  exit 1
fi

if [[ ! -d "${BASE_DATA_PATH}/test/annotations" ]]; then
  echo "[ERROR] test annotations not found: ${BASE_DATA_PATH}/test/annotations"
  echo "[HINT] 解压: tar -xzf ${BASE_DATA_PATH}/test.tar.gz -C ${BASE_DATA_PATH}"
  exit 1
fi

if [[ ! -d "${BASE_DATA_PATH}/test/images" ]]; then
  echo "[ERROR] test images not found: ${BASE_DATA_PATH}/test/images"
  exit 1
fi

if [[ ! -f "${BASE_DATA_PATH}/test/annotations/test_multi_cate.json" ]]; then
  echo "[ERROR] taxonomy ann not found: ${BASE_DATA_PATH}/test/annotations/test_multi_cate.json"
  exit 1
fi

if [[ ! -f "scripts/zero1.json" ]]; then
  echo "[ERROR] DeepSpeed config not found: ${REPO_DIR}/scripts/zero1.json"
  exit 1
fi

echo "[OK] preflight passed (train + test present, no val required)"

mkdir -p "${OUTPUT_DIR}"

if [[ "${RUN_TRAIN}" == "1" ]]; then
echo "========================================"
echo "[1/6] Training TG-Swin-WTI on LaSeRS train (holdout eval)"
echo "========================================"
# train_data.json；eval 用同文件 holdout 5%（非 test）
"${DEEPSPEED}" --master_port="${MASTER_PORT}" --include="${GPU_SLOT}" segearth_r2/train/train.py \
  --model_name_or_path "${MODEL_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --base_data_path "${BASE_DATA_PATH}" \
  --dataset_name "${DATASET_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --bf16 "${BF16}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
  --logging_steps "${LOGGING_STEPS}" \
  --tf32 "${TF32}" \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --deepspeed scripts/zero1.json \
  --mask_config "${MASK_CONFIG}" \
  --data_ratio "${DATA_RATIO}" \
  --switch_bs "${SWITCH_BS}" \
  --seed "${SEED}" \
  --data_seed "${DATA_SEED}" \
  --lasers_holdout_ratio "${LASERS_HOLDOUT_RATIO}" \
  --lasers_holdout_seed "${DATA_SEED}" \
  --report_to wandb
else
echo "[1/6] SKIP training"
fi

BEST_CHECKPOINT="${MERGE_CHECKPOINT:-$(read_best_checkpoint "${OUTPUT_DIR}")}"
[[ -n "${BEST_CHECKPOINT}" ]] || BEST_CHECKPOINT="$(read_last_checkpoint "${OUTPUT_DIR}")"
[[ -n "${BEST_CHECKPOINT}" ]] || { echo "[ERROR] no checkpoint under ${OUTPUT_DIR}"; exit 1; }
echo "[2/6] SELECTED_CHECKPOINT=${BEST_CHECKPOINT}"

if [[ "${RUN_MERGE}" == "1" ]]; then
echo "[3/6] Merge"
merge_ckpt "${BEST_CHECKPOINT}" "${MERGED_DIR}"
else
echo "[3/6] SKIP merge"
fi

echo "[4/6] Check merged model"
check_merged_tgswin_model

if [[ "${RUN_EVAL}" == "1" ]]; then
echo "[5/7] Eval"
eval_model "${MERGED_DIR}" "${EVAL_OUTPUT_DIR}"
run_eval_metrics "${EVAL_OUTPUT_DIR}" "${EVAL_WANDB_RUN_NAME}"
if [[ "${RUN_TAXONOMY}" == "1" ]]; then
  echo "[6/7] Taxonomy (test_multi_cate)"
  run_taxonomy "${EVAL_OUTPUT_DIR}"
else
  echo "[6/7] SKIP taxonomy (RUN_TAXONOMY=${RUN_TAXONOMY})"
fi
else
echo "[5/7] SKIP eval"
echo "[6/7] SKIP taxonomy (RUN_EVAL=${RUN_EVAL})"
fi

echo "[7/7] DONE — output=${OUTPUT_DIR}"
