#!/usr/bin/env bash
set -euo pipefail

########################################
# TCPD-only: train -> merge -> LaSeRS test eval
#
# Experiment cell (LaSeRS 2x2 ablation):
#   Language: current [SEG]  |  Pixel Decoder: SEG-conditioned TCPD
#
# Training:
#   --use_tcpd True --tcpd_train True --test4_mode True --no_lora_enable
#   freeze base pixel_decoder; train only *tcpd* inside PD
#   unfreeze LLM last 2 layers + SEG_token_projector + predictor
#
# Override examples:
#   MAX_STEPS=5000 GPU_SLOT=localhost:0 bash run_train_merge_test.sh
#   OUTPUT_DIR=/path/to/run bash run_train_merge_test.sh
#
# Base model (fixed on this machine — only 8w LaSeRS merged_model exists):
#   ${RESEG_ROOT}/output/base/standard-base-lasers-siglip1-8w-gd4/merged_model
#   Override: MODEL_PATH=/other/merged_model bash run_train_merge_test.sh
########################################

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/test4_common.sh
source "${SCRIPT_DIR}/scripts/test4_common.sh"

REPO_DIR="${REPO_DIR}"
cd "${REPO_DIR}"
export PATH="${CONDA_BIN}:${PATH}"

# Default base: 8w LaSeRS merged_model (only available base on this machine)
DEFAULT_BASE_MODEL="${RESEG_ROOT}/output/base/standard-base-lasers-siglip1-8w-gd4/merged_model"
MODEL_PATH="${MODEL_PATH:-${DEFAULT_BASE_MODEL}}"

########################################
# Runtime
########################################
GPU_SLOT="${GPU_SLOT:-localhost:0}"
MASTER_PORT="${MASTER_PORT:-29530}"

########################################
# Experiment naming / output
########################################
RUN_TAG="${RUN_TAG:-tcpd-only-seg-$(date +%Y%m%d_%H%M%S)}"

########################################
# W&B (train + eval)
########################################
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
WANDB_PROJECT="${WANDB_PROJECT:-segearth-tcpd}"
WANDB_NAME="${WANDB_NAME:-${RUN_TAG}}"
export WANDB_PROJECT
TRAIN_REPORT_TO="${TRAIN_REPORT_TO:-wandb}"

EVAL_USE_WANDB="${EVAL_USE_WANDB:-True}"
EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-segearth-eval-tcpd}"
EVAL_WANDB_RUN_NAME="${EVAL_WANDB_RUN_NAME:-${RUN_TAG}-eval}"

MAX_STEPS="${MAX_STEPS:-50000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-${TEST4_OUTPUT_ROOT}/${RUN_TAG}}"
MERGED_DIR="${MERGED_DIR:-${OUTPUT_DIR}/merged_model}"
TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR:-${OUTPUT_DIR}/lasers_test_results}"
BASELINE_FILE="${BASELINE_FILE:-${OUTPUT_DIR}/baseline_multi_cate_giou.txt}"

########################################
# Train hyperparams (Test4 / TCPD preset)
########################################
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LLM_LR="${LLM_LR:-1e-5}"
MASK_LR="${MASK_LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATA_RATIO="${DATA_RATIO:-1}"
SWITCH_BS="${SWITCH_BS:-4}"
SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-42}"

########################################
# Helpers
########################################
read_last_checkpoint() {
  local out_dir="$1"
  TRAIN_OUTPUT_DIR="${out_dir}" "${PYTHON}" - <<'PY'
import os
import re
import sys

output_dir = os.environ["TRAIN_OUTPUT_DIR"]
if not os.path.isdir(output_dir):
    print("")
    sys.exit(1)

candidates = []
for name in os.listdir(output_dir):
    m = re.match(r"checkpoint-(\d+)$", name)
    if m:
        candidates.append((int(m.group(1)), os.path.join(output_dir, name)))

if not candidates:
    print("")
    sys.exit(1)

candidates.sort()
print(candidates[-1][1])
PY
}

verify_merged_tcpd_config() {
  local merged_dir="$1"
  MERGED_DIR="${merged_dir}" "${PYTHON}" - <<'PY'
import json
import os
import sys

path = os.path.join(os.environ["MERGED_DIR"], "config.json")
if not os.path.isfile(path):
    print(f"[ERROR] missing {path}", file=sys.stderr)
    sys.exit(1)

with open(path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

use_tcpd = cfg.get("use_tcpd", False)
source = cfg.get("tcpd_condition_source", "seg")
print(f"[OK] merged config: use_tcpd={use_tcpd}, tcpd_condition_source={source}")

if not use_tcpd:
    print("[ERROR] merged config use_tcpd is not true", file=sys.stderr)
    sys.exit(1)
if source != "seg":
    print(f"[ERROR] unexpected tcpd_condition_source={source}", file=sys.stderr)
    sys.exit(1)
PY
}

########################################
# Preflight
########################################
echo "========================================"
echo "[0/6] Preflight (TCPD-only)"
echo "========================================"

test4_preflight
test4_resolve_base_model

echo "[INFO] REPO_DIR=${REPO_DIR}"
echo "[INFO] MODEL_PATH=${MODEL_PATH} (${MODEL_BASE_SOURCE})"
echo "[INFO] DATA_PATH=${DATA_PATH}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] GPU_SLOT=${GPU_SLOT}"
echo "[INFO] use_tcpd=True tcpd_train=True test4_mode=True"
echo "[INFO] MAX_STEPS=${MAX_STEPS} SAVE_STEPS=${SAVE_STEPS}"
echo "[INFO] LLM_LR=${LLM_LR} MASK_LR=${MASK_LR}"
echo "[INFO] WANDB_MODE=${WANDB_MODE}"
echo "[INFO] WANDB_PROJECT=${WANDB_PROJECT} WANDB_NAME=${WANDB_NAME}"
echo "[INFO] TRAIN_REPORT_TO=${TRAIN_REPORT_TO}"
echo "[INFO] EVAL_USE_WANDB=${EVAL_USE_WANDB} EVAL_WANDB_PROJECT=${EVAL_WANDB_PROJECT}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ERROR] base model not found: ${MODEL_PATH}"
  exit 1
fi

if [[ ! -d "${DATA_PATH}/test/annotations" && ! -d "${DATA_PATH}/val/annotations" ]]; then
  echo "[ERROR] LaSeRS annotations not found under ${DATA_PATH}/test or .../val"
  exit 1
fi

if [[ ! -f "scripts/zero1.json" ]]; then
  echo "[ERROR] DeepSpeed config not found: ${REPO_DIR}/scripts/zero1.json"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# Baseline gIoU from base model metrics (for gate reporting only)
if [[ ! -f "${BASELINE_FILE}" ]]; then
  BASELINE_METRICS_JSON="$(test4_default_base_metrics_json)"
  test4_seed_baseline_from_metrics "${BASELINE_METRICS_JSON}" "${BASELINE_FILE}"
fi
BASELINE_GIOU="$(test4_read_baseline_giou "${BASELINE_FILE}")"
echo "[INFO] baseline test_multi_cate gIoU = ${BASELINE_GIOU}%"

echo "[OK] preflight passed"

########################################
# 1) Train (TCPD-only)
########################################
echo "========================================"
echo "[1/6] Training TCPD-only (LaSeRS)"
echo "========================================"

RESUME_FLAG=()
if compgen -G "${OUTPUT_DIR}/checkpoint-*" > /dev/null; then
  echo "[INFO] existing checkpoints in ${OUTPUT_DIR}; trainer will resume"
fi

"${DEEPSPEED}" --master_port="${MASTER_PORT}" --include "${GPU_SLOT}" \
  segearth_r2/train/train.py \
  --model_name_or_path "${MODEL_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --base_data_path "${DATA_PATH}" \
  --dataset_name lasers \
  --output_dir "${OUTPUT_DIR}" \
  --use_tcpd True \
  --tcpd_train True \
  --test4_mode True \
  --no_lora_enable \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --llm_lr "${LLM_LR}" \
  --mask_lr "${MASK_LR}" \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --bf16 True \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
  --logging_steps "${LOGGING_STEPS}" \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --deepspeed scripts/zero1.json \
  --mask_config "${MASK_CONFIG}" \
  --data_ratio "${DATA_RATIO}" \
  --switch_bs "${SWITCH_BS}" \
  --seed "${SEED}" \
  --data_seed "${DATA_SEED}" \
  --run_name "${WANDB_NAME}" \
  --report_to "${TRAIN_REPORT_TO}"

########################################
# 2) Select checkpoint (last; no train-time val in test4_mode)
########################################
echo "========================================"
echo "[2/6] Select checkpoint"
echo "========================================"

SELECTED_CHECKPOINT="$(read_last_checkpoint "${OUTPUT_DIR}")"
if [[ -z "${SELECTED_CHECKPOINT}" ]]; then
  echo "[ERROR] no checkpoint found under ${OUTPUT_DIR}"
  exit 1
fi
echo "[OK] SELECTED_CHECKPOINT=${SELECTED_CHECKPOINT}"

########################################
# 3) Merge (preserve use_tcpd=true)
#    LORA_ENABLE=False  — test4_mode has no LoRA
#    USE_TCPD=True      — write use_tcpd=true into merged config.json
#    SPOT_CHECK_BASE    — 8w base merged_model for --init_model_path
#    Dry-run merge only:
#      CHECKPOINT_PATH=.../checkpoint-N bash scripts/tcpd_merge_verify.sh
########################################
echo "========================================"
echo "[3/6] Merge checkpoint (USE_TCPD=True)"
echo "========================================"

rm -rf "${MERGED_DIR}"
CHECKPOINT_PATH="${SELECTED_CHECKPOINT}" \
MERGED_DIR="${MERGED_DIR}" \
LORA_ENABLE=False \
USE_TCPD=True \
TCPD_CONDITION_SOURCE=seg \
SPOT_CHECK_BASE="${MODEL_PATH}" \
bash "${SCRIPT_DIR}/scripts/test4_merge_checkpoint.sh"

########################################
# 4) Verify merged TCPD config
########################################
echo "========================================"
echo "[4/6] Verify merged TCPD config"
echo "========================================"

verify_merged_tcpd_config "${MERGED_DIR}"

########################################
# 5) Eval merged model on LaSeRS test
########################################
echo "========================================"
echo "[5/6] Eval merged model + metrics"
echo "========================================"

USE_WANDB="${EVAL_USE_WANDB}" \
WANDB_PROJECT="${EVAL_WANDB_PROJECT}" \
WANDB_RUN_NAME="${EVAL_WANDB_RUN_NAME}" \
MODEL_PATH="${MERGED_DIR}" \
EVAL_OUTPUT_DIR="${TEST_OUTPUT_DIR}" \
BASELINE_GIOU="${BASELINE_GIOU}" \
BASELINE_FILE="${BASELINE_FILE}" \
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}" \
SUMMARY_FILE="${OUTPUT_DIR}/tcpd_eval_summary.txt" \
bash "${SCRIPT_DIR}/scripts/test4_eval_model.sh"

TABLE2_JSON="$(test4_find_table2_json "${TEST_OUTPUT_DIR}" "${OUTPUT_DIR}")"
RESULT_GIOU="$(test4_parse_multi_cate_giou "${TABLE2_JSON}")"
test4_print_gate "${BASELINE_GIOU}" "${RESULT_GIOU}"

########################################
# 6) Done
########################################
echo "========================================"
echo "[6/6] DONE (TCPD-only)"
echo "Experiment        : TCPD-only ([SEG] -> TCPD, predictor unchanged)"
echo "Output dir        : ${OUTPUT_DIR}"
echo "Selected ckpt     : ${SELECTED_CHECKPOINT}"
echo "Merged model      : ${MERGED_DIR}"
echo "Test outputs      : ${TEST_OUTPUT_DIR}"
echo "Metrics json      : ${TABLE2_JSON}"
echo "Baseline gIoU     : ${BASELINE_GIOU}%"
echo "Result gIoU       : ${RESULT_GIOU}%"
echo "Train config      : ${OUTPUT_DIR}/test4_train_config.txt"
echo "Eval summary      : ${OUTPUT_DIR}/tcpd_eval_summary.txt"
echo "W&B train         : project=${WANDB_PROJECT} run=${WANDB_NAME}"
echo "W&B eval          : project=${EVAL_WANDB_PROJECT} run=${EVAL_WANDB_RUN_NAME}"
echo "========================================"
