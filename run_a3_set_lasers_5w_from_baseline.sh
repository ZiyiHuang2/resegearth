#!/usr/bin/env bash
# =============================================================================
# SegEarth-R2 A3 [SET] frozen training flow on LaSeRS:
#   train A3 modules -> merge A3 checkpoint -> eval -> metrics -> diagnostic
#
# Purpose:
#   Validate SET conditioning on top of the trained LaSeRS baseline merged model.
#   This is not a new base training run and must not start from raw Mipha.
#
# Key constraints:
#   - Use the same SigLIP vision tower as the baseline.
#   - Start from the LaSeRS baseline merged_model.
#   - Freeze the original model; train only set_conditioner/count/category heads.
#   - Do not change tokenizer, decoder, predictor, or answer template.
# =============================================================================
set -euo pipefail

########################################
# Environment
########################################
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_PROJECT="${WANDB_PROJECT:-segearth-set-lasers}"
export WANDB_NAME="${WANDB_NAME:-a3-frozen-lasers-set-5w}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
unset CUDA_VISIBLE_DEVICES

GPU_SLOT="${GPU_SLOT:-localhost:0}"
GPU_ID="${GPU_ID:-0}"
MASTER_PORT="${MASTER_PORT:-$((29600 + RANDOM % 1000))}"

PYTHON="${PYTHON:-/root/rivermind-data/miniconda3/envs/reseg/bin/python}"
DEEPSPEED="${DEEPSPEED:-/root/rivermind-data/miniconda3/envs/reseg/bin/deepspeed}"

########################################
# Project dir
########################################
REPO_DIR="/root/rivermind-data/huangziyi/reseg/segearth+set"
cd "${REPO_DIR}"

########################################
# Common paths
########################################
BASELINE_MERGED="/root/rivermind-data/huangziyi/reseg/output/base/standard-base-lasers-siglip1-28w-gd4/merged_model"
VISION_TOWER="/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384"
VISION_TOWER_MASK="/root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl"
MASK_CONFIG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
VOCAB_PATH="segearth_r2/model/lasers_category_vocab.json"

########################################
# LaSeRS config
########################################
BASE_DATA_PATH="/root/rivermind-data/huangziyi/data/LaSeRS"
DATASET_NAME="lasers"
EVAL_SPLIT="test"
LASERS_BENCHMARK="${LASERS_BENCHMARK:-all}"
DIAG_LASERS_BENCHMARK="${DIAG_LASERS_BENCHMARK:-test_multi_cate}"
LASERS_HOLDOUT_RATIO="${LASERS_HOLDOUT_RATIO:-0.05}"
LASERS_HOLDOUT_SEED="${LASERS_HOLDOUT_SEED:-42}"

########################################
# Output
########################################
OUTPUT_DIR="/root/rivermind-data/huangziyi/reseg/output/set/a3-frozen-lasers-set-5w"
MERGED_DIR="${OUTPUT_DIR}/merged_model"
EVAL_OUTPUT_DIR="${OUTPUT_DIR}/test_results"
DIAG_OUTPUT_DIR="${OUTPUT_DIR}/lasers_diagnostic"
LOG_FILE="${OUTPUT_DIR}/train.log"

########################################
# Eval metrics
########################################
EVAL_METRICS_SCRIPT="/root/rivermind-data/huangziyi/reseg/eval_val_metrics.py"
EVAL_USE_WANDB="${EVAL_USE_WANDB:-True}"
EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-segearth-set-eval}"
EVAL_WANDB_RUN_NAME="${EVAL_WANDB_RUN_NAME:-a3-frozen-lasers-set-5w}"

########################################
# Train config
########################################
MAX_STEPS="${MAX_STEPS:-50000}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"

# Checkpoint every 2k steps (resume-friendly). save_total_limit=3 keeps ~6k steps on disk.
SAVE_STEPS="${SAVE_STEPS:-2000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
# Set FRESH_START=1 to kill stale SET jobs and train from step 0 (no checkpoint resume).
FRESH_START="${FRESH_START:-0}"

LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"

LOGGING_STEPS="${LOGGING_STEPS:-10}"
BF16="${BF16:-True}"
TF32="${TF32:-False}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-False}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"

DATA_RATIO="${DATA_RATIO:-1}"
SWITCH_BS="${SWITCH_BS:-4}"

SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-42}"

LAMBDA_SET_COUNT="${LAMBDA_SET_COUNT:-0.05}"
LAMBDA_SET_CATEGORY="${LAMBDA_SET_CATEGORY:-0.1}"
SET_MAX_COUNT="${SET_MAX_COUNT:-10}"
SET_CONDITIONER_LAYERS="${SET_CONDITIONER_LAYERS:-1}"
SET_CONDITIONER_HEADS="${SET_CONDITIONER_HEADS:-4}"
SET_CONDITIONER_GATE_INIT="${SET_CONDITIONER_GATE_INIT:-0.0}"

########################################
# Helpers
########################################
stop_set_training_jobs () {
  local pids
  pids=$(pgrep -f -- "--output_dir ${OUTPUT_DIR}" 2>/dev/null || true)
  if [[ -z "${pids}" ]]; then
    return 0
  fi
  echo "[INFO] stopping existing SET jobs for ${OUTPUT_DIR}: ${pids}"
  kill ${pids} 2>/dev/null || true
  sleep 3
  pids=$(pgrep -f -- "--output_dir ${OUTPUT_DIR}" 2>/dev/null || true)
  if [[ -n "${pids}" ]]; then
    echo "[WARN] force killing SET jobs: ${pids}"
    kill -9 ${pids} 2>/dev/null || true
    sleep 1
  fi
}

read_last_checkpoint () {
  local out_dir="$1"
  OUTPUT_DIR="${out_dir}" "${PYTHON}" - <<'PY'
import os
import re

output_dir = os.environ["OUTPUT_DIR"]
if not os.path.isdir(output_dir):
    print("")
    raise SystemExit

candidates = []
for name in os.listdir(output_dir):
    m = re.match(r"checkpoint-(\d+)$", name)
    if m:
        candidates.append((int(m.group(1)), os.path.join(output_dir, name)))

if not candidates:
    print("")
else:
    candidates.sort()
    print(candidates[-1][1])
PY
}

merge_ckpt () {
  local ckpt="$1"
  local save_dir="$2"

  rm -rf "${save_dir}"
  mkdir -p "${save_dir}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" segearth_r2/train/merge_lora_weights_and_save_hf_model.py \
    --model_path "${ckpt}" \
    --baseline_model_path "${BASELINE_MERGED}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --mask_config "${MASK_CONFIG}" \
    --save_path "${save_dir}" \
    --a3_only \
    --no-lora \
    --lasers_category_vocab_path "${VOCAB_PATH}"
}

eval_model () {
  local model_dir="$1"
  local out_dir="$2"

  rm -rf "${out_dir}"
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
    --zip_results False
}

run_eval_metrics () {
  local pred_dir="$1"
  local run_name="$2"

  if [[ ! -f "${EVAL_METRICS_SCRIPT}" ]]; then
    echo "[ERROR] eval metrics script not found: ${EVAL_METRICS_SCRIPT}"
    exit 1
  fi

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

run_diagnostic () {
  local model_dir="$1"
  local out_dir="$2"

  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"

  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${PYTHON}" segearth_r2/eval/eval_lasers_diagnostic.py \
    --base_data_path "${BASE_DATA_PATH}" \
    --model_path "${model_dir}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --mask_config "${MASK_CONFIG}" \
    --output_dir "${out_dir}" \
    --lasers_benchmark "${DIAG_LASERS_BENCHMARK}" \
    --no-resume
}

########################################
# 0) Preflight
########################################
echo "========================================"
echo "[0/7] Preflight checks for A3-frozen SET"
echo "========================================"

echo "[INFO] REPO_DIR=${REPO_DIR}"
echo "[INFO] BASELINE_MERGED=${BASELINE_MERGED}"
echo "[INFO] BASE_DATA_PATH=${BASE_DATA_PATH}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] MAX_STEPS=${MAX_STEPS}, SAVE_STEPS=${SAVE_STEPS}, SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}"

if [[ ! -d "${BASELINE_MERGED}" ]]; then
  echo "[ERROR] baseline merged model not found: ${BASELINE_MERGED}"
  exit 1
fi

if [[ ! -f "${BASE_DATA_PATH}/train/annotations/train_data.json" ]]; then
  echo "[ERROR] train annotation not found: ${BASE_DATA_PATH}/train/annotations/train_data.json"
  exit 1
fi

if [[ ! -d "${BASE_DATA_PATH}/test/annotations" ]]; then
  echo "[ERROR] test annotations not found: ${BASE_DATA_PATH}/test/annotations"
  exit 1
fi

if [[ ! -f "${VOCAB_PATH}" ]]; then
  echo "[ERROR] category vocab not found: ${VOCAB_PATH}"
  exit 1
fi

if [[ ! -f "scripts/zero1.json" ]]; then
  echo "[ERROR] DeepSpeed config not found: ${REPO_DIR}/scripts/zero1.json"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

if [[ "${FRESH_START}" == "1" ]]; then
  echo "[INFO] FRESH_START=1: stop stale jobs, remove checkpoints, rotate train.log"
  stop_set_training_jobs
  rm -rf "${OUTPUT_DIR}"/checkpoint-* "${OUTPUT_DIR}"/trainer_state.json
  if [[ -f "${LOG_FILE}" ]]; then
    mv "${LOG_FILE}" "${LOG_FILE}.bak.$(date +%s)"
  fi
fi

echo "[INFO] Checking A3-frozen trainable parameter scope"
"${PYTHON}" scripts/check_a3_trainable_params.py \
  --model_name_or_path "${BASELINE_MERGED}" \
  --use_set_conditioner \
  --a3_train_only_set_modules

# Only block duplicate SET on the same output_dir (TGI/other projects may share the GPU).
if pgrep -f -- "--output_dir ${OUTPUT_DIR}" > /dev/null; then
  echo "[ERROR] SET training already running for output_dir=${OUTPUT_DIR}"
  pgrep -af -- "--output_dir ${OUTPUT_DIR}"
  exit 1
fi
if [[ "${SKIP_GPU_BUSY_CHECK:-0}" != "1" ]] && pgrep -f "segearth_r2/train/train.py" > /dev/null; then
  echo "[WARN] another train.py is running (e.g. TGI). Continuing SET on the same GPU."
  echo "[WARN] Ensure VRAM fits both jobs (~12GB + ~15-25GB on A100 80GB is usually OK)."
  pgrep -af "segearth_r2/train/train.py" || true
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true
fi

echo "[OK] preflight passed"

########################################
# 1) Train A3-frozen
########################################
echo "========================================"
echo "[1/7] Train A3-frozen SET on LaSeRS"
echo "========================================"

{
  echo "=== A3-frozen SET 5w $(date -Iseconds) ==="
  echo "baseline=${BASELINE_MERGED}"
  echo "output=${OUTPUT_DIR}"
} | tee "${LOG_FILE}"

"${DEEPSPEED}" --master_port="${MASTER_PORT}" --include="${GPU_SLOT}" segearth_r2/train/train.py \
  --model_name_or_path "${BASELINE_MERGED}" \
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
  --evaluation_strategy no \
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
  --lora_enable False \
  --deepspeed scripts/zero1.json \
  --mask_config "${MASK_CONFIG}" \
  --data_ratio "${DATA_RATIO}" \
  --switch_bs "${SWITCH_BS}" \
  --seed "${SEED}" \
  --data_seed "${DATA_SEED}" \
  --lasers_holdout_ratio "${LASERS_HOLDOUT_RATIO}" \
  --lasers_holdout_seed "${LASERS_HOLDOUT_SEED}" \
  --lasers_category_vocab_path "${VOCAB_PATH}" \
  --use_set_conditioner True \
  --use_set_count_loss True \
  --use_set_category_loss True \
  --lambda_set_count "${LAMBDA_SET_COUNT}" \
  --lambda_set_category "${LAMBDA_SET_CATEGORY}" \
  --set_max_count "${SET_MAX_COUNT}" \
  --set_conditioner_layers "${SET_CONDITIONER_LAYERS}" \
  --set_conditioner_heads "${SET_CONDITIONER_HEADS}" \
  --set_conditioner_gate_init "${SET_CONDITIONER_GATE_INIT}" \
  --a3_train_only_set_modules True \
  --report_to wandb 2>&1 | tee -a "${LOG_FILE}"

grep -q "Mask Decoder has been trained, init directly" "${LOG_FILE}" || {
  echo "[WARN] baseline mask decoder marker not found in log"
}

if grep -q "Initialize mask modules" "${LOG_FILE}"; then
  echo "[FATAL] mask modules were reinitialized. This is not a valid A3-from-baseline run."
  exit 1
fi

if [[ "${TRAIN_ONLY:-0}" == "1" ]]; then
  echo "========================================"
  echo "[DONE] TRAIN_ONLY=1 — skipping merge / eval / diagnostic"
  echo "Output dir: ${OUTPUT_DIR}"
  echo "========================================"
  exit 0
fi

########################################
# 2) Select last checkpoint
########################################
echo "========================================"
echo "[2/7] Select last checkpoint"
echo "========================================"

SELECTED_CHECKPOINT=$(read_last_checkpoint "${OUTPUT_DIR}")

if [[ -z "${SELECTED_CHECKPOINT}" ]]; then
  echo "[ERROR] no checkpoint found under ${OUTPUT_DIR}"
  exit 1
fi

echo "[OK] SELECTED_CHECKPOINT=${SELECTED_CHECKPOINT}"

########################################
# 3) Merge A3 checkpoint
########################################
echo "========================================"
echo "[3/7] Merge A3 checkpoint"
echo "========================================"

merge_ckpt "${SELECTED_CHECKPOINT}" "${MERGED_DIR}"

########################################
# 4) Check merged model
########################################
echo "========================================"
echo "[4/7] Check merged model"
echo "========================================"

if [[ ! -f "${MERGED_DIR}/config.json" ]]; then
  echo "[ERROR] merged config.json not found: ${MERGED_DIR}/config.json"
  exit 1
fi

PYTHONPATH="${REPO_DIR}" "${PYTHON}" - <<PY
from segearth_r2.datasets.dataset import get_mask_config
from segearth_r2.model.language_model.llava_phi import SegEarthR2

mask_cfg = get_mask_config("${MASK_CONFIG}")
model = SegEarthR2.from_pretrained("${MERGED_DIR}", mask_decoder_cfg=mask_cfg, add_cross_attn=True, device_map="cpu")
model.init_set_conditioning_modules(model.config)
shape = tuple(model.category_set_head.head.weight.shape)
assert shape == (191, 256), f"unexpected category_set_head shape: {shape}"
assert getattr(model.config, "use_set_conditioner", False), "use_set_conditioner missing from merged config"
print("[OK] merged model loads; category_set_head shape =", shape)
PY

########################################
# 5) Eval on LaSeRS test benchmarks + metrics
########################################
echo "========================================"
echo "[5/7] Eval on LaSeRS test benchmarks + metrics"
echo "========================================"

eval_model "${MERGED_DIR}" "${EVAL_OUTPUT_DIR}"
run_eval_metrics "${EVAL_OUTPUT_DIR}" "${EVAL_WANDB_RUN_NAME}"

########################################
# 6) Diagnostic
########################################
echo "========================================"
echo "[6/7] LaSeRS diagnostic"
echo "========================================"

run_diagnostic "${MERGED_DIR}" "${DIAG_OUTPUT_DIR}"

########################################
# 7) Done
########################################
echo "========================================"
echo "[7/7] DONE"
echo "Dataset             : LaSeRS"
echo "Baseline merged     : ${BASELINE_MERGED}"
echo "Output dir          : ${OUTPUT_DIR}"
echo "Selected checkpoint : ${SELECTED_CHECKPOINT}"
echo "Merged model        : ${MERGED_DIR}"
echo "Eval predictions    : ${EVAL_OUTPUT_DIR}"
echo "Diagnostic output   : ${DIAG_OUTPUT_DIR}"
echo "Diagnostic benchmark: ${DIAG_LASERS_BENCHMARK}"
echo "Metrics summary     : ${OUTPUT_DIR}/lasers_${EVAL_SPLIT}_metrics_summary.json"
