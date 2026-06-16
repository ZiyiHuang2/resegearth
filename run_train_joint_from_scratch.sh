#!/usr/bin/env bash
# =============================================================================
# SegEarth-R2 Joint decoder from scratch: Train → Merge → Parallel test eval
#
# Train dataset: LaSeRS (lasers)
# Test eval (parallel): LaSeRS / RRSISD / RefSegRS / RISBench / EarthReason
#
# Starts from Mipha-3B + mask2former init — NOT from base merged_model.
# Code path: segearth+joint (per-image joint [SEG] decoder ablation)
# =============================================================================
set -euo pipefail

########################################
# Environment
########################################
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_PROJECT="${WANDB_PROJECT:-segearth-joint}"
export WANDB_NAME="${WANDB_NAME:-joint-lasers-from-scratch-80k-gd4}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export TMPDIR="${TMPDIR:-/root/rivermind-data/tmp}"
mkdir -p "${TMPDIR}"
unset CUDA_VISIBLE_DEVICES

GPU_SLOT="${GPU_SLOT:-localhost:0}"
GPU_ID="${GPU_ID:-0}"
MASTER_PORT="${MASTER_PORT:-$((29850 + RANDOM % 1000))}"

PYTHON="${PYTHON:-/root/rivermind-data/miniconda3/envs/reseg/bin/python}"
DEEPSPEED="${DEEPSPEED:-/root/rivermind-data/miniconda3/envs/reseg/bin/deepspeed}"

########################################
# Project dir (joint branch)
########################################
REPO_DIR="/root/rivermind-data/huangziyi/reseg/segearth+joint"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
cd "${REPO_DIR}"

########################################
# Common paths — from scratch (NOT merged_model)
########################################
MODEL_NAME_OR_PATH="/root/rivermind-data/huangziyi/reseg/pretrained_model/mllm/Mipha-3B"
VISION_TOWER="/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384"
VISION_TOWER_MASK="/root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl"
MASK_CONFIG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"

########################################
# LaSeRS (train) + cross-dataset test paths
########################################
BASE_DATA_PATH="/root/rivermind-data/huangziyi/data/LaSeRS"
DATASET_NAME="lasers"
EVAL_SPLIT="test"
LASERS_BENCHMARK="${LASERS_BENCHMARK:-all}"
LASERS_HOLDOUT_RATIO="${LASERS_HOLDOUT_RATIO:-0.05}"

DATA_PATH_LASERS="/root/rivermind-data/huangziyi/data/LaSeRS"
DATA_PATH_RRSISD="/root/rivermind-data/huangziyi/data/RRSISD"
DATA_PATH_REFSEGRS="/root/rivermind-data/huangziyi/data/RefSegRS"
DATA_PATH_RISBENCH="/root/rivermind-data/huangziyi/data/RISBench_dataset"
DATA_PATH_EARTHREASON="/root/rivermind-data/huangziyi/data/EarthReason"

########################################
# Output (do not overwrite base)
########################################
OUTPUT_DIR="/root/rivermind-data/huangziyi/reseg/output/joint/full_joint_from_scratch"
MERGED_DIR="${OUTPUT_DIR}/merged_model"
LOG_FILE="${OUTPUT_DIR}/train.log"
PARALLEL_EVAL_LOG="${OUTPUT_DIR}/all_datasets_test_eval_parallel.log"

########################################
# Eval metrics
########################################
EVAL_METRICS_SCRIPT="/root/rivermind-data/huangziyi/reseg/eval_val_metrics.py"
EVAL_USE_WANDB="${EVAL_USE_WANDB:-True}"
EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-segearth-joint-eval}"
EVAL_WANDB_RUN_NAME="${EVAL_WANDB_RUN_NAME:-joint-lasers-from-scratch-80k-gd4}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"
EVAL_DATALOADER_NUM_WORKERS="${EVAL_DATALOADER_NUM_WORKERS:-2}"

########################################
# Train config (align with base 28w / 80k)
########################################
MAX_STEPS="${MAX_STEPS:-80000}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"

SAVE_STEPS="${SAVE_STEPS:-5000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

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

LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

DATA_RATIO="${DATA_RATIO:-1}"
SWITCH_BS="${SWITCH_BS:-4}"

SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-42}"

TRAIN_ONLY="${TRAIN_ONLY:-0}"

########################################
# Helpers
########################################
read_best_checkpoint () {
  local out_dir="$1"
  OUTPUT_DIR="${out_dir}" "${PYTHON}" - <<'PY'
import json, os, sys
output_dir = os.environ["OUTPUT_DIR"]
trainer_state = os.path.join(output_dir, "trainer_state.json")
if not os.path.exists(trainer_state):
    print(""); sys.exit(0)
state = json.load(open(trainer_state, encoding="utf-8"))
print(state.get("best_model_checkpoint", ""))
PY
}

read_last_checkpoint () {
  local out_dir="$1"
  OUTPUT_DIR="${out_dir}" "${PYTHON}" - <<'PY'
import os, re
output_dir = os.environ["OUTPUT_DIR"]
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

run_one_dataset_eval () {
  local name="$1"
  local data_path="$2"
  local model_dir="$3"
  local pred_dir="${OUTPUT_DIR}/${name}_test_results"
  local log_eval="${OUTPUT_DIR}/${name}_test_eval.log"
  local log_metrics="${OUTPUT_DIR}/${name}_test_metrics.log"

  mkdir -p "${pred_dir}"
  echo "[${name}] eval start $(date -Iseconds)"

  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${PYTHON}" segearth_r2/eval/eval.py \
    --base_data_path "${data_path}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --mask_config "${MASK_CONFIG}" \
    --model_path "${model_dir}" \
    --output_dir "${pred_dir}" \
    --dataset_name "${name}" \
    --split "${EVAL_SPLIT}" \
    --eval_batch_size 1 \
    --dataloader_num_workers "${EVAL_DATALOADER_NUM_WORKERS}" \
    --skip_existing True \
    --zip_results False \
    > "${log_eval}" 2>&1

  if [[ "${name}" == "lasers" ]]; then
    USE_WANDB="${EVAL_USE_WANDB}" \
    WANDB_PROJECT="${EVAL_WANDB_PROJECT}" \
    WANDB_RUN_NAME="${EVAL_WANDB_RUN_NAME}" \
    DATASET_TYPE="${name}" \
    BASE_DATA_PATH="${data_path}" \
    SPLIT="${EVAL_SPLIT}" \
    LASERS_BENCHMARK="${LASERS_BENCHMARK}" \
    PRED_DIR="${pred_dir}" \
    "${PYTHON}" "${EVAL_METRICS_SCRIPT}" > "${log_metrics}" 2>&1
  else
    USE_WANDB="${EVAL_USE_WANDB}" \
    WANDB_PROJECT="${EVAL_WANDB_PROJECT}" \
    WANDB_RUN_NAME="${EVAL_WANDB_RUN_NAME}" \
    DATASET_TYPE="${name}" \
    BASE_DATA_PATH="${data_path}" \
    SPLIT="${EVAL_SPLIT}" \
    PRED_DIR="${pred_dir}" \
    "${PYTHON}" "${EVAL_METRICS_SCRIPT}" > "${log_metrics}" 2>&1
  fi

  echo "[${name}] done $(date -Iseconds)"
}

launch_dataset_eval () {
  local name="$1"
  local data_path="$2"
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
    wait -n || true
  done
  run_one_dataset_eval "${name}" "${data_path}" "${MERGED_DIR}" &
  echo "[launcher] started ${name} pid=$! ($(date -Iseconds))"
}

run_all_datasets_parallel () {
  local model_dir="$1"
  [[ -f "${model_dir}/config.json" ]] || { echo "[ERROR] merged model not found: ${model_dir}/config.json"; exit 1; }

  mkdir -p "${OUTPUT_DIR}"
  {
    echo "========================================"
    echo "[PARALLEL-TEST] start $(date -Iseconds)"
    echo "OUT_DIR=${OUTPUT_DIR}"
    echo "MODEL=${model_dir}"
    echo "MAX_PARALLEL=${MAX_PARALLEL}"
    echo "========================================"
  } | tee "${PARALLEL_EVAL_LOG}"

  launch_dataset_eval lasers "${DATA_PATH_LASERS}"
  launch_dataset_eval rrsisd "${DATA_PATH_RRSISD}"
  launch_dataset_eval refsegrs "${DATA_PATH_REFSEGRS}"
  launch_dataset_eval risbench "${DATA_PATH_RISBENCH}"
  launch_dataset_eval earthreason "${DATA_PATH_EARTHREASON}"

  wait || true

  {
    echo "========================================"
    echo "[PARALLEL-TEST] finished $(date -Iseconds)"
    echo "LaSeRS summary     : ${OUTPUT_DIR}/lasers_${EVAL_SPLIT}_metrics_summary.json"
    echo "RRSISD metrics     : ${OUTPUT_DIR}/rrsisd_test_metrics.json"
    echo "RefSegRS metrics   : ${OUTPUT_DIR}/refsegrs_test_metrics.json"
    echo "RISBench metrics   : ${OUTPUT_DIR}/risbench_test_metrics.json"
    echo "EarthReason metrics: ${OUTPUT_DIR}/earthreason_test_metrics.json"
    echo "Master log         : ${PARALLEL_EVAL_LOG}"
    echo "========================================"
  } | tee -a "${PARALLEL_EVAL_LOG}"
}

########################################
# Preflight
########################################
echo "========================================"
echo "[0/6] Preflight (joint from scratch)"
echo "========================================"
echo "[INFO] REPO_DIR=${REPO_DIR}"
echo "[INFO] MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH} (NOT merged_model)"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] MAX_STEPS=${MAX_STEPS}, SAVE_STEPS=${SAVE_STEPS}"

if [[ "${MODEL_NAME_OR_PATH}" == *merged_model* ]]; then
  echo "[ERROR] joint from-scratch must not use merged_model as model_name_or_path"
  exit 1
fi

[[ -d "${MODEL_NAME_OR_PATH}" ]] || { echo "[ERROR] Mipha-3B not found: ${MODEL_NAME_OR_PATH}"; exit 1; }
[[ -d "${VISION_TOWER}" ]] || { echo "[ERROR] vision tower not found: ${VISION_TOWER}"; exit 1; }
[[ -f "${VISION_TOWER_MASK}" ]] || { echo "[ERROR] mask2former weights not found: ${VISION_TOWER_MASK}"; exit 1; }
[[ -f "${BASE_DATA_PATH}/train/annotations/train_data.json" ]] || { echo "[ERROR] LaSeRS train missing"; exit 1; }
[[ -d "${BASE_DATA_PATH}/test/annotations" ]] || { echo "[ERROR] LaSeRS test missing"; exit 1; }
[[ -f "scripts/zero1.json" ]] || { echo "[ERROR] zero1.json missing"; exit 1; }
[[ -f "${EVAL_METRICS_SCRIPT}" ]] || { echo "[ERROR] eval metrics script not found: ${EVAL_METRICS_SCRIPT}"; exit 1; }

for ds_path in "${DATA_PATH_LASERS}" "${DATA_PATH_RRSISD}" "${DATA_PATH_REFSEGRS}" "${DATA_PATH_RISBENCH}" "${DATA_PATH_EARTHREASON}"; do
  [[ -d "${ds_path}" ]] || { echo "[ERROR] dataset path missing: ${ds_path}"; exit 1; }
done

mkdir -p "${OUTPUT_DIR}"
{
  echo "=== joint from scratch $(date -Iseconds) ==="
  echo "model=${MODEL_NAME_OR_PATH}"
  echo "output=${OUTPUT_DIR}"
} | tee "${LOG_FILE}"

echo "[OK] preflight passed"

########################################
# 1) Train
########################################
echo "========================================"
echo "[1/6] Training joint on LaSeRS (from Mipha-3B)"
echo "========================================"

"${DEEPSPEED}" --master_port="${MASTER_PORT}" --include="${GPU_SLOT}" segearth_r2/train/train.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
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
  --report_to wandb 2>&1 | tee -a "${LOG_FILE}"

if [[ "${TRAIN_ONLY}" == "1" ]]; then
  echo "========================================"
  echo "[DONE] TRAIN_ONLY=1 — skipping merge / parallel eval"
  echo "Output dir: ${OUTPUT_DIR}"
  echo "========================================"
  exit 0
fi

########################################
# 2) Select checkpoint
########################################
echo "========================================"
echo "[2/6] Select checkpoint"
echo "========================================"
BEST_CHECKPOINT=$(read_best_checkpoint "${OUTPUT_DIR}")
if [[ -z "${BEST_CHECKPOINT}" ]]; then
  echo "[WARN] best_model_checkpoint not found; fallback to last"
  BEST_CHECKPOINT=$(read_last_checkpoint "${OUTPUT_DIR}")
fi
[[ -n "${BEST_CHECKPOINT}" ]] || { echo "[ERROR] no checkpoint"; exit 1; }
echo "[OK] SELECTED_CHECKPOINT=${BEST_CHECKPOINT}"

########################################
# 3) Merge
########################################
echo "========================================"
echo "[3/6] Merge LoRA checkpoint"
echo "========================================"
merge_ckpt "${BEST_CHECKPOINT}" "${MERGED_DIR}"

########################################
# 4) Check merged
########################################
echo "========================================"
echo "[4/6] Check merged model"
echo "========================================"
[[ -f "${MERGED_DIR}/config.json" ]] || { echo "[ERROR] merged config missing: ${MERGED_DIR}/config.json"; exit 1; }
echo "[OK] merged config.json present"

########################################
# 5) Parallel test eval on 5 datasets + metrics
########################################
echo "========================================"
echo "[5/6] Parallel test eval (5 datasets)"
echo "========================================"
run_all_datasets_parallel "${MERGED_DIR}"

########################################
# 6) Done
########################################
echo "========================================"
echo "[6/6] DONE"
echo "Output dir           : ${OUTPUT_DIR}"
echo "Selected ckpt        : ${BEST_CHECKPOINT}"
echo "Merged model         : ${MERGED_DIR}"
echo "LaSeRS summary       : ${OUTPUT_DIR}/lasers_${EVAL_SPLIT}_metrics_summary.json"
echo "Cross-dataset metrics: ${OUTPUT_DIR}/*_test_metrics.json"
echo "Parallel eval log    : ${PARALLEL_EVAL_LOG}"
echo "Train log            : ${LOG_FILE}"
echo "Compare baseline     : output/base/standard-base-lasers-siglip1-28w-gd4"
echo "========================================"
