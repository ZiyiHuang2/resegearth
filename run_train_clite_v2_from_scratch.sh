#!/usr/bin/env bash
# =============================================================================
# SegEarth-R2 C-lite-v2 training from scratch (Mipha-3B + SigLIP2):
#   Prefix [SET] token + implicit SEG pooling fusion
#   Train (LoRA + SEG path + SET modules + SET_token_projector + q_set_fusion)
#   → Merge → Serial test eval → Diagnostic
#
# Purpose:
#   Validate C-lite-v2 approach: explicit [SET] token as "query-conditioned set
#   planning signal" fused with implicit SEG pooling "execution-stage set evidence".
#
# Key difference from run_train_set_joint_from_scratch.sh:
#   - Adds --use_explicit_set_token True
#   - Dataset auto-inserts [SET] at answer prefix
#   - SET_token_projector extracts q_set_explicit from [SET] hidden state
#   - q_set_fusion = MLP([q_explicit; q_implicit]) -> delta; q_fused = q_implicit + delta
#   - q_set_fusion: layer0 xavier + layer2 zero-init (step0 delta=0, layer2 learns first)
#
# Code path: segearth+set (SetConditioner with q_set_override from fusion)
# =============================================================================
set -euo pipefail

########################################
# Environment
########################################
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_PROJECT="${WANDB_PROJECT:-segearth-set-lasers}"
export WANDB_NAME="${WANDB_NAME:-clite-v2-from-scratch-50k}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export TMPDIR="${TMPDIR:-/root/rivermind-data/tmp}"
mkdir -p "${TMPDIR}"
unset CUDA_VISIBLE_DEVICES

GPU_SLOT="${GPU_SLOT:-localhost:0}"
GPU_ID="${GPU_ID:-0}"
MASTER_PORT="${MASTER_PORT:-$((29750 + RANDOM % 1000))}"

PYTHON="${PYTHON:-/root/rivermind-data/miniconda3/envs/reseg/bin/python}"
DEEPSPEED="${DEEPSPEED:-/root/rivermind-data/miniconda3/envs/reseg/bin/deepspeed}"

########################################
# Project dir (SET branch)
########################################
REPO_DIR="/root/rivermind-data/huangziyi/reseg/segearth+set"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
cd "${REPO_DIR}"

########################################
# Init paths — from scratch (NOT merged_model)
########################################
MODEL_NAME_OR_PATH="/root/rivermind-data/huangziyi/reseg/pretrained_model/mllm/Mipha-3B"
VISION_TOWER="/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384"
VISION_TOWER_MASK="/root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl"
MASK_CONFIG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
VOCAB_PATH="segearth_r2/model/lasers_category_vocab.json"

########################################
# LaSeRS (train) + cross-dataset test paths
########################################
BASE_DATA_PATH="/root/rivermind-data/huangziyi/data/LaSeRS"
DATASET_NAME="lasers"
EVAL_SPLIT="test"
LASERS_BENCHMARK="${LASERS_BENCHMARK:-all}"
DIAG_LASERS_BENCHMARK="${DIAG_LASERS_BENCHMARK:-all}"

DATA_PATH_LASERS="/root/rivermind-data/huangziyi/data/LaSeRS"
DATA_PATH_RRSISD="/root/rivermind-data/huangziyi/data/RRSISD"
DATA_PATH_REFSEGRS="/root/rivermind-data/huangziyi/data/RefSegRS"
DATA_PATH_RISBENCH="/root/rivermind-data/huangziyi/data/RISBench_dataset"
DATA_PATH_EARTHREASON="/root/rivermind-data/huangziyi/data/EarthReason"

########################################
# Output
########################################
OUTPUT_DIR="${OUTPUT_DIR:-/root/rivermind-data/huangziyi/reseg/output/set/clite-v2-from-scratch-50k}"
MERGED_DIR="${OUTPUT_DIR}/merged_model"
DIAG_OUTPUT_DIR="${OUTPUT_DIR}/lasers_diagnostic_eval_v2"
LOG_FILE="${OUTPUT_DIR}/train.log"
SERIAL_EVAL_LOG="${OUTPUT_DIR}/all_datasets_test_eval_serial.log"

########################################
# Eval metrics
########################################
EVAL_METRICS_SCRIPT="/root/rivermind-data/huangziyi/reseg/eval_val_metrics.py"
EVAL_USE_WANDB="${EVAL_USE_WANDB:-True}"
EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-segearth-set-eval}"
EVAL_WANDB_RUN_NAME="${EVAL_WANDB_RUN_NAME:-clite-v2-from-scratch-50k}"
EVAL_DATALOADER_NUM_WORKERS="${EVAL_DATALOADER_NUM_WORKERS:-2}"

########################################
# Train config (align with base/joint 80k recipe)
########################################
MAX_STEPS="${MAX_STEPS:-80000}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
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

LASERS_HOLDOUT_RATIO="${LASERS_HOLDOUT_RATIO:-0.05}"
LASERS_HOLDOUT_SEED="${LASERS_HOLDOUT_SEED:-42}"

# SET modules (full joint — not frozen A3)
LAMBDA_SET_COUNT="${LAMBDA_SET_COUNT:-0.05}"
LAMBDA_SET_CATEGORY="${LAMBDA_SET_CATEGORY:-0.1}"
SET_MAX_COUNT="${SET_MAX_COUNT:-10}"
SET_CONDITIONER_LAYERS="${SET_CONDITIONER_LAYERS:-2}"
SET_CONDITIONER_HEADS="${SET_CONDITIONER_HEADS:-4}"
SET_CONDITIONER_GATE_INIT="${SET_CONDITIONER_GATE_INIT:-0.001}"

# C-lite-v2: q_set_fusion MLP bottleneck dim (default = mask decoder hidden_dim, 256)
Q_SET_FUSION_HIDDEN="${Q_SET_FUSION_HIDDEN:-256}"

TRAIN_ONLY="${TRAIN_ONLY:-0}"
FRESH_START="${FRESH_START:-0}"
RUN_DIAGNOSTIC="${RUN_DIAGNOSTIC:-1}"

########################################
# Helpers
########################################
stop_training_jobs () {
  local pids
  pids=$(pgrep -f -- "--output_dir ${OUTPUT_DIR}" 2>/dev/null || true)
  if [[ -z "${pids}" ]]; then
    return 0
  fi
  echo "[INFO] stopping existing jobs for ${OUTPUT_DIR}: ${pids}"
  kill ${pids} 2>/dev/null || true
  sleep 3
  pids=$(pgrep -f -- "--output_dir ${OUTPUT_DIR}" 2>/dev/null || true)
  if [[ -n "${pids}" ]]; then
    echo "[WARN] force killing jobs: ${pids}"
    kill -9 ${pids} 2>/dev/null || true
    sleep 1
  fi
}

read_best_checkpoint () {
  local out_dir="$1"
  OUTPUT_DIR="${out_dir}" "${PYTHON}" - <<'PY'
import json, os, sys
output_dir = os.environ["OUTPUT_DIR"]
trainer_state = os.path.join(output_dir, "trainer_state.json")
if not os.path.exists(trainer_state):
    print(""); sys.exit(0)
state = json.load(open(trainer_state, encoding="utf-8"))
best = state.get("best_model_checkpoint") or ""
print(best)
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
    --lora_dropout "${LORA_DROPOUT}" \
    --lasers_category_vocab_path "${VOCAB_PATH}" \
    --use_explicit_set_token
}

verify_merged_set () {
  PYTHONPATH="${REPO_DIR}" "${PYTHON}" - <<PY
from segearth_r2.datasets.dataset import get_mask_config
from segearth_r2.model.language_model.llava_phi import SegEarthR2

mask_cfg = get_mask_config("${MASK_CONFIG}")
model = SegEarthR2.from_pretrained("${MERGED_DIR}", mask_decoder_cfg=mask_cfg, device_map="cpu")
if getattr(model.config, "use_set_conditioner", False) and getattr(model, "set_conditioner", None) is None:
    model.init_set_conditioning_modules(model.config)
assert getattr(model.config, "use_set_conditioner", False), "use_set_conditioner missing from merged config"
assert model.set_conditioner is not None, "set_conditioner not loaded"

# C-lite-v2: verify SET_token_projector and q_set_fusion
assert getattr(model.config, "use_explicit_set_token", False), "use_explicit_set_token missing from merged config"
assert hasattr(model, "SET_token_projector") and model.SET_token_projector is not None, "SET_token_projector not loaded"
assert hasattr(model, "q_set_fusion") and model.q_set_fusion is not None, "q_set_fusion not loaded"

shape = tuple(model.category_set_head.head.weight.shape)
print("[OK] merged model loads; use_set_conditioner=True; use_explicit_set_token=True")
print("[OK] SET_token_projector and q_set_fusion present; category_set_head shape =", shape)
PY
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

run_all_datasets_serial () {
  local model_dir="$1"
  [[ -f "${model_dir}/config.json" ]] || { echo "[ERROR] merged model not found: ${model_dir}/config.json"; exit 1; }

  mkdir -p "${OUTPUT_DIR}"
  {
    echo "========================================"
    echo "[SERIAL-TEST] start $(date -Iseconds)"
    echo "OUT_DIR=${OUTPUT_DIR}"
    echo "MODEL=${model_dir}"
    echo "========================================"
  } | tee "${SERIAL_EVAL_LOG}"

  run_one_dataset_eval lasers "${DATA_PATH_LASERS}" "${model_dir}"
  run_one_dataset_eval rrsisd "${DATA_PATH_RRSISD}" "${model_dir}"
  run_one_dataset_eval refsegrs "${DATA_PATH_REFSEGRS}" "${model_dir}"
  run_one_dataset_eval risbench "${DATA_PATH_RISBENCH}" "${model_dir}"
  run_one_dataset_eval earthreason "${DATA_PATH_EARTHREASON}" "${model_dir}"

  {
    echo "========================================"
    echo "[SERIAL-TEST] finished $(date -Iseconds)"
    echo "LaSeRS summary     : ${OUTPUT_DIR}/lasers_${EVAL_SPLIT}_metrics_summary.json"
    echo "RRSISD metrics     : ${OUTPUT_DIR}/rrsisd_test_metrics.json"
    echo "RefSegRS metrics   : ${OUTPUT_DIR}/refsegrs_test_metrics.json"
    echo "RISBench metrics   : ${OUTPUT_DIR}/risbench_test_metrics.json"
    echo "EarthReason metrics: ${OUTPUT_DIR}/earthreason_test_metrics.json"
    echo "Master log         : ${SERIAL_EVAL_LOG}"
    echo "========================================"
  } | tee -a "${SERIAL_EVAL_LOG}"
}

run_diagnostic () {
  local model_dir="$1"
  local out_dir="$2"
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
    --resume
}

########################################
# 0) Preflight
########################################
echo "========================================"
echo "[0/7] Preflight (C-lite-v2 from scratch)"
echo "========================================"
echo "[INFO] REPO_DIR=${REPO_DIR}"
echo "[INFO] MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "[INFO] VISION_TOWER=${VISION_TOWER}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] MAX_STEPS=${MAX_STEPS}, SAVE_STEPS=${SAVE_STEPS}"
echo "[INFO] SET layers=${SET_CONDITIONER_LAYERS}, gate_init=${SET_CONDITIONER_GATE_INIT}"
echo "[INFO] C-lite-v2: use_explicit_set_token=True"
echo "[INFO] C-lite-v2: q_set_fusion_hidden=${Q_SET_FUSION_HIDDEN}"
echo "[INFO] a3_train_only_set_modules=False, lora_enable=True"

if [[ "${MODEL_NAME_OR_PATH}" == *merged_model* ]]; then
  echo "[ERROR] from-scratch run must not use merged_model as model_name_or_path"
  exit 1
fi

[[ -d "${MODEL_NAME_OR_PATH}" ]] || { echo "[ERROR] Mipha-3B not found: ${MODEL_NAME_OR_PATH}"; exit 1; }
[[ -d "${VISION_TOWER}" ]] || { echo "[ERROR] vision tower not found: ${VISION_TOWER}"; exit 1; }
[[ -f "${VISION_TOWER_MASK}" ]] || { echo "[ERROR] mask2former weights not found: ${VISION_TOWER_MASK}"; exit 1; }
[[ -f "${VOCAB_PATH}" ]] || { echo "[ERROR] category vocab not found: ${VOCAB_PATH}"; exit 1; }
[[ -f "${BASE_DATA_PATH}/train/annotations/train_data.json" ]] || { echo "[ERROR] LaSeRS train missing"; exit 1; }
[[ -d "${BASE_DATA_PATH}/test/annotations" ]] || { echo "[ERROR] LaSeRS test missing"; exit 1; }
[[ -f "scripts/zero1.json" ]] || { echo "[ERROR] zero1.json missing"; exit 1; }
[[ -f "${EVAL_METRICS_SCRIPT}" ]] || { echo "[ERROR] eval metrics script not found: ${EVAL_METRICS_SCRIPT}"; exit 1; }

for ds_path in "${DATA_PATH_LASERS}" "${DATA_PATH_RRSISD}" "${DATA_PATH_REFSEGRS}" "${DATA_PATH_RISBENCH}" "${DATA_PATH_EARTHREASON}"; do
  [[ -d "${ds_path}" ]] || { echo "[ERROR] dataset path missing: ${ds_path}"; exit 1; }
done

mkdir -p "${OUTPUT_DIR}"

if [[ "${FRESH_START}" == "1" ]]; then
  echo "[INFO] FRESH_START=1: stop stale jobs, remove checkpoints, rotate train.log"
  stop_training_jobs
  rm -rf "${OUTPUT_DIR}"/checkpoint-* "${OUTPUT_DIR}"/trainer_state.json
  if [[ -f "${LOG_FILE}" ]]; then
    mv "${LOG_FILE}" "${LOG_FILE}.bak.$(date +%s)"
  fi
fi

if pgrep -f -- "--output_dir ${OUTPUT_DIR}" > /dev/null; then
  echo "[ERROR] training already running for output_dir=${OUTPUT_DIR}"
  pgrep -af -- "--output_dir ${OUTPUT_DIR}"
  exit 1
fi

echo "[OK] preflight passed"

########################################
# 1) Train — C-lite-v2 (LoRA + SEG path + SET modules + SET_token_projector + q_set_fusion)
########################################
echo "========================================"
echo "[1/7] Train C-lite-v2 on LaSeRS (from Mipha-3B + SigLIP2)"
echo "========================================"

{
  echo "=== clite-v2 from scratch $(date -Iseconds) ==="
  echo "model=${MODEL_NAME_OR_PATH}"
  echo "vision_tower=${VISION_TOWER}"
  echo "output=${OUTPUT_DIR}"
  echo "set_layers=${SET_CONDITIONER_LAYERS}"
  echo "use_explicit_set_token=True"
  echo "q_set_fusion_hidden=${Q_SET_FUSION_HIDDEN}"
} | tee "${LOG_FILE}"

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
  --lora_enable True \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --train_clip_backbone False \
  --train_swin_backbone False \
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
  --use_explicit_set_token True \
  --q_set_fusion_hidden "${Q_SET_FUSION_HIDDEN}" \
  --a3_train_only_set_modules False \
  --report_to wandb 2>&1 | tee -a "${LOG_FILE}"

if [[ "${TRAIN_ONLY}" == "1" ]]; then
  echo "========================================"
  echo "[DONE] TRAIN_ONLY=1 — skipping merge / eval / diagnostic"
  echo "Output dir: ${OUTPUT_DIR}"
  echo "========================================"
  exit 0
fi

########################################
# 2) Select checkpoint
########################################
echo "========================================"
echo "[2/7] Select checkpoint"
echo "========================================"
SELECTED_CHECKPOINT=$(read_best_checkpoint "${OUTPUT_DIR}")
if [[ -z "${SELECTED_CHECKPOINT}" ]]; then
  echo "[WARN] best_model_checkpoint not found; fallback to last"
  SELECTED_CHECKPOINT=$(read_last_checkpoint "${OUTPUT_DIR}")
fi
[[ -n "${SELECTED_CHECKPOINT}" ]] || { echo "[ERROR] no checkpoint"; exit 1; }
echo "[OK] SELECTED_CHECKPOINT=${SELECTED_CHECKPOINT}"

########################################
# 3) Merge LoRA + SET weights
########################################
echo "========================================"
echo "[3/7] Merge LoRA checkpoint"
echo "========================================"
merge_ckpt "${SELECTED_CHECKPOINT}" "${MERGED_DIR}"

########################################
# 4) Verify merged model
########################################
echo "========================================"
echo "[4/7] Verify merged model (C-lite-v2 config)"
echo "========================================"
[[ -f "${MERGED_DIR}/config.json" ]] || { echo "[ERROR] merged config missing: ${MERGED_DIR}/config.json"; exit 1; }
verify_merged_set

########################################
# 5) Serial test eval (5 datasets)
########################################
echo "========================================"
echo "[5/7] Serial test eval (5 datasets)"
echo "========================================"
run_all_datasets_serial "${MERGED_DIR}"

########################################
# 6) LaSeRS generation diagnostic (optional, resumable)
########################################
if [[ "${RUN_DIAGNOSTIC}" == "1" ]]; then
  echo "========================================"
  echo "[6/7] LaSeRS generation diagnostic (${DIAG_LASERS_BENCHMARK})"
  echo "========================================"
  run_diagnostic "${MERGED_DIR}" "${DIAG_OUTPUT_DIR}"
else
  echo "[SKIP] RUN_DIAGNOSTIC=0"
fi

########################################
# 7) Done
########################################
echo "========================================"
echo "[7/7] DONE"
echo "Output dir           : ${OUTPUT_DIR}"
echo "Selected ckpt        : ${SELECTED_CHECKPOINT}"
echo "Merged model         : ${MERGED_DIR}"
echo "LaSeRS summary       : ${OUTPUT_DIR}/lasers_${EVAL_SPLIT}_metrics_summary.json"
echo "Cross-dataset metrics: ${OUTPUT_DIR}/*_test_metrics.json"
echo "Diagnostic JSONL     : ${DIAG_OUTPUT_DIR}/lasers_per_sample_diagnostic.jsonl"
echo "Serial eval log      : ${SERIAL_EVAL_LOG}"
echo "Train log            : ${LOG_FILE}"
echo "Compare baseline     : output/base/standard-base-lasers-siglip1-28w-gd4"
echo "Compare frozen SET   : output/set/a3-frozen-lasers-set-5w"
echo "Compare joint SET    : output/set/joint-set-siglip2-from-scratch-80k"
echo "Post-hoc analysis    : segearth+set/scripts/analyze_a3_set_diagnostic.py"
echo "========================================"
