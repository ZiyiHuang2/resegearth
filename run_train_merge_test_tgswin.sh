#!/usr/bin/env bash
set -euo pipefail

########################################
# TG-Swin-WTI: Train → Merge → LaSeRS Test Eval
#
# LaSeRS: train + test only (no official val)
#   - train: train_data.json
#   - train-time eval: 5% holdout from train (not test)
#   - final eval: test split + taxonomy
#
# ── Quick start (v1.5 default) ──────────────────────────────────────────
#   GPU_ID=0 bash run_train_merge_test_tgswin.sh
#
# ── v1.6 module ablation (preset) ─────────────────────────────────────────
#   GPU_ID=0 ABLATION=B1 MAX_STEPS=10000 bash run_train_merge_test_tgswin.sh
#   GPU_ID=1 ABLATION=B2 MAX_STEPS=10000 bash run_train_merge_test_tgswin.sh
#   GPU_ID=2 ABLATION=B3 MAX_STEPS=10000 bash run_train_merge_test_tgswin.sh
#
#   B0 = v1.5 equiv (HTER off, PWER state off) — uses v15 yaml
#   B1 = HTER only  (stage phrase + hybrid ρ)
#   B2 = PWER only  (evidence state)
#   B3 = full v1.6  (all on)
#
# ── Manual override (full control) ────────────────────────────────────────
#   GPU_ID=0 \
#   WARMSTART=/root/rivermind-data/huangziyi/reseg/output/tgswin/tgswin-wti-v15-lasers-warmstart-10w-bs2-gd4/merged_model \
#   MASK_CONFIG=segearth_r2/model/mask_decoder/mask_config/maskformer2_tgswin_v16_B1_hter.yaml \
#   MAX_STEPS=10000 SAVE_STEPS=5000 SAVE_TOTAL_LIMIT=1 \
#   OUTPUT_DIR=/root/rivermind-data/huangziyi/reseg/output/tgswin/tgswin-v16-ablation-B1-hter-10k \
#   WANDB_NAME=tgswin-v16-B1-hter-10k \
#   bash run_train_merge_test_tgswin.sh
#
# ── Eval / merge only ─────────────────────────────────────────────────────
#   RUN_TRAIN=0 MERGE_CHECKPOINT=.../checkpoint-10000 OUTPUT_DIR=... bash ...
#
# IMPORTANT:
#   - OUTPUT_DIR must live under ${RESEG_ROOT}/output/ (NOT /output on root disk)
#   - For v1.6 ablations, set WARMSTART to v1.5 merged (see default below)
#   - Each DeepSpeed checkpoint ≈ 9 GB; use SAVE_TOTAL_LIMIT=1 on small disks
########################################

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT="${WANDB_PROJECT:-segearth-tgswin}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"

# ── Paths ─────────────────────────────────────────────────────────────────
RESEG_ROOT="${RESEG_ROOT:-/root/rivermind-data/huangziyi/reseg}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/root/rivermind-data/miniconda3/envs/reseg}"
PYTHON="${CONDA_ENV_DIR}/bin/python"
DEEPSPEED="${CONDA_ENV_DIR}/bin/deepspeed"
export PATH="${CONDA_ENV_DIR}/bin:${PATH}"

REPO_DIR="${REPO_DIR:-${RESEG_ROOT}/segearth+tgswin}"
cd "${REPO_DIR}"

MASK_CONFIG_DIR="${REPO_DIR}/segearth_r2/model/mask_decoder/mask_config"

# Warmstart anchors
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${RESEG_ROOT}/output/base/standard-base-lasers-siglip1-8w-gd4/merged_model}"
V15_WARMSTART="${V15_WARMSTART:-${RESEG_ROOT}/output/tgswin/tgswin-wti-v15-lasers-warmstart-10w-bs2-gd4/merged_model}"
WARMSTART="${WARMSTART:-${V15_WARMSTART}}"

# MODEL_PATH priority: explicit MODEL_PATH > WARMSTART > BASE_MODEL_PATH
MODEL_PATH="${MODEL_PATH:-${WARMSTART}}"

VISION_TOWER="${RESEG_ROOT}/pretrained_model/CLIP/siglip-so400m-patch14-384"
VISION_TOWER_MASK="${RESEG_ROOT}/pretrained_model/mask2former/model_final_54b88a.pkl"

BASE_DATA_PATH="${BASE_DATA_PATH:-/root/rivermind-data/huangziyi/data/LaSeRS}"
DATASET_NAME="lasers"
EVAL_SPLIT="test"
LASERS_BENCHMARK="${LASERS_BENCHMARK:-all}"
LASERS_HOLDOUT_RATIO="${LASERS_HOLDOUT_RATIO:-0.05}"

# ── Ablation preset (optional: B0|B1|B2|B3) ───────────────────────────────
ABLATION="${ABLATION:-}"
case "${ABLATION}" in
  "" ) ;;
  B0|b0)
    MASK_CONFIG="${MASK_CONFIG:-${MASK_CONFIG_DIR}/maskformer2_tgswin_v15.yaml}"
    RUN_TAG="${RUN_TAG:-v16-ablation-B0-v15}"
    ;;
  B1|b1)
    MASK_CONFIG="${MASK_CONFIG:-${MASK_CONFIG_DIR}/maskformer2_tgswin_v16_B1_hter.yaml}"
    RUN_TAG="${RUN_TAG:-v16-ablation-B1-hter}"
    ;;
  B2|b2)
    MASK_CONFIG="${MASK_CONFIG:-${MASK_CONFIG_DIR}/maskformer2_tgswin_v16_B2_pwer.yaml}"
    RUN_TAG="${RUN_TAG:-v16-ablation-B2-pwer}"
    ;;
  B3|b3)
    MASK_CONFIG="${MASK_CONFIG:-${MASK_CONFIG_DIR}/maskformer2_tgswin_v16.yaml}"
    RUN_TAG="${RUN_TAG:-v16-ablation-B3-full}"
    ;;
  *)
    echo "[ERROR] Unknown ABLATION=${ABLATION} (use B0|B1|B2|B3 or leave empty)"
    exit 1
    ;;
esac

MASK_CONFIG="${MASK_CONFIG:-segearth_r2/model/mask_decoder/mask_config/maskformer2_tgswin_v15.yaml}"
# Resolve relative mask config to repo path
if [[ "${MASK_CONFIG}" != /* ]]; then
  MASK_CONFIG="${REPO_DIR}/${MASK_CONFIG}"
fi

MAX_STEPS="${MAX_STEPS:-50000}"
RUN_TAG="${RUN_TAG:-tgswin-${MAX_STEPS}steps}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESEG_ROOT}/output/tgswin/${RUN_TAG}}"
MERGED_DIR="${MERGED_DIR:-${OUTPUT_DIR}/merged_model}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_DIR}/test_results}"

export WANDB_NAME="${WANDB_NAME:-${RUN_TAG}}"
EVAL_WANDB_RUN_NAME="${EVAL_WANDB_RUN_NAME:-${RUN_TAG}}"

# ── GPU / training hyper-params ───────────────────────────────────────────
GPU_ID="${GPU_ID:-0}"
GPU_SLOT="${GPU_SLOT:-localhost:${GPU_ID}}"
MASTER_PORT="${MASTER_PORT:-29641}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
# Short runs: keep 1 checkpoint to save ~9 GB per extra ckpt
if [[ -z "${SAVE_TOTAL_LIMIT:-}" ]]; then
  if [[ "${MAX_STEPS}" -le 10000 ]]; then
    SAVE_TOTAL_LIMIT=1
  else
    SAVE_TOTAL_LIMIT=2
  fi
fi
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
MIN_DISK_GB="${MIN_DISK_GB:-20}"

EVAL_METRICS_SCRIPT="${RESEG_ROOT}/eval_val_metrics.py"
EVAL_USE_WANDB="True"
EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-segearth-eval-tgswin}"

TAXONOMY_SCRIPT="${RESEG_ROOT}/segearth+base/tools/diagnostics/audit_lasers_error_taxonomy.py"
TAXONOMY_ANN_FILE="${BASE_DATA_PATH}/test/annotations/test_multi_cate.json"
TAXONOMY_OUT_JSON="${TAXONOMY_OUT_JSON:-${RESEG_ROOT}/plans/baseline_diagnostics/${RUN_TAG}_multi_cate_error_taxonomy.json}"
TAXONOMY_OUT_MD="${TAXONOMY_OUT_MD:-${RESEG_ROOT}/plans/baseline_diagnostics/${RUN_TAG}_multi_cate_error_taxonomy.md}"

# ── Helpers ───────────────────────────────────────────────────────────────
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
  [[ -f "${TAXONOMY_SCRIPT}" ]] || { echo "[ERROR] taxonomy script not found: ${TAXONOMY_SCRIPT}"; exit 1; }
  [[ -d "${pred_dir}" ]] || { echo "[ERROR] pred-dir not found: ${pred_dir}"; exit 1; }
  [[ -f "${TAXONOMY_ANN_FILE}" ]] || { echo "[ERROR] ann-file not found: ${TAXONOMY_ANN_FILE}"; exit 1; }
  mkdir -p "$(dirname "${TAXONOMY_OUT_JSON}")"
  echo "[INFO] Running multi_cate error taxonomy..."
  "${PYTHON}" "${TAXONOMY_SCRIPT}" \
    --pred-dir "${pred_dir}" \
    --ann-file "${TAXONOMY_ANN_FILE}" \
    --split-name test_multi_cate \
    --out-json "${TAXONOMY_OUT_JSON}" \
    --out-md "${TAXONOMY_OUT_MD}"
  echo "[OK] taxonomy: ${TAXONOMY_OUT_JSON}"
}

check_merged_tgswin_model () {
  MERGED_MODEL_DIR="${MERGED_DIR}" "${PYTHON}" - <<'PY'
import json, os
model_dir = os.environ["MERGED_MODEL_DIR"]
index_path = os.path.join(model_dir, "model.safetensors.index.json")
if not os.path.isfile(index_path):
    print(f"[WARN] missing weight index: {index_path}")
    raise SystemExit(0)
with open(index_path) as f:
    keys = list(json.load(f).get("weight_map", {}).keys())
required = ("tg_swin_tcf", "tg_swin_controller")
missing = [r for r in required if not any(r in k for k in keys)]
if missing:
    print(f"[WARN] merged model may miss TG-Swin weights: {missing}")
else:
    for r in required:
        matched = [k for k in keys if r in k]
        print(f"[OK] TG-Swin weight: {matched[0] if matched else 'N/A'}")
PY
}

assert_output_dir_safe () {
  local out_dir="$1"
  # Must be under RESEG_ROOT (large PVC), not /output on root overlay
  if [[ "${out_dir}" != "${RESEG_ROOT}"/* ]]; then
    echo "[ERROR] OUTPUT_DIR must be under ${RESEG_ROOT}/output/"
    echo "        Got: ${out_dir}"
    echo "[HINT]  Use: OUTPUT_DIR=${RESEG_ROOT}/output/tgswin/your-run-name"
    exit 1
  fi
  mkdir -p "${out_dir}"
  local avail_kb
  avail_kb="$(df -Pk "${out_dir}" | awk 'NR==2 {print $4}')"
  local avail_gb=$(( avail_kb / 1024 / 1024 ))
  echo "[INFO] Disk free at output: ${avail_gb} GB (need >= ${MIN_DISK_GB} GB per checkpoint ~9 GB)"
  if [[ "${avail_gb}" -lt "${MIN_DISK_GB}" ]]; then
    echo "[ERROR] Not enough disk space at ${out_dir}"
    echo "[HINT]  rm old checkpoints or set SAVE_TOTAL_LIMIT=1"
    exit 1
  fi
}

cleanup_failed_checkpoints () {
  local out_dir="$1"
  find "${out_dir}" -maxdepth 1 -type d -name 'tmp-checkpoint-*' -print -exec rm -rf {} + 2>/dev/null || true
}

# ── Preflight ─────────────────────────────────────────────────────────────
echo "========================================"
echo "[0/7] Preflight — TG-Swin-WTI"
echo "========================================"
echo "[INFO] REPO_DIR=${REPO_DIR}"
echo "[INFO] ABLATION=${ABLATION:-none}"
echo "[INFO] RUN_TAG=${RUN_TAG}"
echo "[INFO] MASK_CONFIG=${MASK_CONFIG}"
echo "[INFO] WARMSTART=${WARMSTART}"
echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] GPU_ID=${GPU_ID}"
echo "[INFO] MAX_STEPS=${MAX_STEPS}  batch=${PER_DEVICE_TRAIN_BATCH_SIZE}  save_every=${SAVE_STEPS}  keep=${SAVE_TOTAL_LIMIT}"
echo "[INFO] RUN_TRAIN=${RUN_TRAIN} RUN_MERGE=${RUN_MERGE} RUN_EVAL=${RUN_EVAL} RUN_TAXONOMY=${RUN_TAXONOMY}"
echo "[INFO] WANDB_NAME=${WANDB_NAME}"

assert_output_dir_safe "${OUTPUT_DIR}"
cleanup_failed_checkpoints "${OUTPUT_DIR}"

MASK_CONFIG="${MASK_CONFIG}" "${PYTHON}" - <<'PY'
import os
from segearth_r2.datasets.dataset import get_mask_config
cfg = get_mask_config(os.environ["MASK_CONFIG"])
tg = getattr(cfg, "TG_SWIN", None)
if not tg:
    print("[ERROR] TG_SWIN block missing in mask config")
    raise SystemExit(1)
print(f"[INFO] TG_SWIN VERSION={getattr(tg, 'VERSION', 'v1')}")
print(f"[INFO] TG_SWIN ENABLED={getattr(tg, 'ENABLED', False)}")
print(f"[INFO] TG_SWIN HEAD_AWARE={getattr(tg, 'HEAD_AWARE', False)}")
print(f"[INFO] TG_SWIN USE_STAGE_PHRASE={getattr(tg, 'USE_STAGE_PHRASE', False)}")
print(f"[INFO] TG_SWIN USE_HYBRID_RELIABILITY={getattr(tg, 'USE_HYBRID_RELIABILITY', False)}")
print(f"[INFO] TG_SWIN USE_EVIDENCE_STATE={getattr(tg, 'USE_EVIDENCE_STATE', False)}")
print(f"[INFO] TG_SWIN WTI_STAGES={list(getattr(tg, 'WTI_STAGES', []))}")
PY

[[ -d "${MODEL_PATH}" ]] || { echo "[ERROR] model not found: ${MODEL_PATH}"; exit 1; }
[[ -d "${VISION_TOWER}" ]] || { echo "[ERROR] vision tower not found: ${VISION_TOWER}"; exit 1; }
[[ -f "${VISION_TOWER_MASK}" ]] || { echo "[ERROR] vision tower mask not found: ${VISION_TOWER_MASK}"; exit 1; }
[[ -f "${MASK_CONFIG}" ]] || { echo "[ERROR] mask config not found: ${MASK_CONFIG}"; exit 1; }
[[ -f "scripts/zero1.json" ]] || { echo "[ERROR] DeepSpeed config not found: ${REPO_DIR}/scripts/zero1.json"; exit 1; }

if [[ "${MASK_CONFIG}" == *"v16"* && "${MODEL_PATH}" == "${BASE_MODEL_PATH}" ]]; then
  echo "[WARN] v1.6 config but MODEL_PATH is base model, not v1.5 warmstart"
  echo "[WARN] Set WARMSTART=${V15_WARMSTART} or MODEL_PATH explicitly"
fi

for req in \
  "${BASE_DATA_PATH}/train/annotations/train_data.json" \
  "${BASE_DATA_PATH}/train/images" \
  "${BASE_DATA_PATH}/test/annotations" \
  "${BASE_DATA_PATH}/test/images" \
  "${BASE_DATA_PATH}/test/annotations/test_multi_cate.json"
do
  [[ -e "${req}" ]] || { echo "[ERROR] missing: ${req}"; exit 1; }
done

echo "[OK] preflight passed"

# ── Train ─────────────────────────────────────────────────────────────────
if [[ "${RUN_TRAIN}" == "1" ]]; then
  echo "========================================"
  echo "[1/7] Training (LaSeRS train, holdout ${LASERS_HOLDOUT_RATIO} eval)"
  echo "========================================"
  if ls "${OUTPUT_DIR}"/checkpoint-* &>/dev/null; then
    echo "[INFO] Found existing checkpoint(s) — train.py will auto-resume"
  fi
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
  echo "[1/7] SKIP training"
fi

# ── Merge ─────────────────────────────────────────────────────────────────
BEST_CHECKPOINT="${MERGE_CHECKPOINT:-$(read_best_checkpoint "${OUTPUT_DIR}")}"
[[ -n "${BEST_CHECKPOINT}" ]] || BEST_CHECKPOINT="$(read_last_checkpoint "${OUTPUT_DIR}")"
[[ -n "${BEST_CHECKPOINT}" ]] || { echo "[ERROR] no checkpoint under ${OUTPUT_DIR}"; exit 1; }
echo "[2/7] SELECTED_CHECKPOINT=${BEST_CHECKPOINT}"

if [[ "${RUN_MERGE}" == "1" ]]; then
  echo "[3/7] Merge LoRA + TG-Swin → ${MERGED_DIR}"
  merge_ckpt "${BEST_CHECKPOINT}" "${MERGED_DIR}"
else
  echo "[3/7] SKIP merge"
fi

echo "[4/7] Check merged TG-Swin weights"
check_merged_tgswin_model

# ── Eval ────────────────────────────────────────────────────────────────────
if [[ "${RUN_EVAL}" == "1" ]]; then
  echo "[5/7] Eval → ${EVAL_OUTPUT_DIR}"
  eval_model "${MERGED_DIR}" "${EVAL_OUTPUT_DIR}"
  run_eval_metrics "${EVAL_OUTPUT_DIR}" "${EVAL_WANDB_RUN_NAME}"
  if [[ "${RUN_TAXONOMY}" == "1" ]]; then
    echo "[6/7] Taxonomy (test_multi_cate)"
    run_taxonomy "${EVAL_OUTPUT_DIR}"
  else
    echo "[6/7] SKIP taxonomy"
  fi
else
  echo "[5/7] SKIP eval"
  echo "[6/7] SKIP taxonomy"
fi

echo "[7/7] DONE"
echo "  output_dir=${OUTPUT_DIR}"
echo "  merged_dir=${MERGED_DIR}"
echo "  eval_dir=${EVAL_OUTPUT_DIR}"
