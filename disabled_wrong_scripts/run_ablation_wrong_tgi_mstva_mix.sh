#!/usr/bin/env bash
# =============================================================================
# RRSIS-D long-run ablation: baseline_raw vs public_semantic_v2 (train only).
# Lives under resegearth+source; trains via resegearth+tgi (v2 JSON in ../configs).
#
# Usage (from anywhere):
#   bash /path/to/resegearth+source/scripts/rrsisd_public_semantic_v2_longrun_ablation/run_ablation.sh <smoke|long> <baseline|v2|both>
#
# Optional: RESEG_ROOT, REPO_TGI, MODEL_NAME_OR_PATH, BASE_DATA_PATH, PYTHON_BIN,
#           GPU_SLOT, MASTER_PORT, REPORT_TO, WANDB_*, hyper env vars (see tgi twin).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_SOURCE="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESEG_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO_TGI="${REPO_TGI:-${RESEG_ROOT}/resegearth+tgi}"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
unset CUDA_VISIBLE_DEVICES

PHASE="${1:-}"
ARM="${2:-}"

if [[ "${PHASE}" != "smoke" && "${PHASE}" != "long" ]]; then
  echo "Usage: $0 <smoke|long> <baseline|v2|both>"
  exit 1
fi
if [[ "${ARM}" != "baseline" && "${ARM}" != "v2" && "${ARM}" != "both" ]]; then
  echo "Usage: $0 <smoke|long> <baseline|v2|both>"
  exit 1
fi

GPU_SLOT="${GPU_SLOT:-localhost:0}"
MASTER_PORT="${MASTER_PORT:-29531}"
REPORT_TO="${REPORT_TO:-none}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_TGI}"

CONCEPT_LIB="${CONCEPT_LIB:-${REPO_SOURCE}/configs/concept_public_semantic_library_v2.json}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${RESEG_ROOT}/output/bseg/baseline_standard-base_5w/merged_model}"
VISION_TOWER="${VISION_TOWER:-${RESEG_ROOT}/pretrained_model/CLIP/siglip2-so400m-patch14-384}"
VISION_TOWER_MASK="${VISION_TOWER_MASK:-${RESEG_ROOT}/pretrained_model/mask2former/maskformer2_swin_base_IN21k_384_bs16_50ep.pkl}"
MASK_CONFIG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"

BASE_DATA_PATH="${BASE_DATA_PATH:-/home/wangchengjun/huangziyi/data/RRSISD}"
DATASET_NAME="rrsisd"

OUT_BASELINE="${OUT_BASELINE:-${REPO_TGI}/checkpoints/rrsisd_baseline_raw}"
OUT_V2="${OUT_V2:-${REPO_TGI}/checkpoints/rrsisd_public_semantic_v2}"

if [[ "${PHASE}" == "smoke" ]]; then
  MAX_STEPS="${MAX_STEPS:-100}"
  SAVE_STEPS="${SAVE_STEPS:-100}"
  SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
  LOGGING_STEPS="${LOGGING_STEPS:-5}"
else
  MAX_STEPS="${MAX_STEPS:-70000}"
  SAVE_STEPS="${SAVE_STEPS:-2000}"
  SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
  LOGGING_STEPS="${LOGGING_STEPS:-10}"
fi

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
BF16="${BF16:-False}"
FP16="${FP16:-True}"
TF32="${TF32:-False}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-False}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
DATA_RATIO="${DATA_RATIO:-1}"
SWITCH_BS="${SWITCH_BS:-4}"
SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-42}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

TRAIN_MIDSTAGE_RECALIBRATION="${TRAIN_MIDSTAGE_RECALIBRATION:-True}"
STAGE3_NORM_ONLY="${STAGE3_NORM_ONLY:-False}"
USE_ATTENTION_LOSS="${USE_ATTENTION_LOSS:-False}"
USE_MIDSTAGE_GATE_LOSS="${USE_MIDSTAGE_GATE_LOSS:-False}"
MIDSTAGE_GATE_LOSS_WEIGHT="${MIDSTAGE_GATE_LOSS_WEIGHT:-0.0}"
USE_MSTVA="${USE_MSTVA:-True}"
MSTVA_ALIGN_DIM="${MSTVA_ALIGN_DIM:-256}"
USE_MSTVA_LOSS="${USE_MSTVA_LOSS:-True}"
MSTVA_LOSS_WEIGHT="${MSTVA_LOSS_WEIGHT:-0.01}"
MSTVA_SCALE_WEIGHTS="${MSTVA_SCALE_WEIGHTS:-0.5,0.3,0.2}"

run_one_arm () {
  local arm="$1"
  local output_dir wandb_name use_v2=0

  if [[ "${arm}" == "baseline" ]]; then
    output_dir="${OUT_BASELINE}"
    wandb_name="${WANDB_BASELINE_NAME:-rrsisd_longrun_baseline_raw}"
  elif [[ "${arm}" == "v2" ]]; then
    output_dir="${OUT_V2}"
    wandb_name="${WANDB_V2_NAME:-rrsisd_longrun_public_semantic_v2}"
    use_v2=1
  else
    echo "[ERROR] bad arm=${arm}"
    exit 1
  fi

  mkdir -p "${output_dir}"

  if [[ "${REPORT_TO}" == "wandb" ]]; then
    export WANDB_PROJECT="${WANDB_PROJECT:-segearth-tgi-rrsisd-ablation}"
    export WANDB_NAME="${wandb_name}"
  fi

  echo "========================================"
  echo "[TRAIN] phase=${PHASE} arm=${arm}"
  echo "[TRAIN] REPO_TGI=${REPO_TGI}"
  echo "[TRAIN] CONCEPT_LIB=${CONCEPT_LIB}"
  echo "[TRAIN] output_dir=${output_dir}"
  echo "========================================"

  local -a ds_args=(
    segearth_r2/train/train.py
    --model_name_or_path "${MODEL_NAME_OR_PATH}"
    --vision_tower "${VISION_TOWER}"
    --vision_tower_mask "${VISION_TOWER_MASK}"
    --base_data_path "${BASE_DATA_PATH}"
    --dataset_name "${DATASET_NAME}"
    --output_dir "${output_dir}"
    --max_steps "${MAX_STEPS}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --save_strategy steps
    --save_steps "${SAVE_STEPS}"
    --save_total_limit "${SAVE_TOTAL_LIMIT}"
    --bf16 "${BF16}"
    --fp16 "${FP16}"
    --learning_rate "${LEARNING_RATE}"
    --weight_decay "${WEIGHT_DECAY}"
    --warmup_ratio "${WARMUP_RATIO}"
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
    --logging_steps "${LOGGING_STEPS}"
    --tf32 "${TF32}"
    --model_max_length "${MODEL_MAX_LENGTH}"
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
    --lora_r "${LORA_R}"
    --lora_alpha "${LORA_ALPHA}"
    --lora_dropout "${LORA_DROPOUT}"
    --deepspeed scripts/zero1.json
    --mask_config "${MASK_CONFIG}"
    --data_ratio "${DATA_RATIO}"
    --switch_bs "${SWITCH_BS}"
    --seed "${SEED}"
    --data_seed "${DATA_SEED}"
    --max_grad_norm "${MAX_GRAD_NORM}"
    --train_midstage_recalibration "${TRAIN_MIDSTAGE_RECALIBRATION}"
    --stage3_norm_only "${STAGE3_NORM_ONLY}"
    --use_attention_loss "${USE_ATTENTION_LOSS}"
    --use_midstage_gate_loss "${USE_MIDSTAGE_GATE_LOSS}"
    --midstage_gate_loss_weight "${MIDSTAGE_GATE_LOSS_WEIGHT}"
    --use_mstva "${USE_MSTVA}"
    --mstva_align_dim "${MSTVA_ALIGN_DIM}"
    --use_mstva_loss "${USE_MSTVA_LOSS}"
    --mstva_loss_weight "${MSTVA_LOSS_WEIGHT}"
    --mstva_scale_weights "${MSTVA_SCALE_WEIGHTS}"
  )

  if [[ "${use_v2}" -eq 1 ]]; then
    ds_args+=(--concept_public_semantic_library "${CONCEPT_LIB}")
  fi

  if [[ "${REPORT_TO}" == "wandb" ]]; then
    ds_args+=(--report_to wandb)
  fi

  deepspeed --master_port="${MASTER_PORT}" --include="${GPU_SLOT}" "${ds_args[@]}"
}

echo "========================================"
echo "[0] Preflight"
echo "[INFO] REPO_SOURCE=${REPO_SOURCE}"
echo "[INFO] REPO_TGI=${REPO_TGI}"
echo "========================================"

if [[ ! -d "${REPO_TGI}" ]]; then
  echo "[ERROR] REPO_TGI not found: ${REPO_TGI} (set REPO_TGI or place repos under ${RESEG_ROOT})"
  exit 1
fi
if [[ ! -f "${CONCEPT_LIB}" ]]; then
  echo "[ERROR] CONCEPT_LIB not found: ${CONCEPT_LIB}"
  exit 1
fi
if [[ ! -d "${MODEL_NAME_OR_PATH}" ]]; then
  echo "[ERROR] model path not found: ${MODEL_NAME_OR_PATH}"
  exit 1
fi
if [[ ! -d "${VISION_TOWER}" ]]; then
  echo "[ERROR] vision tower not found: ${VISION_TOWER}"
  exit 1
fi
if [[ ! -f "${VISION_TOWER_MASK}" ]]; then
  echo "[ERROR] vision tower mask not found: ${VISION_TOWER_MASK}"
  exit 1
fi
if [[ ! -f "${REPO_TGI}/${MASK_CONFIG}" ]]; then
  echo "[ERROR] mask config not found: ${REPO_TGI}/${MASK_CONFIG}"
  exit 1
fi
if [[ ! -f "${BASE_DATA_PATH}/rrsisd/refs(unc).p" ]]; then
  echo "[ERROR] refs file not found: ${BASE_DATA_PATH}/rrsisd/refs(unc).p"
  exit 1
fi
if [[ ! -f "${BASE_DATA_PATH}/rrsisd/instances.json" ]]; then
  echo "[ERROR] instances file not found: ${BASE_DATA_PATH}/rrsisd/instances.json"
  exit 1
fi
if [[ ! -d "${BASE_DATA_PATH}/images" ]]; then
  echo "[ERROR] images dir not found: ${BASE_DATA_PATH}/images"
  exit 1
fi

if ! "${PYTHON_BIN}" segearth_r2/train/train.py --help 2>/dev/null | grep -q "concept_public_semantic_library"; then
  echo "[ERROR] train.py --help failed or missing --concept_public_semantic_library (try PYTHON_BIN=conda run -n reseg python)"
  exit 1
fi

echo "[OK] preflight passed"

if [[ "${ARM}" == "both" ]]; then
  run_one_arm baseline
  run_one_arm v2
else
  run_one_arm "${ARM}"
fi

echo "[DONE] phase=${PHASE} arm=${ARM}"
