#!/usr/bin/env bash
# SPIM-v1 1-step sanity: three runs (alpha 0 / 0.001 / 0.05).
# Prereq: conda activate reseg (or any env with torch, transformers, deepspeed).
# Usage:
#   cd /path/to/segearth+att
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate reseg
#   bash tools/debug_spim_sanity.sh
#
# Logs: tools/spim_sanity_logs/t{1,2,3}_*.log

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
LOGDIR="${ROOT}/tools/spim_sanity_logs"
mkdir -p "${LOGDIR}"

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

# Default GPU 1: GPU 0 is often busy on shared machines (override with GPU_SLOT=localhost:0).
GPU_SLOT="${GPU_SLOT:-localhost:1}"
MASTER_PORT="${MASTER_PORT:-29610}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/wangchengjun/huangziyi/reseg/pretrained_model/mllm/Mipha-3B}"
VISION_TOWER="${VISION_TOWER:-/home/wangchengjun/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384}"
VISION_TOWER_MASK="${VISION_TOWER_MASK:-/home/wangchengjun/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl}"
MASK_CONFIG="${MASK_CONFIG:-segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml}"
BASE_DATA_PATH="${BASE_DATA_PATH:-/home/wangchengjun/huangziyi/data/RRSISD}"
DATASET_NAME="${DATASET_NAME:-rrsisd}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/tools/spim_sanity_runs}"

SANITY_MAX_STEPS="${SANITY_MAX_STEPS:-1}"
SANITY_SAVE_STEPS="${SANITY_SAVE_STEPS:-999999}"
SANITY_LOGGING_STEPS="${SANITY_LOGGING_STEPS:-1}"
# Must be >1 if TrainingArguments keeps default dataloader_prefetch_factor (train.py default=2).
SANITY_DATALOADER_NUM_WORKERS="${SANITY_DATALOADER_NUM_WORKERS:-2}"
SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-42}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
BF16="${BF16:-True}"

run_one() {
  local name="$1"
  local alpha="$2"
  local out="${OUTPUT_ROOT}/${name}"
  local log="${LOGDIR}/${name}.log"
  rm -rf "${out}"
  mkdir -p "${out}"
  echo "==== Running ${name} spim_alpha=${alpha} log=${log} ===="
  # shellcheck disable=SC2094
  deepspeed --master_port="${MASTER_PORT}" --include="${GPU_SLOT}" segearth_r2/train/train.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --base_data_path "${BASE_DATA_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --output_dir "${out}" \
    --max_steps "${SANITY_MAX_STEPS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --save_strategy steps \
    --save_steps "${SANITY_SAVE_STEPS}" \
    --save_total_limit 1 \
    --bf16 "${BF16}" \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps "${SANITY_LOGGING_STEPS}" \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers "${SANITY_DATALOADER_NUM_WORKERS}" \
    --lora_r 8 \
    --deepspeed scripts/zero1.json \
    --mask_config "${MASK_CONFIG}" \
    --data_ratio 1 \
    --switch_bs 4 \
    --seed "${SEED}" \
    --data_seed "${DATA_SEED}" \
    --report_to none \
    --use_spim True \
    --spim_alpha "${alpha}" \
    --spim_layer_idx -1 \
    --spim_detach True \
    --spim_norm True \
    --spim_seg_agg mean \
    --spim_near_zero_eps 1e-8 \
    --spim_debug True \
    2>&1 | tee "${log}"
}

echo "[0] CLI check (requires same env as training)"
python segearth_r2/train/train.py --help 2>&1 | grep -E 'spim|use_spim' || true

run_one "t1_alpha0" "0.0"
run_one "t2_alpha0001" "0.001"
run_one "t3_alpha005" "0.05"

echo "==== Done. Logs under ${LOGDIR} ===="
