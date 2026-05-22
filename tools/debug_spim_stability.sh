#!/usr/bin/env bash
# SPIM-v1 small-scale stability: Test S0 (alpha=0) vs Test S1 (alpha=0.05).
# Does NOT modify llava_phi / mask2former / train.py.
#
# Usage:
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate reseg
#   cd /path/to/segearth+att
#   # optional: STABILITY_MAX_STEPS=100 GPU_SLOT=localhost:1 bash tools/debug_spim_stability.sh
#   bash tools/debug_spim_stability.sh
#
# Logs: tools/spim_stability_logs/s0_*.log , s1_*.log
# Optional: MEM_LOG=s0_mem.csv — nvidia-smi sample log next to train log

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
LOGDIR="${ROOT}/tools/spim_stability_logs"
mkdir -p "${LOGDIR}"

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

GPU_SLOT="${GPU_SLOT:-localhost:1}"
MASTER_PORT="${MASTER_PORT:-29620}"

STABILITY_MAX_STEPS="${STABILITY_MAX_STEPS:-500}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/wangchengjun/huangziyi/reseg/pretrained_model/mllm/Mipha-3B}"
VISION_TOWER="${VISION_TOWER:-/home/wangchengjun/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384}"
VISION_TOWER_MASK="${VISION_TOWER_MASK:-/home/wangchengjun/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl}"
MASK_CONFIG="${MASK_CONFIG:-segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml}"
BASE_DATA_PATH="${BASE_DATA_PATH:-/home/wangchengjun/huangziyi/data/RRSISD}"
DATASET_NAME="${DATASET_NAME:-rrsisd}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/tools/spim_stability_runs}"

run_stability() {
  local name="$1"
  local alpha="$2"
  local out="${OUTPUT_ROOT}/${name}"
  local log="${LOGDIR}/${name}_steps${STABILITY_MAX_STEPS}.log"
  local memlog="${LOGDIR}/${name}_mem.csv"
  rm -rf "${out}"
  mkdir -p "${out}"

  # Parse physical GPU index from GPU_SLOT e.g. localhost:1 -> 1
  local phys="${GPU_SLOT##*:}"

  echo "==== ${name} spim_alpha=${alpha} max_steps=${STABILITY_MAX_STEPS} log=${log} ===="
  nvidia-smi -i "${phys}" --query-gpu=timestamp,memory.used --format=csv -l 2 > "${memlog}" &
  local mem_pid=$!

  set +e
  deepspeed --master_port="${MASTER_PORT}" --include="${GPU_SLOT}" segearth_r2/train/train.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --base_data_path "${BASE_DATA_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --output_dir "${out}" \
    --max_steps "${STABILITY_MAX_STEPS}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy steps \
    --save_steps 999999 \
    --save_total_limit 1 \
    --bf16 True \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 2 \
    --lora_r 8 \
    --deepspeed scripts/zero1.json \
    --mask_config "${MASK_CONFIG}" \
    --data_ratio 1 \
    --switch_bs 4 \
    --seed 42 \
    --data_seed 42 \
    --report_to none \
    --use_spim True \
    --spim_alpha "${alpha}" \
    --spim_layer_idx -1 \
    --spim_detach True \
    --spim_norm True \
    --spim_seg_agg mean \
    --spim_near_zero_eps 1e-8 \
    --spim_debug False \
    2>&1 | tee "${log}"
  local rc=$?
  set -e

  kill "${mem_pid}" 2>/dev/null || true
  wait "${mem_pid}" 2>/dev/null || true

  if [[ "${rc}" -ne 0 ]]; then
    echo "[ERROR] ${name} exited with ${rc}"
    return "${rc}"
  fi
  echo "==== ${name} done ===="
}

run_stability "s0_alpha0" "0.0"
run_stability "s1_alpha005" "0.05"

echo "All stability runs finished. Logs: ${LOGDIR}"
