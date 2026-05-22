#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-1}"
PROJECT_ROOT="/home/wangchengjun/huangziyi/reseg/resegearth+prompt"
OUTPUT_ROOT="/home/wangchengjun/huangziyi/reseg/output/prompt"

BASE_DATA_PATH="/home/wangchengjun/huangziyi/data/RRSISD"
VISION_TOWER="/home/wangchengjun/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384"
VISION_TOWER_MASK="/home/wangchengjun/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl"
MASK_CONFIG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"

MERGE_PY="${PROJECT_ROOT}/segearth_r2/train/merge_lora_weights_and_save_hf_model.py"
EVAL_PY="${PROJECT_ROOT}/segearth_r2/eval/eval.py"

LOG_DIR="${OUTPUT_ROOT}/logs_merge_eval"
mkdir -p "${LOG_DIR}"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

RUNS=(
  "rrsisd_baseline"
  "rrsisd_semkb_cat"
  "rrsisd_semkb_cat_rel"
  "rrsisd_semkb_cat_rel_ctx"
)

find_latest_checkpoint() {
  local run_dir="$1"
  local latest_ckpt

  latest_ckpt=$(
    find "${run_dir}" -maxdepth 1 -type d -name "checkpoint-*" | \
    sed 's#.*/checkpoint-##' | sort -n | tail -n 1
  )

  if [[ -z "${latest_ckpt}" ]]; then
    return 1
  fi

  echo "${run_dir}/checkpoint-${latest_ckpt}"
}

process_one_run() {
  local run_name="$1"
  local run_dir="${OUTPUT_ROOT}/${run_name}"

  echo "============================================================"
  echo "Processing run: ${run_name}"
  echo "Run dir: ${run_dir}"
  echo "============================================================"

  if [[ ! -d "${run_dir}" ]]; then
    echo "[ERROR] Run directory not found: ${run_dir}"
    return 1
  fi

  local ckpt_dir
  ckpt_dir="$(find_latest_checkpoint "${run_dir}")" || {
    echo "[ERROR] No checkpoint-* found under ${run_dir}"
    return 1
  }

  local merged_dir="${ckpt_dir}/merged_model"
  local eval_dir="${ckpt_dir}/val_results"

  echo "[INFO] Latest checkpoint: ${ckpt_dir}"
  echo "[INFO] Merge output: ${merged_dir}"
  echo "[INFO] Eval output: ${eval_dir}"

  cd "${PROJECT_ROOT}"

  if [[ ! -d "${merged_dir}" ]]; then
    echo "[INFO] Merging LoRA weights ..."
    python "${MERGE_PY}" \
      --model_path "${ckpt_dir}" \
      --vision_tower "${VISION_TOWER}" \
      --vision_tower_mask "${VISION_TOWER_MASK}" \
      --mask_config "${MASK_CONFIG}" \
      --save_path "${merged_dir}" \
      --lora_r 4 \
      2>&1 | tee "${LOG_DIR}/${run_name}_merge.log"
  else
    echo "[INFO] merged_model already exists, skip merge: ${merged_dir}"
  fi

  echo "[INFO] Running validation ..."
  python "${EVAL_PY}" \
    --base_data_path "${BASE_DATA_PATH}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --mask_config "${MASK_CONFIG}" \
    --model_path "${merged_dir}" \
    --output_dir "${eval_dir}" \
    --dataset_name rrsisd \
    --split val \
    --eval_batch_size 1 \
    --zip_results False \
    2>&1 | tee "${LOG_DIR}/${run_name}_val.log"

  echo "[INFO] Finished: ${run_name}"
  echo
}

for run_name in "${RUNS[@]}"; do
  process_one_run "${run_name}"
done

echo "All merge + val jobs finished."