#!/usr/bin/env bash
# Minimal smoke train: frozen Seg Spatial Refiner only (1-2 steps).
set -euo pipefail
RESEG_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${RESEG_ROOT}"
export PYTHONPATH="detectron2:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESEG_ROOT}/output/tgi/smoke_seg_spatial_refiner_v1}"
mkdir -p "${OUTPUT_DIR}"

MODEL_PATH="${MODEL_PATH:-/root/rivermind-data/huangziyi/reseg/output/base/standard-base-siglip1-28w-gd4/merged_model}"
VISION_TOWER="${VISION_TOWER:-/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384}"
VISION_TOWER_MASK="${VISION_TOWER_MASK:-/root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/maskformer2_swin_base_IN21k_384_bs16_50ep.pkl}"
BASE_DATA_PATH="${BASE_DATA_PATH:-/root/rivermind-data/huangziyi/data/RRSISD}"
PYTHON="${PYTHON:-/root/rivermind-data/miniconda3/envs/reseg/bin/python}"

"${PYTHON}" segearth_r2/train/train.py \
  --model_name_or_path "${MODEL_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --base_data_path "${BASE_DATA_PATH}" \
  --dataset_name rrsisd \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --logging_steps 1 \
  --save_steps "${SAVE_STEPS:-1}" \
  --save_strategy steps \
  --save_total_limit 1 \
  --evaluation_strategy no \
  --load_best_model_at_end False \
  --report_to none \
  --use_seg_spatial_refiner True \
  --train_seg_spatial_refiner_only True \
  --seg_spatial_refiner_alpha 0.1 \
  --seg_spatial_refiner_loss_weight 0.1 \
  --seg_spatial_refiner_dice_weight 1.0 \
  --seg_spatial_refiner_bce_weight 1.0 \
  --use_query_aware_decoder_bias False \
  --use_decoder_attn_bias False \
  --use_mstva False \
  --use_mstva_loss False \
  --use_text_film False \
  --use_attention_loss False \
  --lora_enable False \
  --train_midstage_recalibration False \
  --deepspeed "" \
  2>&1 | tee "${OUTPUT_DIR}/smoke_train.log"

echo "[PASS] smoke_train_seg_spatial_refiner log: ${OUTPUT_DIR}/smoke_train.log"
