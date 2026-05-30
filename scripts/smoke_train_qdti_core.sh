#!/usr/bin/env bash
# Minimal smoke train: QDTI-Core only (few steps).
set -euo pipefail
RESEG_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${RESEG_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESEG_ROOT}/output/tgi/smoke_qdti_core_v26}"
mkdir -p "${OUTPUT_DIR}"

python segearth_r2/train/train.py \
  --model_name_or_path "${MODEL_PATH:-pretrained_model/mllm/Mipha-3B}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --logging_steps 1 \
  --save_steps 1000000 \
  --report_to none \
  --use_query_aware_decoder_bias True \
  --decoder_attn_bias_apply_layers last3 \
  --decoder_attn_bias_max_abs 0.02 \
  --use_qdti_mask_feedback False \
  --use_qdti_rank_loss False \
  --use_qdti_neg_loss False \
  --use_qdti_div_loss False \
  --use_mstva False \
  --use_text_film False \
  2>&1 | tee "${OUTPUT_DIR}/smoke_train.log"

echo "[PASS] smoke_train_qdti_core log: ${OUTPUT_DIR}/smoke_train.log"
