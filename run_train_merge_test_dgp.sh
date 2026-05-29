#!/usr/bin/env bash
set -euo pipefail

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

REPO_DIR="/root/rivermind-data/huangziyi/reseg/segearth+DGP"
cd "${REPO_DIR}"

GPU_ID="0"
MODEL_NAME_OR_PATH="/root/rivermind-data/huangziyi/reseg/pretrained_model/mllm/Mipha-3B"
VISION_TOWER="/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384"
VISION_TOWER_MASK="/root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl"
MASK_CONFIG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
BASE_DATA_PATH="/root/rivermind-data/huangziyi/data/RRSISD"
OUTPUT_DIR="/root/rivermind-data/huangziyi/reseg/output/dgp/stage3-smoke-v2"
MERGED_DIR="${OUTPUT_DIR}/merged_model"
EVAL_OUT="${OUTPUT_DIR}/eval_smoke"

echo "[1/3] module shape + forward smoke"
python tools/smoke_dgp_stage3.py

echo "[2/3] minimal train (2 steps)"
bash scripts/train_dgp_stage3.sh

CKPT=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1)
if [[ -z "${CKPT}" ]]; then
  echo "[ERROR] no checkpoint under ${OUTPUT_DIR}"
  exit 1
fi

echo "[3/3] merge + eval smoke from ${CKPT}"
rm -rf "${MERGED_DIR}" "${EVAL_OUT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" python segearth_r2/train/merge_lora_weights_and_save_hf_model.py \
  --model_path "${CKPT}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --mask_config "${MASK_CONFIG}" \
  --save_path "${MERGED_DIR}" \
  --use_dgp_qdti True \
  --lora_r 4

CUDA_VISIBLE_DEVICES="${GPU_ID}" python segearth_r2/eval/eval.py \
  --base_data_path "${BASE_DATA_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --mask_config "${MASK_CONFIG}" \
  --model_path "${MERGED_DIR}" \
  --output_dir "${EVAL_OUT}" \
  --dataset_name rrsisd \
  --split test \
  --eval_batch_size 1 \
  --max_eval_samples 1 \
  --use_dgp_qdti True \
  --zip_results False

echo "[OK] DGP Stage3 smoke pipeline finished"
