#!/usr/bin/env bash
# DGP v6.1 Stage A: train only prompt_adapter + query_refiner from SegEarth baseline merged model.
set -euo pipefail

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

REPO_DIR="/root/rivermind-data/huangziyi/reseg/segearth+DGP"
cd "${REPO_DIR}"

PYTHON="${PYTHON:-/root/rivermind-data/miniconda3/envs/reseg/bin/python}"
DEEPSPEED="${DEEPSPEED:-/root/rivermind-data/miniconda3/envs/reseg/bin/deepspeed}"

BASE_MERGED="${BASE_MERGED:-/root/rivermind-data/huangziyi/reseg/output/base/standard-base-siglip1-28w-gd4/merged_model}"
VISION_TOWER="/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384"
VISION_TOWER_MASK="/root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl"
MASK_CONFIG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
BASE_DATA_PATH="/root/rivermind-data/huangziyi/data/RRSISD"
OUTPUT_DIR="${OUTPUT_DIR:-/root/rivermind-data/huangziyi/reseg/output/dgp/v6.1-stage-a-from-base}"

MAX_STEPS="${MAX_STEPS:-500}"
BATCH="${BATCH:-4}"
GRAD_ACC="${GRAD_ACC:-1}"

if [[ ! -f "${BASE_MERGED}/config.json" ]]; then
  echo "[ERROR] baseline merged model not found: ${BASE_MERGED}"
  exit 1
fi

echo "[train] model=${BASE_MERGED}"
echo "[train] output=${OUTPUT_DIR} max_steps=${MAX_STEPS} batch=${BATCH} grad_acc=${GRAD_ACC}"

"${DEEPSPEED}" --master_port=29512 --include localhost:0 segearth_r2/train/train.py \
  --model_name_or_path "${BASE_MERGED}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --base_data_path "${BASE_DATA_PATH}" \
  --dataset_name rrsisd \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${BATCH}" \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --save_strategy steps \
  --save_steps 250 \
  --save_total_limit 3 \
  --bf16 True \
  --learning_rate 1e-4 \
  --logging_steps 10 \
  --model_max_length 2048 \
  --dataloader_num_workers 2 \
  --deepspeed scripts/zero1.json \
  --mask_config "${MASK_CONFIG}" \
  --data_ratio 1 \
  --switch_bs 1 \
  --use_dgp_qdti True \
  --use_qdti_bias False \
  --dgp_training_stage a \
  --gate_g_init 0.01 \
  --gate_l_init 0.02 \
  --dgp_monitor_wandb False \
  --evaluation_strategy no \
  --report_to none \
  --scale_hard_loss_weight 0.0 \
  --lora_enable False
