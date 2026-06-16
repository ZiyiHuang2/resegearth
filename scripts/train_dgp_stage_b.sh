#!/usr/bin/env bash
set -euo pipefail

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

REPO_DIR="/root/rivermind-data/huangziyi/reseg/segearth+DGP"
cd "${REPO_DIR}"

MODEL_NAME_OR_PATH="/root/rivermind-data/huangziyi/reseg/pretrained_model/mllm/Mipha-3B"
STAGE_A_CHECKPOINT="/root/rivermind-data/huangziyi/reseg/output/dgp/v6.1-stage-a"
VISION_TOWER="/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384"
VISION_TOWER_MASK="/root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl"
MASK_CONFIG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
BASE_DATA_PATH="/root/rivermind-data/huangziyi/data/RRSISD"
OUTPUT_DIR="/root/rivermind-data/huangziyi/reseg/output/dgp/v6.1-stage-b"

deepspeed --master_port=29503 --include localhost:0 segearth_r2/train/train.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --base_data_path "${BASE_DATA_PATH}" \
  --dataset_name rrsisd \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps 500 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --save_strategy steps \
  --save_steps 250 \
  --save_total_limit 2 \
  --bf16 True \
  --learning_rate 1e-4 \
  --logging_steps 10 \
  --model_max_length 2048 \
  --dataloader_num_workers 2 \
  --lora_r 4 \
  --deepspeed scripts/zero1.json \
  --mask_config "${MASK_CONFIG}" \
  --data_ratio 1 \
  --switch_bs 1 \
  --use_dgp_qdti True \
  --use_qdti_bias True \
  --qdti_apply_layers last1 \
  --qdti_scale_init 1e-3 \
  --dgp_training_stage b \
  --dgp_stage_a_checkpoint "${STAGE_A_CHECKPOINT}" \
  --dgp_reset_optimizer True \
  --dgp_monitor_wandb True \
  --dgp_monitor_steps 10 \
  --report_to wandb \
  --scale_hard_loss_weight 0.0
