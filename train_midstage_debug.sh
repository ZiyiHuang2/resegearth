#!/usr/bin/env bash
set -e

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# =========================
# Mid-stage text recalibration debug training
# =========================

REPO_DIR="/home/wangchengjun/huangziyi/reseg/resegearth+tgi"
cd "${REPO_DIR}"

# 按你的实际环境修改 GPU
GPU_ID=1
MASTER_PORT=29501

# 路径配置
MODEL_PATH="/home/wangchengjun/huangziyi/reseg/output/bseg/baseline_standard-base_5w/merged_model"
VISION_TOWER="/home/wangchengjun/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384"
VISION_TOWER_MASK="/home/wangchengjun/huangziyi/reseg/pretrained_model/mask2former/maskformer2_swin_base_IN21k_384_bs16_50ep.pkl"
MASK_CONFIG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"

# 数据路径：这里你必须改成你当前训练数据路径
BASE_DATA_PATH="/home/wangchengjun/huangziyi/data/RRSISD"

# 输出路径
OUTPUT_DIR="/home/wangchengjun/huangziyi/reseg/output/tgi/midstage_text_recalib_debug_500step"

mkdir -p "${OUTPUT_DIR}"

deepspeed \
  --master_port=${MASTER_PORT} \
  --include localhost:${GPU_ID} \
  segearth_r2/train/train.py \
  --model_name_or_path "${MODEL_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --base_data_path "${BASE_DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --mask_config "${MASK_CONFIG}" \
  --max_steps 500 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 250 \
  --save_total_limit 2 \
  --learning_rate 5e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 5 \
  --bf16 False \
  --fp16 True \
  --tf32 False \
  --model_max_length 2048 \
  --gradient_checkpointing False \
  --dataloader_num_workers 4 \
  --lora_r 4 \
  --deepspeed scripts/zero1.json \
  --data_ratio "1" \
  --switch_bs 4 \
  --train_midstage_recalibration True \
  --stage3_norm_only False