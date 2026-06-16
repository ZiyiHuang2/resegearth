#!/usr/bin/env bash
# DGP v6.1 Stage A 2k validation (resume from 500-step checkpoint).
set -euo pipefail

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-dgp-v61-stage-a-2k}"

REPO_DIR="/root/rivermind-data/huangziyi/reseg/segearth+DGP"
cd "${REPO_DIR}"

PYTHON="${PYTHON:-/root/rivermind-data/miniconda3/envs/reseg/bin/python}"
DEEPSPEED="${DEEPSPEED:-/root/rivermind-data/miniconda3/envs/reseg/bin/deepspeed}"

MODEL_NAME_OR_PATH="/root/rivermind-data/huangziyi/reseg/pretrained_model/mllm/Mipha-3B"
VISION_TOWER="/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384"
VISION_TOWER_MASK="/root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl"
MASK_CONFIG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
BASE_DATA_PATH="/root/rivermind-data/huangziyi/data/RRSISD"
OUTPUT_DIR="/root/rivermind-data/huangziyi/reseg/output/dgp/v6.1-stage-a-500diag-full"
LOG_FILE="${OUTPUT_DIR}/train_2k.log"
MASTER_PORT="${MASTER_PORT:-29508}"
GPU_SLOT="${GPU_SLOT:-localhost:0}"

PER_DEVICE_BS="${PER_DEVICE_BS:-4}"
GRAD_ACC="${GRAD_ACC:-1}"

mkdir -p "${OUTPUT_DIR}"

if [[ ! -d "${OUTPUT_DIR}/checkpoint-500" ]]; then
  echo "[FAIL] missing checkpoint-500 under ${OUTPUT_DIR}"
  exit 1
fi

echo "[2k] Resume from checkpoint-500 -> max_steps=2000"
echo "[2k] batch=${PER_DEVICE_BS} grad_acc=${GRAD_ACC} effective=$((PER_DEVICE_BS * GRAD_ACC))"
echo "[2k] save/eval every 500 steps; health log every 50 steps"

exec "${DEEPSPEED}" --master_port="${MASTER_PORT}" --include "${GPU_SLOT}" segearth_r2/train/train.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --base_data_path "${BASE_DATA_PATH}" \
  --dataset_name rrsisd \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps 2000 \
  --per_device_train_batch_size "${PER_DEVICE_BS}" \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 4 \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --bf16 True \
  --learning_rate 1e-4 \
  --logging_steps 50 \
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
  --dgp_monitor_wandb True \
  --dgp_monitor_steps 50 \
  --report_to wandb \
  --scale_hard_loss_weight 0.0 \
  2>&1 | tee -a "${LOG_FILE}"
