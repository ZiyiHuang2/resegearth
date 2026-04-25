#!/usr/bin/env bash
set -euo pipefail

########################################
# Environment
########################################
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_PROJECT=segearth-tgi
export WANDB_INIT_TIMEOUT=300
unset CUDA_VISIBLE_DEVICES

GPU_SLOT="localhost:1"
GPU_ID="1"

MASTER_PORT_MIDSTAGE="29511"

########################################
# Project dir
########################################
REPO_DIR="/home/wangchengjun/huangziyi/reseg/resegearth+tgi"
cd "${REPO_DIR}"

########################################
# Common paths
########################################
MODEL_NAME_OR_PATH="/home/wangchengjun/huangziyi/reseg/output/bseg/baseline_standard-base_5w/merged_model"

VISION_TOWER="/home/wangchengjun/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384"
VISION_TOWER_MASK="/home/wangchengjun/huangziyi/reseg/pretrained_model/mask2former/maskformer2_swin_base_IN21k_384_bs16_50ep.pkl"
MASK_CONFIG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"

########################################
# Dataset config
########################################
BASE_DATA_PATH="/home/wangchengjun/huangziyi/data/RRSISD"
DATASET_NAME="rrsisd"
TEST_SPLIT="test"

########################################
# Train config
########################################
MAX_STEPS="50000"
PER_DEVICE_TRAIN_BATCH_SIZE="1"
GRADIENT_ACCUMULATION_STEPS="1"

SAVE_STEPS="1000"
SAVE_TOTAL_LIMIT="2"

# 从已收敛 baseline 继续训练，并解冻 Stage3 局部参数，先用稳一点的 lr
LEARNING_RATE="3e-5"
WEIGHT_DECAY="0.0"
WARMUP_RATIO="0.03"
LR_SCHEDULER_TYPE="cosine"

LOGGING_STEPS="10"

# bf16 曾触发 Swin 初始化 erfinv_cuda 问题；这里使用 fp16
BF16="False"
FP16="True"
TF32="False"

MODEL_MAX_LENGTH="2048"
GRADIENT_CHECKPOINTING="False"
DATALOADER_NUM_WORKERS="4"

# 主方案用 r=8，不再跟 debug 脚本的 r=4
LORA_R="8"
LORA_ALPHA="16"
LORA_DROPOUT="0.05"

DATA_RATIO="1"
SWITCH_BS="4"

SEED="42"
DATA_SEED="42"
MAX_GRAD_NORM="1.0"

########################################
# Mid-stage text recalibration config
########################################
TRAIN_MIDSTAGE_RECALIBRATION="True"
STAGE3_NORM_ONLY="False"

########################################
# Attention loss config
########################################
# 之前训练出现 loss_attention=inf，导致 fp16 loss scale 降到最低退出。
# 这里需要 train.py 已支持 --use_attention_loss。
USE_ATTENTION_LOSS="False"

########################################
# Output
########################################
MIDSTAGE_OUTPUT_DIR="/home/wangchengjun/huangziyi/reseg/output/tgi/midstage_text_recalib_stage3_lora8_lr3e5_attn_off_5k"
MIDSTAGE_MERGED_DIR="${MIDSTAGE_OUTPUT_DIR}/merged_model"
MIDSTAGE_TEST_OUTPUT_DIR="${MIDSTAGE_OUTPUT_DIR}/test_results"

########################################
# Helpers
########################################
read_best_checkpoint () {
  local out_dir="$1"
  OUTPUT_DIR="${out_dir}" python - <<'PY'
import json
import os
import sys

output_dir = os.environ["OUTPUT_DIR"]
trainer_state = os.path.join(output_dir, "trainer_state.json")

if not os.path.exists(trainer_state):
    print("")
    sys.exit(0)

with open(trainer_state, "r", encoding="utf-8") as f:
    state = json.load(f)

print(state.get("best_model_checkpoint", ""))
PY
}

read_last_checkpoint () {
  local out_dir="$1"
  OUTPUT_DIR="${out_dir}" python - <<'PY'
import os
import re

output_dir = os.environ["OUTPUT_DIR"]
if not os.path.isdir(output_dir):
    print("")
    raise SystemExit

candidates = []
for name in os.listdir(output_dir):
    m = re.match(r"checkpoint-(\d+)$", name)
    if m:
        candidates.append((int(m.group(1)), os.path.join(output_dir, name)))

if not candidates:
    print("")
else:
    candidates.sort()
    print(candidates[-1][1])
PY
}

merge_ckpt () {
  local ckpt="$1"
  local save_dir="$2"

  rm -rf "${save_dir}"
  mkdir -p "${save_dir}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" python segearth_r2/train/merge_lora_weights_and_save_hf_model.py \
    --model_path "${ckpt}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --mask_config "${MASK_CONFIG}" \
    --save_path "${save_dir}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}"
}

eval_model () {
  local model_dir="$1"
  local out_dir="$2"

  mkdir -p "${out_dir}"

  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  python segearth_r2/eval/eval.py \
    --base_data_path "${BASE_DATA_PATH}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --mask_config "${MASK_CONFIG}" \
    --model_path "${model_dir}" \
    --output_dir "${out_dir}" \
    --dataset_name "${DATASET_NAME}" \
    --split "${TEST_SPLIT}" \
    --eval_batch_size 1 \
    --zip_results False
}

########################################
# Preflight
########################################
echo "========================================"
echo "[0/5] Preflight checks"
echo "========================================"

echo "[INFO] REPO_DIR=${REPO_DIR}"
echo "[INFO] MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "[INFO] VISION_TOWER=${VISION_TOWER}"
echo "[INFO] VISION_TOWER_MASK=${VISION_TOWER_MASK}"
echo "[INFO] MASK_CONFIG=${MASK_CONFIG}"
echo "[INFO] BASE_DATA_PATH=${BASE_DATA_PATH}"
echo "[INFO] OUTPUT_DIR=${MIDSTAGE_OUTPUT_DIR}"
echo "[INFO] GPU_SLOT=${GPU_SLOT}"
echo "[INFO] LORA_R=${LORA_R}"
echo "[INFO] LEARNING_RATE=${LEARNING_RATE}"
echo "[INFO] USE_ATTENTION_LOSS=${USE_ATTENTION_LOSS}"

if [[ ! -d "${MODEL_NAME_OR_PATH}" ]]; then
  echo "[ERROR] model path not found: ${MODEL_NAME_OR_PATH}"
  exit 1
fi

if [[ ! -d "${VISION_TOWER}" ]]; then
  echo "[ERROR] vision tower not found: ${VISION_TOWER}"
  exit 1
fi

if [[ ! -f "${VISION_TOWER_MASK}" ]]; then
  echo "[ERROR] vision tower mask not found: ${VISION_TOWER_MASK}"
  exit 1
fi

if [[ ! -f "${MASK_CONFIG}" ]]; then
  echo "[ERROR] mask config not found: ${MASK_CONFIG}"
  exit 1
fi

if [[ ! -f "${BASE_DATA_PATH}/rrsisd/refs(unc).p" ]]; then
  echo "[ERROR] refs file not found: ${BASE_DATA_PATH}/rrsisd/refs(unc).p"
  exit 1
fi

if [[ ! -f "${BASE_DATA_PATH}/rrsisd/instances.json" ]]; then
  echo "[ERROR] instances file not found: ${BASE_DATA_PATH}/rrsisd/instances.json"
  exit 1
fi

if [[ ! -d "${BASE_DATA_PATH}/images" ]]; then
  echo "[ERROR] images dir not found: ${BASE_DATA_PATH}/images"
  exit 1
fi

if ! python segearth_r2/train/train.py --help 2>/dev/null | grep -q "use_attention_loss"; then
  echo "[ERROR] train.py does not expose --use_attention_loss."
  echo "        Please apply the code patch that adds use_attention_loss to HfArgumentParser first."
  exit 1
fi

mkdir -p "${MIDSTAGE_OUTPUT_DIR}"

echo "[OK] preflight passed"

########################################
# 1) Train main scheme
########################################
echo "========================================"
echo "[1/5] Training mid-stage text recalibration main scheme"
echo "========================================"

export WANDB_NAME="tgi-midstage-stage3-lora8-lr3e5-attn-off-5k"

deepspeed --master_port="${MASTER_PORT_MIDSTAGE}" --include="${GPU_SLOT}" segearth_r2/train/train.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --base_data_path "${BASE_DATA_PATH}" \
  --dataset_name "${DATASET_NAME}" \
  --output_dir "${MIDSTAGE_OUTPUT_DIR}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --bf16 "${BF16}" \
  --fp16 "${FP16}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
  --logging_steps "${LOGGING_STEPS}" \
  --tf32 "${TF32}" \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --lora_r "${LORA_R}" \
  --deepspeed scripts/zero1.json \
  --mask_config "${MASK_CONFIG}" \
  --data_ratio "${DATA_RATIO}" \
  --switch_bs "${SWITCH_BS}" \
  --seed "${SEED}" \
  --data_seed "${DATA_SEED}" \
  --max_grad_norm "${MAX_GRAD_NORM}" \
  --train_midstage_recalibration "${TRAIN_MIDSTAGE_RECALIBRATION}" \
  --stage3_norm_only "${STAGE3_NORM_ONLY}" \
  --use_attention_loss "${USE_ATTENTION_LOSS}" \
  --report_to wandb

########################################
# 2) Select checkpoint
########################################
echo "========================================"
echo "[2/5] Select checkpoint"
echo "========================================"

MIDSTAGE_BEST_CHECKPOINT=$(read_best_checkpoint "${MIDSTAGE_OUTPUT_DIR}")

if [[ -z "${MIDSTAGE_BEST_CHECKPOINT}" ]]; then
  echo "[WARN] best_model_checkpoint not found; fallback to last checkpoint"
  MIDSTAGE_BEST_CHECKPOINT=$(read_last_checkpoint "${MIDSTAGE_OUTPUT_DIR}")
fi

if [[ -z "${MIDSTAGE_BEST_CHECKPOINT}" ]]; then
  echo "[ERROR] no checkpoint found under ${MIDSTAGE_OUTPUT_DIR}"
  exit 1
fi

echo "[OK] MIDSTAGE_CHECKPOINT=${MIDSTAGE_BEST_CHECKPOINT}"

########################################
# 3) Merge
########################################
echo "========================================"
echo "[3/5] Merge mid-stage checkpoint"
echo "========================================"

merge_ckpt "${MIDSTAGE_BEST_CHECKPOINT}" "${MIDSTAGE_MERGED_DIR}"

########################################
# 4) Eval
########################################
echo "========================================"
echo "[4/5] Eval merged model"
echo "========================================"

eval_model "${MIDSTAGE_MERGED_DIR}" "${MIDSTAGE_TEST_OUTPUT_DIR}"

########################################
# 5) Done
########################################
echo "========================================"
echo "[5/5] DONE"
echo "Midstage output      : ${MIDSTAGE_OUTPUT_DIR}"
echo "Midstage checkpoint  : ${MIDSTAGE_BEST_CHECKPOINT}"
echo "Midstage merged      : ${MIDSTAGE_MERGED_DIR}"
echo "Midstage test output : ${MIDSTAGE_TEST_OUTPUT_DIR}"
echo "========================================"