#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-1}"
MASTER_PORT="${MASTER_PORT:-29100}"
MAX_STEPS="${MAX_STEPS:-5000}"

PROJECT_ROOT="/home/wangchengjun/huangziyi/reseg/resegearth+prompt"
TRAIN_PY="${PROJECT_ROOT}/segearth_r2/train/train.py"

MODEL_PATH="/home/wangchengjun/huangziyi/reseg/pretrained_model/mllm/Mipha-3B"
VISION_TOWER="/home/wangchengjun/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384"
VISION_TOWER_MASK="/home/wangchengjun/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl"
DATA_PATH="/home/wangchengjun/huangziyi/data/RRSISD"

OUTPUT_ROOT="/home/wangchengjun/huangziyi/reseg/output/prompt"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_PROJECT=resegearth_prompt_rrsisd

run_exp() {
  local run_name="$1"
  local include_fields="$2"

  export WANDB_NAME="${run_name}"

  echo "============================================================"
  echo "Running: ${run_name}"
  echo "GPU: ${GPU_ID}"
  echo "MASTER_PORT: ${MASTER_PORT}"
  echo "include_fields: ${include_fields}"
  echo "============================================================"

  cd "${PROJECT_ROOT}"

  deepspeed \
    --master_port="${MASTER_PORT}" \
    --include="localhost:${GPU_ID}" \
    "${TRAIN_PY}" \
    --model_name_or_path "${MODEL_PATH}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --base_data_path "${DATA_PATH}" \
    --dataset_name rrsisd \
    --output_dir "${OUTPUT_ROOT}/${run_name}" \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 2 \
    --bf16 True \
    --learning_rate 5e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --lora_r 4 \
    --deepspeed scripts/zero1.json \
    --mask_config "segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml" \
    --data_ratio 1 \
    --switch_bs 4 \
    --use_static_kb False \
    --use_ss_kb False \
    --use_semantic_kb True \
    --semantic_kb_inject_mode hard \
    --semantic_kb_hard_query_max_tokens 7 \
    --semantic_kb_max_prefix_chars 72 \
    --semantic_kb_include_fields "${include_fields}" \
    --report_to wandb \
    2>&1 | tee "${LOG_DIR}/${run_name}.log"
}

# baseline 已经跑过，这里跳过
run_exp "rrsisd_semkb_cat" "cat"
run_exp "rrsisd_semkb_cat_rel" "cat,rel"
run_exp "rrsisd_semkb_cat_rel_ctx" "cat,rel,ctx"

echo "All semantic KB runs finished."