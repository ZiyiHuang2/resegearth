#!/usr/bin/env bash
# 50~100 step smoke：public_semantic_v2 + concept_match_strict + debug 打印（stderr）。
# 用法示例：conda run -n reseg bash scripts/run_public_semantic_v2_match_strict_smoke.sh
set -euo pipefail

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_PROJECT="${WANDB_PROJECT:-segearth-source}"
unset CUDA_VISIBLE_DEVICES

GPU_SLOT="${GPU_SLOT:-localhost:1}"
MASTER_PORT="${MASTER_PORT:-29502}"

REPO_DIR="/home/wangchengjun/huangziyi/reseg/resegearth+source"
RESEG_ROOT="/home/wangchengjun/huangziyi/reseg"
cd "${REPO_DIR}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/wangchengjun/huangziyi/reseg/output/bseg/baseline_standard-base_5w/merged_model}"
VISION_TOWER="${VISION_TOWER:-/home/wangchengjun/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384}"
VISION_TOWER_MASK="${VISION_TOWER_MASK:-/home/wangchengjun/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl}"
MASK_CONFIG="${MASK_CONFIG:-segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml}"
BASE_DATA_PATH="${BASE_DATA_PATH:-/home/wangchengjun/huangziyi/data/RRSISD}"
CONCEPT_PUBLIC_SEMANTIC_LIBRARY="${CONCEPT_PUBLIC_SEMANTIC_LIBRARY:-configs/concept_public_semantic_library_v2.json}"

MAX_STEPS="${MAX_STEPS:-80}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESEG_ROOT}/output/source/rrsisd_public_semantic_v2_match_strict_smoke}"
DEBUG_MAX="${DEBUG_CONCEPT_MATCH_STRICT_MAX_SAMPLES:-80}"

if ! python segearth_r2/train/train.py --help 2>&1 | grep -q "concept_match_strict"; then
  echo "[ERROR] train.py --help 未包含 concept_match_strict。"
  exit 1
fi

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

deepspeed --master_port="${MASTER_PORT}" --include="${GPU_SLOT}" segearth_r2/train/train.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --base_data_path "${BASE_DATA_PATH}" \
  --dataset_name rrsisd \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --save_strategy steps \
  --save_steps 1000000 \
  --save_total_limit 1 \
  --bf16 True \
  --learning_rate 1e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --tf32 False \
  --model_max_length 2048 \
  --gradient_checkpointing False \
  --dataloader_num_workers 2 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --deepspeed scripts/zero1.json \
  --mask_config "${MASK_CONFIG}" \
  --data_ratio 1 \
  --switch_bs 4 \
  --seed 42 \
  --data_seed 42 \
  --report_to none \
  --concept_public_semantic_library "${CONCEPT_PUBLIC_SEMANTIC_LIBRARY}" \
  --concept_match_strict True \
  --debug_concept_match_strict True \
  --debug_concept_match_strict_max_samples "${DEBUG_MAX}"

echo "[OK] smoke done. Check stderr above for [RRSISD][debug_concept_match_strict] lines."
echo "[OK] OUTPUT_DIR=${OUTPUT_DIR}"
