#!/usr/bin/env bash
set -euo pipefail

########################################
# Mode
#   SPIM_SANITY_ONLY=1 (default): run three 1-step SPIM checks, then exit.
#   SPIM_SANITY_ONLY=0: one full SPIM-v1 train (alpha=0.05) + merge best ckpt + eval (long run). No baseline.
#
# SPIM sanity goals (no baseline run required):
#   1) SPIM flags reach model.config ([SPIM config]).
#   2) Prior path works (Test1: alpha=0 → no decoder bias path).
#   3) Decoder bias path works (Test2/3: alpha>0 → build_spim_bias + merge mask).
#   4) Compare first-step train loss: L_alpha0 vs L_alpha0001 vs L_alpha005.
#
# Recommended order (script runs in this order):
#   Test1 alpha=0.0   → L_alpha0    (prior only; no spatial_bias logs)
#   Test2 alpha=0.001 → L_alpha0001 (tiny bias; path + shape assert)
#   Test3 alpha=0.05  → L_alpha005  (visible loss delta expected)
#
# Record per run: [SPIM config], [SPIM debug] lines, first-step loss, NaN or not.
# Test1 will NOT print spatial_bias.shape (expected).
########################################

SPIM_SANITY_ONLY="${SPIM_SANITY_ONLY:-1}"

########################################
# Environment
########################################
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_PROJECT="${WANDB_PROJECT:-segearth-spim-sanity}"
export WANDB_INIT_TIMEOUT=300
unset CUDA_VISIBLE_DEVICES

GPU_SLOT="${GPU_SLOT:-localhost:2}"
GPU_ID="${GPU_ID:-2}"
MASTER_PORT="${MASTER_PORT:-21111}"

########################################
# Project dir (this repo: segearth+att)
########################################
REPO_DIR="${REPO_DIR:-/home/wangchengjun/huangziyi/reseg/segearth+att}"
cd "${REPO_DIR}"

########################################
# Common paths
########################################
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/wangchengjun/huangziyi/reseg/pretrained_model/mllm/Mipha-3B}"
VISION_TOWER="${VISION_TOWER:-/home/wangchengjun/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384}"
VISION_TOWER_MASK="${VISION_TOWER_MASK:-/home/wangchengjun/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl}"
MASK_CONFIG="${MASK_CONFIG:-segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml}"

########################################
# Dataset config
########################################
BASE_DATA_PATH="${BASE_DATA_PATH:-/home/wangchengjun/huangziyi/data/RRSISD}"
DATASET_NAME="${DATASET_NAME:-rrsisd}"
TEST_SPLIT="${TEST_SPLIT:-test}"

########################################
# Output roots
########################################
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/wangchengjun/huangziyi/reseg/output/segearth+att-spim-sanity}"
# Full training (SPIM_SANITY_ONLY=0): SPIM-v1 only, alpha=0.05 — no baseline / no alpha=0 full run in this script.
OUTPUT_DIR="${OUTPUT_DIR:-/home/wangchengjun/huangziyi/reseg/output/segearth+att-spim-alpha005}"
MERGED_DIR="${OUTPUT_DIR}/merged_model"
TEST_OUTPUT_DIR="${OUTPUT_DIR}/test_results"

########################################
# Train config (full pipeline)
########################################
MAX_STEPS="${MAX_STEPS:-280000}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"

SAVE_STEPS="${SAVE_STEPS:-2000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"

LOGGING_STEPS="${LOGGING_STEPS:-10}"
BF16="${BF16:-True}"
TF32="${TF32:-False}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-False}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"

LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

DATA_RATIO="${DATA_RATIO:-1}"
SWITCH_BS="${SWITCH_BS:-4}"

# Keep identical across the three SPIM sanity runs for comparable first-step loss.
SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-42}"

########################################
# SPIM sanity: shared knobs (1 step each)
########################################
SANITY_MAX_STEPS="${SANITY_MAX_STEPS:-1}"
SANITY_SAVE_STEPS="${SANITY_SAVE_STEPS:-999999}"
SANITY_LOGGING_STEPS="${SANITY_LOGGING_STEPS:-1}"
# Lower worker count improves step-to-step loss comparability (optional override).
SANITY_DATALOADER_NUM_WORKERS="${SANITY_DATALOADER_NUM_WORKERS:-4}"

run_spim_sanity_one() {
  local tag="$1"
  local spim_alpha="$2"
  local out_dir="${OUTPUT_ROOT}/${tag}"
  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"
  export WANDB_NAME="${tag}"

  echo "========================================"
  echo "[SPIM sanity] ${tag}  spim_alpha=${spim_alpha}"
  echo "  output_dir=${out_dir}"
  echo "  max_steps=${SANITY_MAX_STEPS}  seed=${SEED}  data_seed=${DATA_SEED}"
  echo "========================================"

  deepspeed --master_port="${MASTER_PORT}" --include="${GPU_SLOT}" segearth_r2/train/train.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --base_data_path "${BASE_DATA_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --output_dir "${out_dir}" \
    --max_steps "${SANITY_MAX_STEPS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --save_strategy steps \
    --save_steps "${SANITY_SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --bf16 "${BF16}" \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
    --logging_steps "${SANITY_LOGGING_STEPS}" \
    --tf32 "${TF32}" \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
    --dataloader_num_workers "${SANITY_DATALOADER_NUM_WORKERS}" \
    --lora_r "${LORA_R}" \
    --deepspeed scripts/zero1.json \
    --mask_config "${MASK_CONFIG}" \
    --data_ratio "${DATA_RATIO}" \
    --switch_bs "${SWITCH_BS}" \
    --seed "${SEED}" \
    --data_seed "${DATA_SEED}" \
    --report_to none \
    --use_spim True \
    --spim_alpha "${spim_alpha}" \
    --spim_layer_idx -1 \
    --spim_detach True \
    --spim_norm True \
    --spim_seg_agg mean \
    --spim_near_zero_eps 1e-8 \
    --spim_debug True
}

########################################
# SPIM_SANITY_ONLY=1  → three 1-step runs
########################################
if [[ "${SPIM_SANITY_ONLY}" == "1" ]]; then
  echo ""
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  echo " SPIM_SANITY_ONLY=1"
  echo "  Test1: alpha=0.0   (prior path only; no decoder bias)"
  echo "  Test2: alpha=0.001 (decoder path + merge + shape assert)"
  echo "  Test3: alpha=0.05  (stronger bias; expect loss vs Test1)"
  echo "  Logs: [SPIM config], [SPIM debug], first-step train loss"
  echo "  Note: Test1 will NOT show spatial_bias.shape (expected)."
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  echo ""

  run_spim_sanity_one "t1_alpha0" "0.0"
  run_spim_sanity_one "t2_alpha0001" "0.001"
  run_spim_sanity_one "t3_alpha005" "0.05"

  echo ""
  echo "========================================"
  echo "[SPIM sanity DONE]"
  echo "  Compare first-step loss in each run log:"
  echo "    L_alpha0     ← ${OUTPUT_ROOT}/t1_alpha0"
  echo "    L_alpha0001  ← ${OUTPUT_ROOT}/t2_alpha0001"
  echo "    L_alpha005   ← ${OUTPUT_ROOT}/t3_alpha005"
  echo "  Full train+merge+eval: SPIM_SANITY_ONLY=0 bash $0"
  echo "========================================"
  exit 0
fi

########################################
# 1) Full training (SPIM_SANITY_ONLY=0)
#     Single run: SPIM-v1 with spim_alpha=0.05 (requires outputs.attentions; keep gradient_checkpointing=False).
########################################
echo "========================================"
echo "[1/4] Training (SPIM-v1 alpha=0.05; val + best ckpt)"
echo "========================================"

export WANDB_NAME="${WANDB_NAME:-spim-alpha005-layer-1}"
mkdir -p "${OUTPUT_DIR}"

deepspeed --master_port="${MASTER_PORT}" --include="${GPU_SLOT}" segearth_r2/train/train.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --base_data_path "${BASE_DATA_PATH}" \
  --dataset_name "${DATASET_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --bf16 "${BF16}" \
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
  --report_to wandb \
  --use_spim True \
  --spim_alpha 0.05 \
  --spim_layer_idx -1 \
  --spim_detach True \
  --spim_norm True \
  --spim_seg_agg mean \
  --spim_near_zero_eps 1e-8 \
  --spim_debug False

########################################
# 2) Read best checkpoint from trainer_state.json
########################################
echo "========================================"
echo "[2/4] Reading best checkpoint"
echo "========================================"

export OUTPUT_DIR
BEST_CHECKPOINT=$(python - <<'PY'
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
)

if [[ -z "${BEST_CHECKPOINT}" ]]; then
  echo "[ERROR] best_model_checkpoint not found in ${OUTPUT_DIR}/trainer_state.json"
  exit 1
fi

echo "[OK] BEST_CHECKPOINT=${BEST_CHECKPOINT}"

########################################
# 3) Merge best checkpoint only
########################################
echo "========================================"
echo "[3/4] Merging best checkpoint"
echo "========================================"

rm -rf "${MERGED_DIR}"
mkdir -p "${MERGED_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python segearth_r2/train/merge_lora_weights_and_save_hf_model.py \
  --model_path "${BEST_CHECKPOINT}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --mask_config "${MASK_CONFIG}" \
  --save_path "${MERGED_DIR}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}"

########################################
# 4) Test merged best model
########################################
echo "========================================"
echo "[4/4] Testing merged best model"
echo "========================================"

mkdir -p "${TEST_OUTPUT_DIR}"

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
python segearth_r2/eval/eval.py \
  --base_data_path "${BASE_DATA_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --mask_config "${MASK_CONFIG}" \
  --model_path "${MERGED_DIR}" \
  --output_dir "${TEST_OUTPUT_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --split "${TEST_SPLIT}" \
  --eval_batch_size 1 \
  --zip_results False

echo "========================================"
echo "[DONE]"
echo "Best checkpoint : ${BEST_CHECKPOINT}"
echo "Merged model    : ${MERGED_DIR}"
echo "Test outputs    : ${TEST_OUTPUT_DIR}"
echo "========================================"
