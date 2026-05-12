#!/usr/bin/env bash
set -euo pipefail

########################################
# Environment
########################################
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_PROJECT=segearth-tgi
export WANDB_INIT_TIMEOUT=300

# 如果你机器连 W&B 不稳定，就保留 offline。
# W&B 支持 WANDB_MODE=offline，在无网络时把日志保存在本地，后续可再 sync。

unset CUDA_VISIBLE_DEVICES

########################################
# GPU config
########################################
GPU_SLOT="localhost:0"
GPU_ID="0"
MASTER_PORT_TEXT_FILM="29541"

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
MAX_STEPS="100000"
PER_DEVICE_TRAIN_BATCH_SIZE="1"
GRADIENT_ACCUMULATION_STEPS="1"

SAVE_STEPS="2000"
SAVE_TOTAL_LIMIT="2"

LEARNING_RATE="3e-5"
WEIGHT_DECAY="0.0"
WARMUP_RATIO="0.03"
LR_SCHEDULER_TYPE="cosine"

LOGGING_STEPS="10"

BF16="False"
FP16="True"
TF32="False"

MODEL_MAX_LENGTH="2048"
GRADIENT_CHECKPOINTING="False"
DATALOADER_NUM_WORKERS="4"

LORA_R="8"
LORA_ALPHA="16"
LORA_DROPOUT="0.05"

DATA_RATIO="1"
SWITCH_BS="4"

SEED="42"
DATA_SEED="42"
MAX_GRAD_NORM="1.0"

########################################
# Disable old modules / losses
########################################
TRAIN_MIDSTAGE_RECALIBRATION="False"
STAGE3_NORM_ONLY="False"

USE_ATTENTION_LOSS="False"

USE_MIDSTAGE_GATE_LOSS="False"
MIDSTAGE_GATE_LOSS_WEIGHT="0.0"

USE_MSTVA="False"
USE_MSTVA_LOSS="False"
MSTVA_ALIGN_DIM="256"
MSTVA_LOSS_WEIGHT="0.0"
MSTVA_SCALE_WEIGHTS="0.5,0.3,0.2"

########################################
# Text-FiLM config
########################################
USE_TEXT_FILM="True"
TEXT_FILM_VISUAL_DIM="512"
TEXT_FILM_INIT_STD="1e-3"
TEXT_FILM_BRANCH_ALPHA="1.0"

########################################
# Output
########################################
TEXT_FILM_OUTPUT_DIR="/home/wangchengjun/huangziyi/reseg/output/tgi/text_film_res4_10w"
TEXT_FILM_MERGED_DIR="${TEXT_FILM_OUTPUT_DIR}/merged_model"
# 注意：eval_val_metrics.py 把 json 写到 dirname(PRED_DIR)/${DATASET}_${SPLIT}_metrics.json
# 因此 normal / bypass 必须用不同父目录，否则会互相覆盖同一份 rrsisd_test_metrics.json
TEXT_FILM_EVAL_NORMAL_DIR="${TEXT_FILM_OUTPUT_DIR}/eval_normal"
TEXT_FILM_EVAL_BYPASS_DIR="${TEXT_FILM_OUTPUT_DIR}/eval_bypass"
TEXT_FILM_TEST_NORMAL_DIR="${TEXT_FILM_EVAL_NORMAL_DIR}/pred_masks"
TEXT_FILM_TEST_BYPASS_DIR="${TEXT_FILM_EVAL_BYPASS_DIR}/pred_masks"
TEXT_FILM_METRICS_NORMAL_JSON="${TEXT_FILM_EVAL_NORMAL_DIR}/${DATASET_NAME}_${TEST_SPLIT}_metrics.json"
TEXT_FILM_METRICS_BYPASS_JSON="${TEXT_FILM_EVAL_BYPASS_DIR}/${DATASET_NAME}_${TEST_SPLIT}_metrics.json"

TRAIN_LOG="${TEXT_FILM_OUTPUT_DIR}/train_text_film_res4_5k.log"

########################################
# Eval metrics config（eval_val_metrics.py 读环境变量）
#   DATASET_TYPE / BASE_DATA_PATH / SPLIT / PRED_DIR
#   USE_WANDB / WANDB_PROJECT / WANDB_RUN_NAME
########################################
EVAL_METRICS_SCRIPT="/home/wangchengjun/huangziyi/reseg/eval_val_metrics.py"
EVAL_USE_WANDB="True"
EVAL_WANDB_PROJECT="segearth-eval-tgi-val"
# 若 EVAL_USE_WANDB=True，两次 metrics 会各起一个 run（由 run_name 区分）
EVAL_WANDB_RUN_NAME_PREFIX="text_film_res4_10w"

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
    --lora_dropout "${LORA_DROPOUT}" \
    --use_mstva "${USE_MSTVA}" \
    --mstva_align_dim "${MSTVA_ALIGN_DIM}" \
    --use_mstva_loss "${USE_MSTVA_LOSS}" \
    --mstva_loss_weight "${MSTVA_LOSS_WEIGHT}" \
    --mstva_scale_weights "${MSTVA_SCALE_WEIGHTS}" \
    --use_text_film "${USE_TEXT_FILM}" \
    --text_film_visual_dim "${TEXT_FILM_VISUAL_DIM}" \
    --text_film_init_std "${TEXT_FILM_INIT_STD}" \
    --text_film_branch_alpha "${TEXT_FILM_BRANCH_ALPHA}"
}

eval_model () {
  local model_dir="$1"
  local out_dir="$2"
  local text_film_eval_mode="$3"

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
    --zip_results False \
    --text_film_eval_mode "${text_film_eval_mode}" \
    --text_film_force_alpha 1.0
}

run_eval_metrics () {
  local pred_dir="$1"
  local run_name="$2"

  if [[ ! -f "${EVAL_METRICS_SCRIPT}" ]]; then
    echo "[WARN] eval metrics script not found: ${EVAL_METRICS_SCRIPT}"
    echo "[WARN] Skip metrics. You can run eval_val_metrics.py manually."
    return 0
  fi

  USE_WANDB="${EVAL_USE_WANDB}" \
  WANDB_PROJECT="${EVAL_WANDB_PROJECT}" \
  WANDB_RUN_NAME="${run_name}" \
  DATASET_TYPE="${DATASET_NAME}" \
  BASE_DATA_PATH="${BASE_DATA_PATH}" \
  SPLIT="${TEST_SPLIT}" \
  PRED_DIR="${pred_dir}" \
  python "${EVAL_METRICS_SCRIPT}"
}

########################################
# Preflight
########################################
echo "========================================"
echo "[0/7] Preflight checks"
echo "========================================"

echo "[INFO] REPO_DIR=${REPO_DIR}"
echo "[INFO] MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "[INFO] VISION_TOWER=${VISION_TOWER}"
echo "[INFO] VISION_TOWER_MASK=${VISION_TOWER_MASK}"
echo "[INFO] MASK_CONFIG=${MASK_CONFIG}"
echo "[INFO] BASE_DATA_PATH=${BASE_DATA_PATH}"
echo "[INFO] OUTPUT_DIR=${TEXT_FILM_OUTPUT_DIR}"
echo "[INFO] GPU_SLOT=${GPU_SLOT}"
echo "[INFO] USE_TEXT_FILM=${USE_TEXT_FILM}"
echo "[INFO] TEXT_FILM_VISUAL_DIM=${TEXT_FILM_VISUAL_DIM}"
echo "[INFO] TEXT_FILM_INIT_STD=${TEXT_FILM_INIT_STD}"
echo "[INFO] TEXT_FILM_BRANCH_ALPHA=${TEXT_FILM_BRANCH_ALPHA}"
echo "[INFO] USE_MSTVA=${USE_MSTVA}"
echo "[INFO] USE_MSTVA_LOSS=${USE_MSTVA_LOSS}"
echo "[INFO] EVAL_METRICS_SCRIPT=${EVAL_METRICS_SCRIPT}"
echo "[INFO] EVAL_USE_WANDB=${EVAL_USE_WANDB}"
echo "[INFO] EVAL_WANDB_PROJECT=${EVAL_WANDB_PROJECT}"
echo "[INFO] EVAL_WANDB_RUN_NAME_PREFIX=${EVAL_WANDB_RUN_NAME_PREFIX}"
echo "[INFO] TEXT_FILM_METRICS_NORMAL_JSON=${TEXT_FILM_METRICS_NORMAL_JSON}"
echo "[INFO] TEXT_FILM_METRICS_BYPASS_JSON=${TEXT_FILM_METRICS_BYPASS_JSON}"

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

if ! python segearth_r2/train/train.py --help 2>/dev/null | grep -q "use_text_film"; then
  echo "[ERROR] train.py does not expose --use_text_film."
  exit 1
fi

if ! python segearth_r2/train/train.py --help 2>/dev/null | grep -q "text_film_visual_dim"; then
  echo "[ERROR] train.py does not expose --text_film_visual_dim."
  exit 1
fi

if ! python segearth_r2/eval/eval.py --help 2>/dev/null | grep -q "text_film_eval_mode"; then
  echo "[ERROR] eval.py does not expose --text_film_eval_mode."
  exit 1
fi

mkdir -p "${TEXT_FILM_OUTPUT_DIR}"

echo "[OK] preflight passed"

########################################
# 1) Train
########################################
echo "========================================"
echo "[1/7] Training Text-FiLM res4 experiment: 5k"
echo "========================================"

export WANDB_NAME="tgi-text-film-res4-5k"

deepspeed --master_port="${MASTER_PORT_TEXT_FILM}" --include="${GPU_SLOT}" segearth_r2/train/train.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --base_data_path "${BASE_DATA_PATH}" \
  --dataset_name "${DATASET_NAME}" \
  --output_dir "${TEXT_FILM_OUTPUT_DIR}" \
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
  --use_midstage_gate_loss "${USE_MIDSTAGE_GATE_LOSS}" \
  --midstage_gate_loss_weight "${MIDSTAGE_GATE_LOSS_WEIGHT}" \
  --use_mstva "${USE_MSTVA}" \
  --mstva_align_dim "${MSTVA_ALIGN_DIM}" \
  --use_mstva_loss "${USE_MSTVA_LOSS}" \
  --mstva_loss_weight "${MSTVA_LOSS_WEIGHT}" \
  --mstva_scale_weights "${MSTVA_SCALE_WEIGHTS}" \
  --use_text_film "${USE_TEXT_FILM}" \
  --text_film_visual_dim "${TEXT_FILM_VISUAL_DIM}" \
  --text_film_init_std "${TEXT_FILM_INIT_STD}" \
  --text_film_branch_alpha "${TEXT_FILM_BRANCH_ALPHA}" \
  --report_to wandb 2>&1 | tee "${TRAIN_LOG}"

########################################
# 2) Quick train log check
########################################
echo "========================================"
echo "[2/7] Quick Text-FiLM log check"
echo "========================================"

echo "[INFO] Show Text-FiLM related logs:"
grep -E "text_film_branch|TextFiLM|text_film_gamma_norm|text_film_beta_norm|text_film_branch_alpha|trainable|requires_grad|bypass|warning|WARNING" "${TRAIN_LOG}" | head -n 160 || true

echo "[INFO] If you see TextFiLM bypass warning or channel mismatch warning above, stop and inspect."

########################################
# 3) Select checkpoint
########################################
echo "========================================"
echo "[3/7] Select checkpoint"
echo "========================================"

TEXT_FILM_BEST_CHECKPOINT=$(read_best_checkpoint "${TEXT_FILM_OUTPUT_DIR}")

if [[ -z "${TEXT_FILM_BEST_CHECKPOINT}" ]]; then
  echo "[WARN] best_model_checkpoint not found; fallback to last checkpoint"
  TEXT_FILM_BEST_CHECKPOINT=$(read_last_checkpoint "${TEXT_FILM_OUTPUT_DIR}")
fi

if [[ -z "${TEXT_FILM_BEST_CHECKPOINT}" ]]; then
  echo "[ERROR] no checkpoint found under ${TEXT_FILM_OUTPUT_DIR}"
  exit 1
fi

echo "[OK] TEXT_FILM_CHECKPOINT=${TEXT_FILM_BEST_CHECKPOINT}"

########################################
# 4) Merge
########################################
echo "========================================"
echo "[4/7] Merge selected checkpoint"
echo "========================================"

merge_ckpt "${TEXT_FILM_BEST_CHECKPOINT}" "${TEXT_FILM_MERGED_DIR}"

########################################
# 5) Check merged config
########################################
echo "========================================"
echo "[5/7] Check merged config for Text-FiLM fields"
echo "========================================"

if [[ ! -f "${TEXT_FILM_MERGED_DIR}/config.json" ]]; then
  echo "[ERROR] merged config.json not found: ${TEXT_FILM_MERGED_DIR}/config.json"
  exit 1
fi

grep -n '"use_text_film"\|"text_film_visual_dim"\|"text_film_init_std"\|"text_film_branch_alpha"' "${TEXT_FILM_MERGED_DIR}/config.json" || {
  echo "[ERROR] merged config.json missing Text-FiLM keys"
  exit 1
}

########################################
# 6) Eval normal and bypass
########################################
echo "========================================"
echo "[6/7] Eval Text-FiLM normal and bypass"
echo "========================================"

echo "[INFO] Eval normal"
eval_model "${TEXT_FILM_MERGED_DIR}" "${TEXT_FILM_TEST_NORMAL_DIR}" "normal"

echo "[INFO] Eval bypass"
eval_model "${TEXT_FILM_MERGED_DIR}" "${TEXT_FILM_TEST_BYPASS_DIR}" "bypass"

echo "[INFO] Metrics normal -> ${TEXT_FILM_METRICS_NORMAL_JSON}"
run_eval_metrics "${TEXT_FILM_TEST_NORMAL_DIR}" "${EVAL_WANDB_RUN_NAME_PREFIX}_normal"

echo "[INFO] Metrics bypass -> ${TEXT_FILM_METRICS_BYPASS_JSON}"
run_eval_metrics "${TEXT_FILM_TEST_BYPASS_DIR}" "${EVAL_WANDB_RUN_NAME_PREFIX}_bypass"

########################################
# 7) Done
########################################
echo "========================================"
echo "[7/7] DONE"
echo "Output dir       : ${TEXT_FILM_OUTPUT_DIR}"
echo "Selected ckpt    : ${TEXT_FILM_BEST_CHECKPOINT}"
echo "Merged model     : ${TEXT_FILM_MERGED_DIR}"
echo "Normal pred dir  : ${TEXT_FILM_TEST_NORMAL_DIR}"
echo "Bypass pred dir  : ${TEXT_FILM_TEST_BYPASS_DIR}"
echo "Metrics JSON (n) : ${TEXT_FILM_METRICS_NORMAL_JSON}"
echo "Metrics JSON (b) : ${TEXT_FILM_METRICS_BYPASS_JSON}"
echo "Train log        : ${TRAIN_LOG}"
echo "========================================"

echo ""
echo "[NEXT] Check these:"
echo "1) grep -E \"text_film_gamma_norm|text_film_beta_norm|text_film_branch_alpha|TextFiLM|bypass|WARNING\" ${TRAIN_LOG} | tail -n 100"
echo "2) Compare mIoU/cIoU: jq '.mIoU,.cIoU' ${TEXT_FILM_METRICS_NORMAL_JSON} ${TEXT_FILM_METRICS_BYPASS_JSON}"
echo "3) If normal == bypass, Text-FiLM has no causal inference contribution."