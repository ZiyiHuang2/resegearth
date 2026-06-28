#!/usr/bin/env bash
set -euo pipefail

########################################
# SET++ 全流程：Warm-start Train → Merge → LaSeRS Test Eval → W&B Metrics
#
# 数据集：LaSeRS（dataset_name=lasers）
# 起点：LaSeRS base-5w merged_model（standard-base-lasers-siglip1-5w-gd4，含 [SEG]）
#
# 官方 LaSeRS 只有 train + test（无 val）：
#   - 训练：train_data.json；训练期 validation 从 train holdout 5%
#   - 最终评估：5 数据集 test 并行（LaSeRS + RRSISD + RefSegRS + RISBench + EarthReason）
#
# Override examples:
#   GPU_ID=0 MAX_STEPS=50000 bash run_train_merge_test.sh
#   ABLATION=A0 bash run_train_merge_test.sh   # Base only
#   ABLATION=A1 bash run_train_merge_test.sh   # +ClosedLoop
#   ABLATION=A2 bash run_train_merge_test.sh   # +CSQR only
#   ABLATION=A3 bash run_train_merge_test.sh   # Full (default)
#   WARM_START_MODEL=/path/to/merged_model bash run_train_merge_test.sh
########################################

########################################
# Environment
########################################
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT="${WANDB_PROJECT:-segearth-setpp}"

########################################
# Ablation matrix (A0–A3)
########################################
ABLATION="${ABLATION:-A3}"

case "${ABLATION}" in
  A0)
    SETPP_CLOSED_LOOP=False
    SETPP_CSQR_ENABLE=False
    ;;
  A1)
    SETPP_CLOSED_LOOP=True
    SETPP_CSQR_ENABLE=False
    ;;
  A2)
    SETPP_CLOSED_LOOP=False
    SETPP_CSQR_ENABLE=True
    ;;
  A3)
    SETPP_CLOSED_LOOP=True
    SETPP_CSQR_ENABLE=True
    ;;
  *)
    echo "[ERROR] unknown ABLATION=${ABLATION}, expected A0/A1/A2/A3"
    exit 1
    ;;
esac

RUN_TAG="${RUN_TAG:-setpp-${ABLATION}-lasers-warmstart-5w-gd4}"
export WANDB_NAME="${WANDB_NAME:-${RUN_TAG}}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"

RESEG_ROOT="/root/rivermind-data/huangziyi/reseg"
CONDA_ENV_DIR="/root/rivermind-data/miniconda3/envs/reseg"
PYTHON="${CONDA_ENV_DIR}/bin/python"
DEEPSPEED="${CONDA_ENV_DIR}/bin/deepspeed"
export PATH="${CONDA_ENV_DIR}/bin:${PATH}"

GPU_ID="${GPU_ID:-0}"
GPU_SLOT="${GPU_SLOT:-localhost:${GPU_ID}}"
MASTER_PORT="${MASTER_PORT:-29621}"

########################################
# Project dir
########################################
REPO_DIR="${REPO_DIR:-${RESEG_ROOT}/segearth+set++}"
cd "${REPO_DIR}"

########################################
# Common paths
########################################
WARM_START_MODEL="${WARM_START_MODEL:-${RESEG_ROOT}/output/base/standard-base-lasers-siglip1-8w-gd4/merged_model}"
VISION_TOWER="${RESEG_ROOT}/pretrained_model/CLIP/siglip-so400m-patch14-384"
VISION_TOWER_MASK="${RESEG_ROOT}/pretrained_model/mask2former/model_final_54b88a.pkl"
MASK_CONFIG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"

########################################
# LaSeRS 数据集配置
########################################
BASE_DATA_PATH="${BASE_DATA_PATH:-/root/rivermind-data/huangziyi/data/LaSeRS}"
DATASET_NAME="lasers"
EVAL_SPLIT="test"
LASERS_BENCHMARK="${LASERS_BENCHMARK:-all}"
LASERS_HOLDOUT_RATIO="${LASERS_HOLDOUT_RATIO:-0.05}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"

########################################
# Output
########################################
OUTPUT_DIR="${OUTPUT_DIR:-${RESEG_ROOT}/output/setpp/${RUN_TAG}}"
MERGED_DIR="${MERGED_DIR:-${OUTPUT_DIR}/merged_model}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_DIR}/test_results}"
RUN_CROSS_DATASET_EVAL="${RUN_CROSS_DATASET_EVAL:-1}"

########################################
# Eval metrics config (auto upload to W&B)
########################################
EVAL_METRICS_SCRIPT="${RESEG_ROOT}/eval_val_metrics.py"
EVAL_USE_WANDB="True"
EVAL_WANDB_PROJECT="segearth-eval-setpp"
EVAL_WANDB_RUN_NAME="${EVAL_WANDB_RUN_NAME:-${RUN_TAG}}"

########################################
# Train config（对齐 base-5w LaSeRS preset，SET++ 继续训 50k steps）
########################################
MAX_STEPS="${MAX_STEPS:-50000}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"

SAVE_STEPS="${SAVE_STEPS:-5000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="0.0"
WARMUP_RATIO="0.03"
LR_SCHEDULER_TYPE="cosine"

LOGGING_STEPS="10"
BF16="True"
TF32="False"
MODEL_MAX_LENGTH="2048"
GRADIENT_CHECKPOINTING="False"
DATALOADER_NUM_WORKERS="8"

LORA_R="8"
LORA_ALPHA="16"
LORA_DROPOUT="0.05"

DATA_RATIO="1"
SWITCH_BS="4"

SEED="42"
DATA_SEED="42"

# 分段跑：已有 checkpoint 时 train.py 会自动 resume
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_MERGE="${RUN_MERGE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
MERGE_CHECKPOINT="${MERGE_CHECKPOINT:-}"

########################################
# Helpers
########################################
read_best_checkpoint () {
  local out_dir="$1"
  OUTPUT_DIR_FOR_READ="${out_dir}" "${PYTHON}" - <<'PY'
import json
import os
import sys

output_dir = os.environ["OUTPUT_DIR_FOR_READ"]
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
  OUTPUT_DIR_FOR_READ="${out_dir}" "${PYTHON}" - <<'PY'
import os
import re

output_dir = os.environ["OUTPUT_DIR_FOR_READ"]
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

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" segearth_r2/train/merge_lora_weights_and_save_hf_model.py \
    --model_path "${ckpt}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --mask_config "${MASK_CONFIG}" \
    --save_path "${save_dir}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --setpp_closed_loop "${SETPP_CLOSED_LOOP}" \
    --setpp_csqr_enable "${SETPP_CSQR_ENABLE}"
}

eval_model () {
  local model_dir="$1"
  local out_dir="$2"

  mkdir -p "${out_dir}"

  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${PYTHON}" segearth_r2/eval/eval.py \
    --base_data_path "${BASE_DATA_PATH}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --mask_config "${MASK_CONFIG}" \
    --model_path "${model_dir}" \
    --output_dir "${out_dir}" \
    --dataset_name "${DATASET_NAME}" \
    --split "${EVAL_SPLIT}" \
    --eval_batch_size 1 \
    --max_eval_samples "${EVAL_MAX_SAMPLES}" \
    --dataloader_num_workers 0 \
    --skip_existing True \
    --zip_results False
}

check_merged_setpp_model () {
  local model_dir="$1"
  local expect_csqr="${2:-1}"
  MERGED_MODEL_DIR="${model_dir}" EXPECT_CSQR="${expect_csqr}" "${PYTHON}" - <<'PY'
import json
import os
import sys

model_dir = os.environ["MERGED_MODEL_DIR"]
expect_csqr = os.environ.get("EXPECT_CSQR", "1") == "1"
index_path = os.path.join(model_dir, "model.safetensors.index.json")
required_substrings = ("SET_token_projector", "SET_query_embed")

if not os.path.isfile(index_path):
    print(f"[ERROR] missing weight index: {index_path}")
    sys.exit(1)

with open(index_path, "r", encoding="utf-8") as f:
    weight_map = json.load(f).get("weight_map", {})

keys = list(weight_map.keys())
missing = [name for name in required_substrings if not any(name in k for k in keys)]
if missing:
    print(f"[ERROR] merged model missing SET++ weights: {missing}")
    sys.exit(1)

for name in required_substrings:
    matched = [k for k in keys if name in k]
    print(f"[OK] {name}: {matched[0]}")

csqr_keys = [k for k in keys if "csqr_block" in k]
if expect_csqr and not csqr_keys:
    print("[ERROR] setpp_csqr_enable=True but merged model has no predictor.csqr_block weights")
    sys.exit(1)
if csqr_keys:
    print(f"[OK] csqr_block: {len(csqr_keys)} tensors (e.g. {csqr_keys[0]})")
else:
    print("[WARN] no csqr_block weights (setpp_csqr_enable=False or legacy run)")

config_path = os.path.join(model_dir, "config.json")
if os.path.isfile(config_path):
    cfg = json.load(open(config_path, encoding="utf-8"))
    print(f"[OK] config setpp_csqr_enable={cfg.get('setpp_csqr_enable')}, "
          f"setpp_closed_loop={cfg.get('setpp_closed_loop')}")

tok_cfg = os.path.join(model_dir, "tokenizer_config.json")
if os.path.isfile(tok_cfg):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    seg = tok.convert_tokens_to_ids("[SEG]")
    set_id = tok.convert_tokens_to_ids("[SET]")
    print(f"[OK] merged tokenizer len={len(tok)}, SEG_id={seg}, SET_id={set_id}")
    if seg is None or set_id is None or seg < 0 or set_id < 0:
        print("[ERROR] merged tokenizer missing [SEG]/[SET]")
        sys.exit(1)
PY
}

verify_training_checkpoint () {
  local ckpt="$1"
  if [[ ! -d "${ckpt}" ]]; then
    echo "[ERROR] checkpoint directory not found: ${ckpt}"
    exit 1
  fi
  if [[ ! -f "${ckpt}/trainer_state.json" ]]; then
    echo "[ERROR] trainer_state.json missing in ${ckpt}"
    exit 1
  fi
  if [[ ! -f "${ckpt}/zero_to_fp32.py" ]]; then
    echo "[ERROR] DeepSpeed checkpoint incomplete (zero_to_fp32.py missing): ${ckpt}"
    exit 1
  fi
  CKPT_DIR="${ckpt}" "${PYTHON}" - <<'PY'
import json, os, sys
ckpt = os.environ["CKPT_DIR"]
state = json.load(open(os.path.join(ckpt, "trainer_state.json"), encoding="utf-8"))
step = int(state.get("global_step", 0))
best = state.get("best_model_checkpoint", "")
print(f"[OK] trainer global_step={step}, best_model_checkpoint={best}")
if step <= 0:
    print("[ERROR] checkpoint global_step <= 0; training may not have run")
    sys.exit(1)
PY
}

run_cross_dataset_eval_all () {
  local model_dir="$1"
  echo "[INFO] 5-dataset parallel test eval: LaSeRS / RRSISD / RefSegRS / RISBench / EarthReason"
  MODEL_PATH="${model_dir}" \
  OUT_DIR="${OUTPUT_DIR}" \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  EVAL_SPLIT="${EVAL_SPLIT}" \
  EVAL_USE_WANDB="${EVAL_USE_WANDB}" \
  EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT}" \
  EVAL_WANDB_RUN_NAME="${EVAL_WANDB_RUN_NAME}" \
  LASERS_BENCHMARK="${LASERS_BENCHMARK}" \
  EVAL_METRICS_SCRIPT="${EVAL_METRICS_SCRIPT}" \
  VISION_TOWER="${VISION_TOWER}" \
  VISION_TOWER_MASK="${VISION_TOWER_MASK}" \
  MASK_CONFIG="${MASK_CONFIG}" \
  PYTHON="${PYTHON}" \
  REPO_DIR="${REPO_DIR}" \
  bash "${REPO_DIR}/run_cross_dataset_test_eval.sh"
}

run_eval_metrics () {
  local pred_dir="$1"
  local run_name="$2"

  if [[ ! -f "${EVAL_METRICS_SCRIPT}" ]]; then
    echo "[ERROR] eval metrics script not found: ${EVAL_METRICS_SCRIPT}"
    exit 1
  fi

  USE_WANDB="${EVAL_USE_WANDB}" \
  WANDB_PROJECT="${EVAL_WANDB_PROJECT}" \
  WANDB_RUN_NAME="${run_name}" \
  DATASET_TYPE="${DATASET_NAME}" \
  BASE_DATA_PATH="${BASE_DATA_PATH}" \
  SPLIT="${EVAL_SPLIT}" \
  LASERS_BENCHMARK="${LASERS_BENCHMARK}" \
  PRED_DIR="${pred_dir}" \
  "${PYTHON}" "${EVAL_METRICS_SCRIPT}"
}

########################################
# Preflight（LaSeRS train + test，无 val）
########################################
echo "========================================"
echo "[0/6] Preflight checks (LaSeRS train+test)"
echo "========================================"

echo "[INFO] REPO_DIR=${REPO_DIR}"
echo "[INFO] ABLATION=${ABLATION} closed_loop=${SETPP_CLOSED_LOOP} csqr=${SETPP_CSQR_ENABLE}"
echo "[INFO] RUN_TAG=${RUN_TAG}"
echo "[INFO] WARM_START_MODEL=${WARM_START_MODEL}"
echo "[INFO] BASE_DATA_PATH=${BASE_DATA_PATH}"
echo "[INFO] DATASET_NAME=${DATASET_NAME}"
echo "[INFO] EVAL_SPLIT=${EVAL_SPLIT} (LaSeRS 无 val，评估应对 test 子集)"
echo "[INFO] LASERS_BENCHMARK=${LASERS_BENCHMARK}"
echo "[INFO] LASERS_HOLDOUT_RATIO=${LASERS_HOLDOUT_RATIO}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] GPU_SLOT=${GPU_SLOT}"
echo "[INFO] GPU_ID=${GPU_ID}"
echo "[INFO] LEARNING_RATE=${LEARNING_RATE}"
echo "[INFO] MAX_STEPS=${MAX_STEPS}"
echo "[INFO] RUN_TRAIN=${RUN_TRAIN} RUN_MERGE=${RUN_MERGE} RUN_EVAL=${RUN_EVAL}"
echo "[INFO] MERGE_CHECKPOINT=${MERGE_CHECKPOINT:-<auto>}"
echo "[INFO] EVAL_WANDB_PROJECT=${EVAL_WANDB_PROJECT}"
echo "[INFO] EVAL_WANDB_RUN_NAME=${EVAL_WANDB_RUN_NAME}"
echo "[INFO] SET++ defaults: closed_loop=True, csqr=True"
echo "[INFO] RUN_CROSS_DATASET_EVAL=${RUN_CROSS_DATASET_EVAL} (5 datasets test parallel)"

if [[ ! -d "${WARM_START_MODEL}" ]]; then
  echo "[ERROR] warm-start model not found: ${WARM_START_MODEL}"
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

if [[ ! -f "${BASE_DATA_PATH}/train/annotations/train_data.json" ]]; then
  echo "[ERROR] train annotation not found: ${BASE_DATA_PATH}/train/annotations/train_data.json"
  echo "[HINT] 解压: tar -xzf ${BASE_DATA_PATH}/train.tar.gz -C ${BASE_DATA_PATH}"
  exit 1
fi

if [[ ! -d "${BASE_DATA_PATH}/train/images" ]]; then
  echo "[ERROR] train images not found: ${BASE_DATA_PATH}/train/images"
  exit 1
fi

if [[ ! -d "${BASE_DATA_PATH}/test/annotations" ]]; then
  echo "[ERROR] test annotations not found: ${BASE_DATA_PATH}/test/annotations"
  echo "[HINT] 解压: tar -xzf ${BASE_DATA_PATH}/test.tar.gz -C ${BASE_DATA_PATH}"
  exit 1
fi

if [[ ! -d "${BASE_DATA_PATH}/test/images" ]]; then
  echo "[ERROR] test images not found: ${BASE_DATA_PATH}/test/images"
  exit 1
fi

if [[ ! -f "scripts/zero1.json" ]]; then
  echo "[ERROR] DeepSpeed config not found: ${REPO_DIR}/scripts/zero1.json"
  exit 1
fi

if [[ ! -f "${EVAL_METRICS_SCRIPT}" ]]; then
  echo "[ERROR] eval metrics script not found: ${EVAL_METRICS_SCRIPT}"
  exit 1
fi

WARM_START_MODEL="${WARM_START_MODEL}" "${PYTHON}" - <<'PY'
from transformers import AutoTokenizer
import os

model_path = os.environ["WARM_START_MODEL"]
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
tokenizer.add_tokens(["[SEG]", "[SET]"])
ids = tokenizer("[SET][SEG]", add_special_tokens=False)["input_ids"]
if max(ids) >= 51200:
    raise SystemExit(f"[ERROR] SET/SEG token id exceeds historical lm_head size 51200: ids={ids}")
print(f"[OK] tokenizer len after SET/SEG={len(tokenizer)}, adjacent ids={ids}")
PY

mkdir -p "${OUTPUT_DIR}"

echo "[OK] preflight passed (train + test present, no val required)"

########################################
# 1) Train
#    train_data.json；eval 用同文件 holdout 5%（非 test，避免泄漏）
########################################
if [[ "${RUN_TRAIN}" != "1" ]]; then
  echo "========================================"
  echo "[1/6] SKIP training (RUN_TRAIN=${RUN_TRAIN})"
  echo "========================================"
else
echo "========================================"
echo "[1/6] Training SET++ on LaSeRS train (holdout eval)"
echo "========================================"

"${DEEPSPEED}" --master_port="${MASTER_PORT}" --include="${GPU_SLOT}" segearth_r2/train/train.py \
  --model_name_or_path "${WARM_START_MODEL}" \
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
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --deepspeed scripts/zero1.json \
  --mask_config "${MASK_CONFIG}" \
  --data_ratio "${DATA_RATIO}" \
  --switch_bs "${SWITCH_BS}" \
  --seed "${SEED}" \
  --data_seed "${DATA_SEED}" \
  --lasers_holdout_ratio "${LASERS_HOLDOUT_RATIO}" \
  --lasers_holdout_seed "${DATA_SEED}" \
  --setpp_closed_loop "${SETPP_CLOSED_LOOP}" \
  --setpp_csqr_enable "${SETPP_CSQR_ENABLE}" \
  --report_to wandb
fi

########################################
# 2) Select checkpoint
########################################
echo "========================================"
echo "[2/6] Select checkpoint"
echo "========================================"

BEST_CHECKPOINT="${MERGE_CHECKPOINT}"
if [[ -z "${BEST_CHECKPOINT}" ]]; then
  BEST_CHECKPOINT=$(read_best_checkpoint "${OUTPUT_DIR}")
fi

if [[ -z "${BEST_CHECKPOINT}" ]]; then
  echo "[WARN] best_model_checkpoint not found; fallback to last checkpoint"
  BEST_CHECKPOINT=$(read_last_checkpoint "${OUTPUT_DIR}")
fi

if [[ -z "${BEST_CHECKPOINT}" ]]; then
  echo "[ERROR] no checkpoint found under ${OUTPUT_DIR}"
  exit 1
fi

verify_training_checkpoint "${BEST_CHECKPOINT}"

echo "[OK] SELECTED_CHECKPOINT=${BEST_CHECKPOINT}"

########################################
# 3) Merge
########################################
if [[ "${RUN_MERGE}" != "1" ]]; then
  echo "========================================"
  echo "[3/6] SKIP merge (RUN_MERGE=${RUN_MERGE})"
  echo "========================================"
else
echo "========================================"
echo "[3/6] Merge selected checkpoint"
echo "========================================"

merge_ckpt "${BEST_CHECKPOINT}" "${MERGED_DIR}"
fi

########################################
# 4) Check merged SET++ weights
########################################
echo "========================================"
echo "[4/6] Check merged SET++ weights"
echo "========================================"

if [[ ! -f "${MERGED_DIR}/config.json" ]]; then
  echo "[ERROR] merged config.json not found: ${MERGED_DIR}/config.json"
  exit 1
fi

if [[ "${SETPP_CSQR_ENABLE}" == "True" ]]; then
  EXPECT_CSQR=1
else
  EXPECT_CSQR=0
fi

check_merged_setpp_model "${MERGED_DIR}" "${EXPECT_CSQR}"

########################################
# 5) Eval on 5 datasets (test) in parallel + metrics
########################################
if [[ "${RUN_EVAL}" != "1" ]]; then
  echo "========================================"
  echo "[5/6] SKIP eval (RUN_EVAL=${RUN_EVAL})"
  echo "========================================"
else
echo "========================================"
echo "[5/6] Eval on 5 datasets (test, parallel) + upload metrics"
echo "========================================"

if [[ "${RUN_CROSS_DATASET_EVAL}" == "1" ]]; then
  run_cross_dataset_eval_all "${MERGED_DIR}"
else
  eval_model "${MERGED_DIR}" "${EVAL_OUTPUT_DIR}"
  run_eval_metrics "${EVAL_OUTPUT_DIR}" "${EVAL_WANDB_RUN_NAME}"
fi
fi

########################################
# 6) Done
########################################
echo "========================================"
echo "[6/6] DONE"
echo "Dataset           : LaSeRS (train + test, no val)"
echo "Output dir        : ${OUTPUT_DIR}"
echo "Selected ckpt     : ${BEST_CHECKPOINT}"
echo "Merged model      : ${MERGED_DIR}"
echo "Test outputs      : ${OUTPUT_DIR}/*_test_results (5 datasets)"
echo "Eval logs         : ${OUTPUT_DIR}/*_test_eval.log"
echo "LaSeRS metrics    : ${OUTPUT_DIR}/lasers_test_metrics.json (if generated)"
echo "Eval W&B project  : ${EVAL_WANDB_PROJECT}"
echo "Eval W&B run name : ${EVAL_WANDB_RUN_NAME}"
echo "========================================"
