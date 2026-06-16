#!/usr/bin/env bash
# Phase 1: joint decoder engineering smoke from base merged_model
set -euo pipefail

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_PROJECT="${WANDB_PROJECT:-segearth-joint-smoke}"
export WANDB_NAME="${WANDB_NAME:-joint-smoke-from-base}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
unset CUDA_VISIBLE_DEVICES

GPU_SLOT="${GPU_SLOT:-localhost:0}"
GPU_ID="${GPU_ID:-0}"
MASTER_PORT="${MASTER_PORT:-$((29700 + RANDOM % 1000))}"

PYTHON="${PYTHON:-/root/rivermind-data/miniconda3/envs/reseg/bin/python}"
DEEPSPEED="${DEEPSPEED:-/root/rivermind-data/miniconda3/envs/reseg/bin/deepspeed}"

REPO_DIR="/root/rivermind-data/huangziyi/reseg/segearth+joint"
cd "${REPO_DIR}"

BASELINE_MERGED="/root/rivermind-data/huangziyi/reseg/output/base/standard-base-lasers-siglip1-28w-gd4/merged_model"
BASE_DATA_PATH="/root/rivermind-data/huangziyi/data/LaSeRS"
VISION_TOWER="/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384"
VISION_TOWER_MASK="/root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl"
MASK_CONFIG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"

OUTPUT_DIR="/root/rivermind-data/huangziyi/reseg/output/joint/smoke_joint_from_base"
MERGED_DIR="${OUTPUT_DIR}/merged_model"
EVAL_SINGLE_DIR="${OUTPUT_DIR}/eval_single_seg"
EVAL_MULTI_DIR="${OUTPUT_DIR}/eval_multi_seg"
LOG_FILE="${OUTPUT_DIR}/smoke.log"
SMOKE_STEPS="${SMOKE_STEPS:-400}"
SKIP_STEPS_1_2="${SKIP_STEPS_1_2:-0}"

mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "========================================"
echo "[0/5] Preflight"
echo "========================================"
echo "REPO_DIR=${REPO_DIR}"
echo "BASELINE_MERGED=${BASELINE_MERGED}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "SMOKE_STEPS=${SMOKE_STEPS}"

echo "SMOKE_STEPS=${SMOKE_STEPS} batch=4 grad_accum=1"

[[ -d "${BASELINE_MERGED}" ]] || { echo "[FAIL] missing merged model"; exit 1; }
[[ -f "${BASE_DATA_PATH}/train/annotations/train_data.json" ]] || { echo "[FAIL] missing train data"; exit 1; }
[[ -f "scripts/zero1.json" ]] || { echo "[FAIL] missing zero1.json"; exit 1; }

echo "[INFO] GPU status (will not kill any process):"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv 2>/dev/null || true

if [[ "${SKIP_STEPS_1_2}" == "1" ]]; then
  echo "[INFO] SKIP_STEPS_1_2=1, skip unit/inference smoke (steps 1-2 already passed)"
else
echo "========================================"
echo "[1/5] Unit shape smoke (scripts/smoke_joint_seg_decoder.py)"
echo "========================================"
SKIP_REAL_CKPT=1 "${PYTHON}" scripts/smoke_joint_seg_decoder.py
"${PYTHON}" -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

echo "========================================"
echo "[2/5] Single + multi [SEG] inference smoke"
echo "========================================"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" scripts/smoke_joint_inference.py \
  --model_path "${BASELINE_MERGED}" \
  --base_data_path "${BASE_DATA_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --mask_config "${MASK_CONFIG}"
"${PYTHON}" -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()" 2>/dev/null || true
fi

echo "========================================"
echo "[3/5] Short train smoke (${SMOKE_STEPS} steps from merged_model)"
echo "========================================"
rm -rf "${OUTPUT_DIR}"/checkpoint-* "${OUTPUT_DIR}"/trainer_state.json 2>/dev/null || true

"${DEEPSPEED}" --master_port="${MASTER_PORT}" --include="${GPU_SLOT}" segearth_r2/train/train.py \
  --model_name_or_path "${BASELINE_MERGED}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --base_data_path "${BASE_DATA_PATH}" \
  --dataset_name lasers \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps "${SMOKE_STEPS}" \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --save_strategy steps \
  --save_steps "${SMOKE_STEPS}" \
  --save_total_limit 1 \
  --evaluation_strategy no \
  --bf16 True \
  --learning_rate 1e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --tf32 False \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --deepspeed scripts/zero1.json \
  --mask_config "${MASK_CONFIG}" \
  --data_ratio 1 \
  --switch_bs 4 \
  --seed 42 \
  --data_seed 42 \
  --lasers_holdout_ratio 0.05 \
  --lasers_holdout_seed 42 \
  --report_to none

echo "========================================"
echo "[4/5] Check training loss (no NaN)"
echo "========================================"
OUTPUT_DIR="${OUTPUT_DIR}" "${PYTHON}" - <<'PY'
import json, math, os, sys
out = os.environ["OUTPUT_DIR"]
state_path = os.path.join(out, "trainer_state.json")
if not os.path.isfile(state_path):
    print("[FAIL] trainer_state.json missing")
    sys.exit(1)
state = json.load(open(state_path))
logs = state.get("log_history", [])
losses = [x["loss"] for x in logs if "loss" in x]
if not losses:
    print("[FAIL] no loss entries in log_history")
    sys.exit(1)
bad = [l for l in losses if l is None or (isinstance(l, float) and (math.isnan(l) or math.isinf(l)))]
if bad:
    print("[FAIL] NaN/Inf loss:", bad[:5])
    sys.exit(1)
print(f"[PASS] {len(losses)} loss entries, last={losses[-1]:.6f}, min={min(losses):.6f}, max={max(losses):.6f}")
PY

LAST_CKPT=$("${PYTHON}" - <<PY
import os, re
out = "${OUTPUT_DIR}"
cands = []
for n in os.listdir(out):
    m = re.match(r"checkpoint-(\d+)$", n)
    if m:
        cands.append((int(m.group(1)), os.path.join(out, n)))
print(sorted(cands)[-1][1] if cands else "")
PY
)
[[ -n "${LAST_CKPT}" ]] || { echo "[FAIL] no checkpoint after smoke train"; exit 1; }
echo "[OK] LAST_CKPT=${LAST_CKPT}"

echo "========================================"
echo "[5/5] Eval tif smoke on smoke checkpoint (single + multi SEG)"
echo "========================================"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" scripts/smoke_joint_inference.py \
  --model_path "${LAST_CKPT}" \
  --base_data_path "${BASE_DATA_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --mask_config "${MASK_CONFIG}" \
  --output_dir "${OUTPUT_DIR}"

echo "========================================"
echo "PHASE1_SMOKE_PASSED"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CHECKPOINT=${LAST_CKPT}"
echo "LOG=${LOG_FILE}"
echo "========================================"
