#!/usr/bin/env bash
set -euo pipefail

PYTHON="/root/rivermind-data/miniconda3/envs/reseg/bin/python"
DEEPSPEED="/root/rivermind-data/miniconda3/envs/reseg/bin/deepspeed"
RESEG_ROOT="/root/rivermind-data/huangziyi/reseg"
CSDEG_DIR="${RESEG_ROOT}/segearth+csdeg"
BASE_DIR="${RESEG_ROOT}/segearth+base"
SMOKE_ROOT="/tmp/csdeg_e2e_smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SMOKE_ROOT}"

export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${CSDEG_DIR}:${PYTHONPATH:-}"
export WANDB_MODE=disabled

# LaSeRS: align with segearth+set++/run_train_merge_test.sh
DATA="/root/rivermind-data/huangziyi/data/LaSeRS"
DATASET_NAME="lasers"
LASERS_HOLDOUT_RATIO="0.05"
LASERS_HOLDOUT_SEED="42"

MODEL="${RESEG_ROOT}/pretrained_model/mllm/Mipha-3B"
VISION="${RESEG_ROOT}/pretrained_model/CLIP/siglip-so400m-patch14-384"
MASK2FORMER="${RESEG_ROOT}/pretrained_model/mask2former/model_final_54b88a.pkl"
MASK_OFF="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
MASK_ON="segearth_r2/model/mask_decoder/mask_config/maskformer2_csdeg_smoke.yaml"

COMMON_TRAIN_ARGS=(
  --model_name_or_path "${MODEL}"
  --vision_tower "${VISION}"
  --vision_tower_mask "${MASK2FORMER}"
  --base_data_path "${DATA}"
  --dataset_name "${DATASET_NAME}"
  --lasers_holdout_ratio "${LASERS_HOLDOUT_RATIO}"
  --lasers_holdout_seed "${LASERS_HOLDOUT_SEED}"
  --data_ratio "1"
  --switch_bs "4"
  --per_device_train_batch_size 1
  --gradient_accumulation_steps 1
  --bf16 True
  --dataloader_num_workers 2
  --deepspeed scripts/zero1.json
  --report_to none
)

log() { echo "[$(date -Iseconds)] $*" | tee -a "${SMOKE_ROOT}/smoke.log"; }

log "SMOKE_ROOT=${SMOKE_ROOT}"
log "DATA=${DATA} dataset=${DATASET_NAME} holdout=${LASERS_HOLDOUT_RATIO}"
log "GPU: $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | head -1)"

log "=== Standalone module smoke ==="
"${PYTHON}" "${CSDEG_DIR}/tools/diagnostics/smoke_csdeg_core.py" | tee -a "${SMOKE_ROOT}/smoke.log"

log "=== Decoder identity: base vs csdeg ENABLED=false (shared weights) ==="
"${PYTHON}" "${CSDEG_DIR}/tools/diagnostics/smoke_csdeg_identity_decoder.py" \
  --repo "${BASE_DIR}" --out "${SMOKE_ROOT}/dec_base.pt" --device cuda
"${PYTHON}" "${CSDEG_DIR}/tools/diagnostics/smoke_csdeg_identity_decoder.py" \
  --repo "${CSDEG_DIR}" --state-in "${SMOKE_ROOT}/dec_base.pt" --out "${SMOKE_ROOT}/dec_csdeg_off.pt" --device cuda

"${PYTHON}" - <<PY | tee -a "${SMOKE_ROOT}/smoke.log"
import torch
base = torch.load("${SMOKE_ROOT}/dec_base.pt", map_location="cpu")
off = torch.load("${SMOKE_ROOT}/dec_csdeg_off.pt", map_location="cpu")
diff = (base["pred_masks"] - off["pred_masks"]).abs().max().item()
print(f"identity base vs csdeg_off max_abs_diff={diff}")
assert diff < 1e-5, f"identity failed: {diff}"
assert not off["has_evidence"]
print("PASS decoder identity")
PY

cd "${CSDEG_DIR}"

log "=== 1-step train LaSeRS CS_DEG ENABLED=false ==="
"${DEEPSPEED}" --include localhost:0 --master_port=29641 \
  segearth_r2/train/train.py \
  "${COMMON_TRAIN_ARGS[@]}" \
  --mask_config "${MASK_OFF}" \
  --output_dir "${SMOKE_ROOT}/train_off_1step" \
  --max_steps 1 \
  --save_strategy no \
  --logging_steps 1 \
  2>&1 | tee "${SMOKE_ROOT}/train_off_1step.log"
log "PASS 1-step LaSeRS ENABLED=false"

log "=== 1-step train LaSeRS CS_DEG ENABLED=true ==="
"${DEEPSPEED}" --include localhost:0 --master_port=29642 \
  segearth_r2/train/train.py \
  "${COMMON_TRAIN_ARGS[@]}" \
  --mask_config "${MASK_ON}" \
  --output_dir "${SMOKE_ROOT}/train_on_1step" \
  --max_steps 1 \
  --save_strategy no \
  --logging_steps 1 \
  2>&1 | tee "${SMOKE_ROOT}/train_on_1step.log"
log "PASS 1-step LaSeRS ENABLED=true"

log "=== 50-step sanity LaSeRS CS_DEG ENABLED=true ==="
"${DEEPSPEED}" --include localhost:0 --master_port=29643 \
  segearth_r2/train/train.py \
  "${COMMON_TRAIN_ARGS[@]}" \
  --mask_config "${MASK_ON}" \
  --output_dir "${SMOKE_ROOT}/train_on_50step" \
  --max_steps 50 \
  --save_strategy no \
  --logging_steps 10 \
  2>&1 | tee "${SMOKE_ROOT}/train_on_50step.log"
log "PASS 50-step LaSeRS ENABLED=true"

log "ALL E2E SMOKES PASSED: ${SMOKE_ROOT}"
