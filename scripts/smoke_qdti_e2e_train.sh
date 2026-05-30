#!/usr/bin/env bash
# E2E smoke: 1-2 step QDTI-Core train + diagnostic log (no mask feedback / rank / neg / div).
set -euo pipefail
RESEG_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${RESEG_ROOT}"
export PYTHONPATH="detectron2:${PYTHONPATH:-}"

OUTPUT_DIR="${OUTPUT_DIR:-${RESEG_ROOT}/output/tgi/smoke_qdti_e2e_v26}"
DIAG_LOG="${OUTPUT_DIR}/qdti_e2e_diag.log"
mkdir -p "${OUTPUT_DIR}"

MODEL_PATH="${MODEL_PATH:-/root/rivermind-data/huangziyi/reseg/pretrained_model/mllm/Mipha-3B}"
VISION_TOWER="${VISION_TOWER:-/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384}"
VISION_TOWER_MASK="${VISION_TOWER_MASK:-/root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/maskformer2_swin_base_IN21k_384_bs16_50ep.pkl}"
BASE_DATA_PATH="${BASE_DATA_PATH:-/root/rivermind-data/huangziyi/data/RRSISD}"

{
  echo "=== QDTI E2E smoke $(date -Iseconds) ==="
  PYTHON="${PYTHON:-/root/rivermind-data/miniconda3/envs/reseg/bin/python}"
  echo "PYTHON=${PYTHON}"
  "${PYTHON}" -c "import transformers; print('transformers', transformers.__version__)" 2>&1 || echo "MISSING: transformers"
  "${PYTHON}" -c "import deepspeed; print('deepspeed', deepspeed.__version__)" 2>&1 || echo "MISSING: deepspeed"
  echo "--- unit smoke ---"
  "${PYTHON}" tools/smoke_qdti_core_v26.py 2>&1 || true
  echo "--- train 2 steps ---"
  "${PYTHON}" segearth_r2/train/train.py \
    --model_name_or_path "${MODEL_PATH}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --base_data_path "${BASE_DATA_PATH}" \
    --dataset_name rrsisd \
    --output_dir "${OUTPUT_DIR}" \
    --max_steps 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --save_steps 1000000 \
    --report_to none \
    --use_query_aware_decoder_bias True \
    --decoder_attn_bias_apply_layers last3 \
    --decoder_attn_bias_max_abs 0.02 \
    --qdti_gate_init 0.0 \
    --qdti_warmup_steps 500 \
    --use_qdti_mask_feedback False \
    --use_qdti_rank_loss False \
    --use_qdti_neg_loss False \
    --use_qdti_div_loss False \
    --use_decoder_attn_bias False \
    --use_mstva False \
    --use_text_film False \
    --allow_random_qdti_init False \
    --deepspeed "" \
    2>&1
} | tee "${DIAG_LOG}"

echo ""
echo "=== Parse train log for QDTI diagnostics ==="
grep -E '\[DEBUG\]\[QDTICore\]|\[QDTI\]|qdti_|loss_qdti|use_query_aware' "${DIAG_LOG}" | tail -30 || echo "(no QDTI lines in log)"

echo "Diag log: ${DIAG_LOG}"
