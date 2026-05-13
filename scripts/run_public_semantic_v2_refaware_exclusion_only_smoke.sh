#!/usr/bin/env bash
# Smoke：public_semantic_v2 + concept_refaware_prior + debug（stderr）。
# 用法：conda run -n reseg bash scripts/run_public_semantic_v2_refaware_exclusion_only_smoke.sh
set -euo pipefail

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_PROJECT="${WANDB_PROJECT:-segearth-source}"
unset CUDA_VISIBLE_DEVICES

GPU_SLOT="${GPU_SLOT:-localhost:1}"
MASTER_PORT="${MASTER_PORT:-29503}"

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
OUTPUT_DIR="${OUTPUT_DIR:-${RESEG_ROOT}/output/source/rrsisd_public_semantic_v2_refaware_exclusion_only_smoke}"
DEBUG_MAX="${DEBUG_CONCEPT_REFAWARE_PRIOR_MAX_SAMPLES:-80}"

if ! python -c "import transformers" 2>/dev/null; then
  if command -v conda >/dev/null 2>&1; then
    _TSR_PY="$(conda run -n reseg which python 2>/dev/null || true)"
    if [[ -n "${_TSR_PY}" && -x "${_TSR_PY}" ]]; then
      export PATH="$(dirname "${_TSR_PY}"):${PATH}"
    fi
  fi
fi
if ! python -c "import transformers" 2>/dev/null; then
  for _cand in "${HOME}/miniconda3/envs/reseg/bin" "${HOME}/anaconda3/envs/reseg/bin" "${HOME}/mambaforge/envs/reseg/bin"; do
    if [[ -x "${_cand}/python" ]]; then
      export PATH="${_cand}:${PATH}"
      break
    fi
  done
fi
if ! python -c "import transformers" 2>/dev/null; then
  echo "[ERROR] 需要可 import transformers 的 Python（请先 conda activate reseg 或安装依赖）。"
  exit 1
fi

HELP_OUT="$(python segearth_r2/train/train.py --help 2>&1 || true)"
echo "${HELP_OUT}" | grep -q "concept_refaware_prior" || {
  echo "[ERROR] train.py --help 未包含 concept_refaware_prior（请确认使用含 transformers 的 conda 环境，例如 conda activate reseg）。"
  exit 1
}
echo "${HELP_OUT}" | grep -qE "debug_concept_refaware_prior|debug_concept_refaware_prior_max_samples" || {
  echo "[ERROR] train.py --help 未包含 refaware debug 参数。"
  exit 1
}

echo "[CHECK] strict + refaware 同时开启应报错"
set +e
CONFLICT_ERR="$(python segearth_r2/train/train.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --base_data_path "${BASE_DATA_PATH}" \
  --dataset_name rrsisd \
  --output_dir /tmp/refaware_conflict_probe \
  --max_steps 1 \
  --concept_public_semantic_library "${CONCEPT_PUBLIC_SEMANTIC_LIBRARY}" \
  --concept_match_strict True \
  --concept_refaware_prior True 2>&1)"
CONFLICT_RC=$?
set -euo pipefail
if [[ "${CONFLICT_RC}" -eq 0 ]]; then
  echo "[ERROR] 预期 concept_match_strict + concept_refaware_prior 会失败，但进程退出码为 0"
  exit 1
fi
if ! echo "${CONFLICT_ERR}" | grep -q "concept_match_strict and concept_refaware_prior cannot both be True"; then
  echo "[ERROR] 未在 stderr/stdout 中找到互斥错误文案。"
  echo "${CONFLICT_ERR}" | head -n 40
  exit 1
fi
echo "[OK] 互斥校验通过"

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
  --concept_refaware_prior True \
  --debug_concept_refaware_prior True \
  --debug_concept_refaware_prior_max_samples "${DEBUG_MAX}" \
  2> "${OUTPUT_DIR}/smoke_train.stderr.log"

echo "[OK] smoke done. See ${OUTPUT_DIR}/smoke_train.stderr.log for [RRSISD][debug_concept_refaware_prior] lines."
echo "[OK] OUTPUT_DIR=${OUTPUT_DIR}"
