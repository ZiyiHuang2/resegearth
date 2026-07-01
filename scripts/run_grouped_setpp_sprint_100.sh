#!/usr/bin/env bash
# 100-step short sprint: validate grouped SET++ + TG-Swin training stability.
set -euo pipefail

RESEG_ROOT="/root/rivermind-data/huangziyi/reseg"
export RUN_TAG="${RUN_TAG:-grouped-setpp-sprint-100}"
export WANDB_NAME="${WANDB_NAME:-${RUN_TAG}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${RESEG_ROOT}/output/full/${RUN_TAG}}"
export WARM_START_MODEL="${WARM_START_MODEL:-${RESEG_ROOT}/output/setpp/setpp-lasers-warmstart-8w-gd4/merged_model}"

export MAX_STEPS="${MAX_STEPS:-100}"
export SAVE_STEPS="${SAVE_STEPS:-100}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
export LOGGING_STEPS="${LOGGING_STEPS:-10}"
export REPORT_TO="${REPORT_TO:-wandb}"

export RUN_TRAIN=1
export RUN_MERGE=0
export RUN_EVAL=0
export RUN_PREFLIGHT=0
export RUN_CROSS_DATASET_EVAL=0

export GPU_ID="${GPU_ID:-0}"
export MASTER_PORT="${MASTER_PORT:-29631}"

echo "========================================"
echo "Grouped SET++ sprint: ${MAX_STEPS} steps"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  WARM_START=${WARM_START_MODEL}"
echo "  GPU_ID=${GPU_ID}"
echo "========================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/../run_train_merge_test.sh"
