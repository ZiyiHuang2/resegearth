#!/usr/bin/env bash
# Eval-only launcher (clean v1.5 code) for warmstart-2w-bs2-gd4 checkpoint.
set -euo pipefail

RESEG_ROOT="/root/rivermind-data/huangziyi/reseg"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-${RESEG_ROOT}/output/tgswin/tgswin-wti-v15-lasers-warmstart-2w-bs2-gd4}"
GPU_ID="${GPU_ID:-0}"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

RUN_LASERS="${RUN_LASERS:-0}"
RUN_CROSS="${RUN_CROSS:-1}"
RUN_TAXONOMY="${RUN_TAXONOMY:-0}"

if [[ "${RUN_LASERS}" == "1" ]]; then
  echo "[1/2] LaSeRS test eval + metrics"
  RUN_TRAIN=0 RUN_MERGE=0 RUN_EVAL=1 RUN_TAXONOMY="${RUN_TAXONOMY}" \
    GPU_ID="${GPU_ID}" OUTPUT_DIR="${OUT_DIR}" \
    bash "${REPO_DIR}/run_train_merge_test_tgswin.sh"
fi

if [[ "${RUN_CROSS}" == "1" ]]; then
  echo "[2/2] Cross-dataset test eval (RRSISD → RefSegRS → RISBench → EarthReason)"
  REPO_DIR="${REPO_DIR}" OUT_DIR="${OUT_DIR}" CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    bash "${REPO_DIR}/run_cross_dataset_test_eval_seq.sh"
fi

echo "[DONE] results under ${OUT_DIR}"
