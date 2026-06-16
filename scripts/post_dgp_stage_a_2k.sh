#!/usr/bin/env bash
# Post-training: health probes + full test eval + report (stopped before 2k).
set -euo pipefail

REPO="/root/rivermind-data/huangziyi/reseg/segearth+DGP"
OUT="/root/rivermind-data/huangziyi/reseg/output/dgp/v6.1-stage-a-500diag-full"
PYTHON="${PYTHON:-/root/rivermind-data/miniconda3/envs/reseg/bin/python}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

cd "${REPO}"
mkdir -p "${OUT}/health_2k" "${OUT}/eval_2k"

echo "[1/4] full test eval step1000..."
bash scripts/eval_dgp_stage_a_2k_full.sh "${OUT}/checkpoint-1000" step1000

echo "[2/4] full test eval step1300 (final, training stopped ~1400)..."
bash scripts/eval_dgp_stage_a_2k_full.sh "${OUT}/checkpoint-1300" step1300

echo "[3/4] health probes (500 root, 1000, 1300)..."
for spec in \
  "500:${OUT}/dgp_stage_a_weights.bin" \
  "1000:${OUT}/checkpoint-1000/dgp_stage_a_weights.bin" \
  "1300:${OUT}/checkpoint-1300/dgp_stage_a_weights.bin"; do
  step="${spec%%:*}"
  w="${spec#*:}"
  if [[ -f "${w}" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" tools/probe_dgp_health.py \
      --weights "${w}" --out "${OUT}/health_2k/step${step}.json" || true
    sleep 5
  fi
done

echo "[4/4] report..."
"${PYTHON}" tools/generate_dgp_2k_report.py --out-dir "${OUT}"

echo "[OK] post-process done -> ${OUT}/report_2k.md"
