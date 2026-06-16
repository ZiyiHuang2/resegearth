#!/usr/bin/env bash
# Dry-run: merge an existing TCPD DeepSpeed checkpoint without training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=test4_common.sh
source "${SCRIPT_DIR}/test4_common.sh"

REPO_DIR="${REPO_DIR}"
cd "${REPO_DIR}"
export PATH="${CONDA_BIN}:${PATH}"

DEFAULT_BASE_MODEL="${RESEG_ROOT}/output/base/standard-base-lasers-siglip1-8w-gd4/merged_model"
MODEL_PATH="${MODEL_PATH:-${DEFAULT_BASE_MODEL}}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "Usage: CHECKPOINT_PATH=/path/to/checkpoint-N [MERGED_DIR=/tmp/out] bash scripts/tcpd_merge_verify.sh"
  exit 1
fi

MERGED_DIR="${MERGED_DIR:-/tmp/tcpd-merge-verify}"
rm -rf "${MERGED_DIR}"

echo "[tcpd_merge_verify] checkpoint: ${CHECKPOINT_PATH}"
echo "[tcpd_merge_verify] base init : ${MODEL_PATH}"
echo "[tcpd_merge_verify] merged out: ${MERGED_DIR}"

CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
MERGED_DIR="${MERGED_DIR}" \
LORA_ENABLE=False \
USE_TCPD=True \
TCPD_CONDITION_SOURCE=seg \
SPOT_CHECK_BASE="${MODEL_PATH}" \
bash "${SCRIPT_DIR}/test4_merge_checkpoint.sh"

"${PYTHON}" - <<PY
import json, sys
cfg = json.load(open("${MERGED_DIR}/config.json"))
idx = json.load(open("${MERGED_DIR}/model.safetensors.index.json"))
tcpd = [k for k in idx["weight_map"] if "tcpd" in k]
assert cfg.get("use_tcpd") is True, cfg
assert cfg.get("tcpd_condition_source") == "seg", cfg
assert len(tcpd) > 0, "no tcpd weights in merged index"
print(f"[OK] use_tcpd={cfg['use_tcpd']}, tcpd weights={len(tcpd)}")
PY

echo "[tcpd_merge_verify] PASSED"
