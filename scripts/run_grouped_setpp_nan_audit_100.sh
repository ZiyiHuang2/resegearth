#!/usr/bin/env bash
# 100-step NaN stability audit (grouped SET++), with debug logging on NaN/Inf.
set -euo pipefail

RESEG_ROOT="/root/rivermind-data/huangziyi/reseg"
export RUN_TAG="${RUN_TAG:-grouped-setpp-nan-audit-100}"
export WANDB_NAME="${WANDB_NAME:-${RUN_TAG}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${RESEG_ROOT}/output/full/${RUN_TAG}}"
export WARM_START_MODEL="${WARM_START_MODEL:-${RESEG_ROOT}/output/setpp/setpp-lasers-warmstart-8w-gd4/merged_model}"

export MAX_STEPS="${MAX_STEPS:-100}"
export SAVE_STEPS="${SAVE_STEPS:-100}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
export LOGGING_STEPS="${LOGGING_STEPS:-10}"
export REPORT_TO="${REPORT_TO:-wandb}"

export GROUPED_SETPP_NAN_DEBUG=1
export GROUPED_SETPP_NAN_DUMP_DIR="${OUTPUT_DIR}/nan_dumps"

export RUN_TRAIN=1
export RUN_MERGE=0
export RUN_EVAL=0
export RUN_PREFLIGHT=0
export RUN_CROSS_DATASET_EVAL=0

export GPU_ID="${GPU_ID:-0}"
export MASTER_PORT="${MASTER_PORT:-29641}"

echo "========================================"
echo "Grouped SET++ NaN audit: ${MAX_STEPS} steps"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  NAN_DUMP_DIR=${GROUPED_SETPP_NAN_DUMP_DIR}"
echo "========================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/../run_train_merge_test.sh"

PYTHON="/root/rivermind-data/miniconda3/envs/reseg/bin/python"
"${PYTHON}" - <<'PY'
import json, math
from pathlib import Path
import os

out = Path(os.environ["OUTPUT_DIR"])
state_path = out / "trainer_state.json"
if not state_path.exists():
    print(f"[audit summary] missing {state_path}")
    raise SystemExit(0)

state = json.loads(state_path.read_text())
logs = state.get("log_history", [])
mask_logs = [e for e in logs if "loss_mask" in e]
nan_steps = [e["step"] for e in mask_logs if isinstance(e.get("loss_mask"), float) and math.isnan(e["loss_mask"])]
loss_entries = [e for e in logs if e.get("loss") is not None and "loss_mask" not in e]

def loss_at(step):
    for e in loss_entries:
        if e.get("step") == step:
            return e["loss"]
    return float("nan")

summary = {
    "nan_mask_steps": nan_steps,
    "nan_count": len(nan_steps),
    "total_logged": len(mask_logs),
    "loss_step10": loss_at(10),
    "loss_step50": loss_at(50),
    "loss_step100": loss_at(100),
}
summary_path = out / "nan_audit_summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print("[audit summary]", json.dumps(summary, indent=2))
PY
