#!/usr/bin/env bash
# Shared paths and helpers for Test 4 scripts. Source only; do not execute directly.

REPO_DIR="${REPO_DIR:-/root/rivermind-data/huangziyi/reseg/segearth+pd}"
RESEG_ROOT="${RESEG_ROOT:-/root/rivermind-data/huangziyi/reseg}"
CONDA_BIN="${CONDA_BIN:-/root/rivermind-data/miniconda3/envs/reseg/bin}"
PYTHON="${PYTHON:-${CONDA_BIN}/python}"
DEEPSPEED="${DEEPSPEED:-${CONDA_BIN}/deepspeed}"

# Root overlay /tmp is often tiny; keep caches and temp on the data volume.
export TMPDIR="${TMPDIR:-${RESEG_ROOT}/.tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${RESEG_ROOT}/.triton_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RESEG_ROOT}/.cache}"
mkdir -p "${TMPDIR}" "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}"

DATA_PATH="${DATA_PATH:-/root/rivermind-data/huangziyi/data/LaSeRS}"
VISION_TOWER="${VISION_TOWER:-${RESEG_ROOT}/pretrained_model/CLIP/siglip-so400m-patch14-384}"
VISION_TOWER_MASK="${VISION_TOWER_MASK:-${RESEG_ROOT}/pretrained_model/mask2former/model_final_54b88a.pkl}"
MASK_CONFIG="${MASK_CONFIG:-segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml}"
EVAL_METRICS_SCRIPT="${EVAL_METRICS_SCRIPT:-${RESEG_ROOT}/eval_val_metrics.py}"

MODEL_PATH_8W="${MODEL_PATH_8W:-${RESEG_ROOT}/output/base/standard-base-lasers-siglip1-8w-gd4/merged_model}"
# LaSeRS 8w base only on this machine; no standard-base-lasers-siglip1-28w-gd4/merged_model.

# All segearth+pd (Test 4) experiment artifacts live here — not under output/base/.
TEST4_OUTPUT_ROOT="${TEST4_OUTPUT_ROOT:-${RESEG_ROOT}/output/pd}"
mkdir -p "${TEST4_OUTPUT_ROOT}"

BASE_METRICS_JSON_8W="${BASE_METRICS_JSON_8W:-${RESEG_ROOT}/output/base/standard-base-lasers-siglip1-8w-gd4/lasers_test_table2_metrics.json}"

GATE_DELTA="${GATE_DELTA:-2.0}"
FAILFAST_DELTA="${FAILFAST_DELTA:-0.5}"
SPOT_CHECK_KEY="${SPOT_CHECK_KEY:-model.model.layers.30.self_attn.q_proj.weight}"
EVAL_CHECKPOINT_STEPS="${EVAL_CHECKPOINT_STEPS:-2000,5000}"

test4_resolve_base_model() {
  if [[ -n "${MODEL_PATH:-}" && -d "${MODEL_PATH}" ]]; then
    MODEL_BASE_SOURCE="explicit MODEL_PATH"
    return 0
  fi
  if [[ -d "${MODEL_PATH_8W}" ]]; then
    MODEL_PATH="${MODEL_PATH_8W}"
    MODEL_BASE_SOURCE="8w LaSeRS merged"
    return 0
  fi
  echo "[ERROR] No merged base found at ${MODEL_PATH_8W}. Set MODEL_PATH explicitly."
  return 1
}

test4_parse_multi_cate_giou() {
  local json_path="$1"
  TEST4_JSON_PATH="${json_path}" "${PYTHON}" - <<'PY'
import json, os, sys
path = os.environ.get("TEST4_JSON_PATH", "")
if not path or not os.path.isfile(path):
    sys.exit(1)
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
rows = data
if isinstance(data, dict):
    rows = data.get("rows") or data.get("table") or []
if not isinstance(rows, list):
    sys.exit(1)
for row in rows:
    if not isinstance(row, dict):
        continue
    stem = str(row.get("stem") or row.get("benchmark") or "")
    if "multi_cate" in stem:
        val = row.get("gIoU")
        if val is None:
            val = row.get("giou")
        if val is not None:
            print(float(val))
            sys.exit(0)
sys.exit(1)
PY
}

test4_read_baseline_giou() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] baseline file not found: ${path}" >&2
    return 1
  fi
  if [[ "${path}" == *.json ]]; then
    test4_parse_multi_cate_giou "${path}"
  else
    tr -d '[:space:]' < "${path}"
  fi
}

test4_default_base_metrics_json() {
  if [[ -n "${MODEL_PATH:-}" ]]; then
    local parent
    parent="$(dirname "${MODEL_PATH}")"
    if [[ -f "${parent}/lasers_test_table2_metrics.json" ]]; then
      echo "${parent}/lasers_test_table2_metrics.json"
      return 0
    fi
  fi
  if [[ -f "${BASE_METRICS_JSON_8W}" ]]; then
    echo "${BASE_METRICS_JSON_8W}"
    return 0
  fi
  echo "[ERROR] No lasers_test_table2_metrics.json under base model dir or ${BASE_METRICS_JSON_8W}" >&2
  return 1
}

# Write numeric baseline txt from base folder table2 JSON (Step 0 substitute).
test4_seed_baseline_from_metrics() {
  local metrics_json="$1"
  local out_txt="$2"
  local gio
  gio="$(test4_parse_multi_cate_giou "${metrics_json}")" || {
    echo "[ERROR] failed to parse test_multi_cate from ${metrics_json}" >&2
    return 1
  }
  mkdir -p "$(dirname "${out_txt}")"
  printf '%s\n' "${gio}" > "${out_txt}"
  echo "[Test4] baseline ${gio}% from ${metrics_json} -> ${out_txt}" >&2
}

test4_gate_label() {
  local baseline="$1"
  local gio="$2"
  BASELINE_GIOU="${baseline}" TEST4_GIOU="${gio}" \
  TEST4_GATE_DELTA="${GATE_DELTA}" TEST4_FAILFAST="${FAILFAST_DELTA}" \
  "${PYTHON}" - <<'PY'
import os
b=float(os.environ["BASELINE_GIOU"]); g=float(os.environ["TEST4_GIOU"])
gate=float(os.environ["TEST4_GATE_DELTA"]); fail=float(os.environ["TEST4_FAILFAST"])
d=g-b
if g>=b+gate: print("PASS")
elif d<fail: print("FAIL-FAST")
else: print("INCONCLUSIVE")
PY
}

test4_find_table2_json() {
  local pred_dir="$1"
  local train_dir="${2:-}"
  local candidates=(
    "$(dirname "${pred_dir}")/lasers_test_table2_metrics.json"
    "${pred_dir}/../lasers_test_table2_metrics.json"
    "${train_dir}/lasers_test_table2_metrics.json"
  )
  local p
  for p in "${candidates[@]}"; do
    if [[ -f "${p}" ]]; then
      echo "${p}"
      return 0
    fi
  done
  find "$(dirname "${pred_dir}")" -maxdepth 3 -name 'lasers_test_table2_metrics.json' 2>/dev/null | head -1
}

test4_print_gate() {
  local baseline="$1"
  local gio="$2"
  TEST4_BASELINE="${baseline}" TEST4_GIOU="${gio}" \
  TEST4_GATE_DELTA="${GATE_DELTA}" TEST4_FAILFAST="${FAILFAST_DELTA}" \
  "${PYTHON}" - <<'PY'
import os
base = float(os.environ["TEST4_BASELINE"])
gio = float(os.environ["TEST4_GIOU"])
gate = float(os.environ["TEST4_GATE_DELTA"])
fail = float(os.environ["TEST4_FAILFAST"])
delta = gio - base
print(f"baseline={base:.2f}%  result={gio:.2f}%  delta={delta:+.2f} pt")
if gio >= base + gate:
    print("GATE=PASS")
elif delta < fail:
    print("GATE=FAIL-FAST")
else:
    print("GATE=INCONCLUSIVE")
PY
}

test4_is_hf_model_dir() {
  local dir="$1"
  [[ -f "${dir}/config.json" ]] && {
    [[ -f "${dir}/model.safetensors.index.json" ]] || \
    [[ -f "${dir}/pytorch_model.bin" ]] || \
    compgen -G "${dir}/model-*.safetensors" > /dev/null
  }
}

test4_is_deepspeed_checkpoint() {
  local dir="$1"
  [[ -f "${dir}/zero_to_fp32.py" ]] || [[ -d "${dir}/global_step0" ]]
}

# Returns a merged HF model dir ready for eval.py (merge if needed).
test4_export_checkpoint() {
  local ckpt="$1"
  local out_merged="$2"
  local spot_base="${3:-}"

  if test4_is_hf_model_dir "${ckpt}"; then
    echo "[Test4] HF checkpoint detected; skip ZeRO merge: ${ckpt}" >&2
    echo "${ckpt}"
    return 0
  fi

  if test4_is_deepspeed_checkpoint "${ckpt}"; then
    CHECKPOINT_PATH="${ckpt}" MERGED_DIR="${out_merged}" SPOT_CHECK_BASE="${spot_base}" \
      bash "${SCRIPT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/test4_merge_checkpoint.sh"
    echo "${out_merged}"
    return 0
  fi

  echo "[ERROR] Unknown checkpoint format: ${ckpt}"
  return 1
}

test4_preflight() {
  if [[ ! -x "${DEEPSPEED}" ]]; then
    echo "[ERROR] deepspeed not found: ${DEEPSPEED}"
    return 1
  fi
  if [[ ! -x "${PYTHON}" ]]; then
    echo "[ERROR] python not found: ${PYTHON}"
    return 1
  fi
  if [[ ! -f "${EVAL_METRICS_SCRIPT}" ]]; then
    echo "[ERROR] eval metrics script not found: ${EVAL_METRICS_SCRIPT}"
    return 1
  fi
  if [[ ! -d "${DATA_PATH}" ]]; then
    echo "[ERROR] DATA_PATH not found: ${DATA_PATH}"
    return 1
  fi
  return 0
}
