#!/usr/bin/env bash
# EarthReason test eval for 28w + 5w models in parallel on one GPU.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
BASE="/root/rivermind-data/huangziyi/reseg/output/base"
SCRIPT="/root/rivermind-data/huangziyi/reseg/segearth+base/run_earthreason_test_eval.sh"

run_one () {
  local tag="$1"
  local out_dir="$2"
  echo "[INFO] start ${tag} -> ${out_dir}"
  OUT_DIR="${out_dir}" bash "${SCRIPT}"
}

run_one "28w" "${BASE}/standard-base-lasers-siglip1-28w-gd4" &
PID_28W=$!

run_one "5w" "${BASE}/standard-base-lasers-siglip1-5w-gd4" &
PID_5W=$!

echo "[INFO] parallel PIDs: 28w=${PID_28W} 5w=${PID_5W}"
wait "${PID_28W}" && echo "[OK] 28w finished" || echo "[FAIL] 28w exit=$?"
wait "${PID_5W}" && echo "[OK] 5w finished" || echo "[FAIL] 5w exit=$?"

/root/rivermind-data/miniconda3/envs/reseg/bin/python3 <<'PY'
import json, glob, os

models = {
    "28w (80k)": "/root/rivermind-data/huangziyi/reseg/output/base/standard-base-lasers-siglip1-28w-gd4",
    "5w (50k)": "/root/rivermind-data/huangziyi/reseg/output/base/standard-base-lasers-siglip1-5w-gd4",
}
datasets = ["rrsisd", "refsegrs", "risbench", "earthreason"]

def lasers_overall(base):
    files = glob.glob(f"{base}/lasers_test_test_*_metrics.json")
    giou, ciou, pr, n = [], [], [], 0
    for f in files:
        d = json.load(open(f))
        ns = d.get("num_samples", 0)
        if ns:
            giou.append(d["gIoU"] * ns)
            ciou.append(d["cIoU"] * ns)
            pr.append(d["Pr@0.5"] * ns)
            n += ns
    if not n:
        return None
    return dict(gIoU=sum(giou) / n * 100, cIoU=sum(ciou) / n * 100, pr05=sum(pr) / n * 100, n=n)

def load_ds(base, ds):
    p = f"{base}/{ds}_test_metrics.json"
    if not os.path.isfile(p):
        return None
    d = json.load(open(p))
    return dict(
        gIoU=d["gIoU"] * 100,
        cIoU=d["cIoU"] * 100,
        pr05=d["Pr@0.5"] * 100,
        n=d.get("num_samples", 0),
        missing=d.get("missing_pred", 0),
    )

print("\n=== Cross-dataset test metrics (gIoU / cIoU / Pr@0.5) ===")
header = f"{'Dataset':<14}" + "".join(f"{m:>28}" for m in models)
print(header)
print("-" * len(header))

rows = [("LaSeRS", "lasers")] + [(ds, ds) for ds in datasets]
for label, key in rows:
    cells = [f"{label:<14}"]
    for mname, base in models.items():
        if key == "lasers":
            m = lasers_overall(base)
        else:
            m = load_ds(base, key)
        if m is None:
            cells.append(f"{'N/A':>28}")
        else:
            cells.append(f"{m['gIoU']:.2f}/{m['cIoU']:.2f}/{m['pr05']:.2f}".rjust(28))
    print("".join(cells))
PY
