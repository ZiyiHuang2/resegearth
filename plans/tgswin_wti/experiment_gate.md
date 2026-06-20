# TG-Swin-WTI v1.5 — Experiment Gate

## Baseline anchor (8w warmstart)

| Metric | Value |
|--------|-------|
| multi_cate gIoU | **42.45** |
| Path | `/root/rivermind-data/huangziyi/reseg/output/base/standard-base-lasers-siglip1-8w-gd4` |

## Code gate (before any training)

```bash
cd /root/rivermind-data/huangziyi/reseg/segearth+tgswin
/root/rivermind-data/miniconda3/envs/reseg/bin/python -m py_compile ...
/root/rivermind-data/miniconda3/envs/reseg/bin/python tools/diagnostics/probe_tgswin_v15_train_init_gradient.py
/root/rivermind-data/miniconda3/envs/reseg/bin/python tools/diagnostics/smoke_tgswin_core.py
```

Requirements:
- `gated bias ≈ 0` at `alpha=0`
- `raw_bias` non-zero
- `alpha.grad > 0` (real init, no manual perturbation)
- Swin backbone frozen; TG-Swin trainable

## 1-step smoke

- import / shape / backward / loss non-NaN
- `alpha.grad` non-zero
- Swin backbone frozen

```bash
GPU_ID=0 MAX_STEPS=1 RUN_MERGE=0 RUN_EVAL=0 bash run_train_merge_test_tgswin.sh
```

## 100-step probe

- `alpha` moves away from 0
- `bias_abs_mean` moves away from 0
- `reliability` non-NaN
- loss stable (no explosion)

```bash
GPU_ID=0 MAX_STEPS=100 RUN_MERGE=0 RUN_EVAL=0 bash run_train_merge_test_tgswin.sh
```

## 1000-step adjudication

- Run eval + **auto taxonomy** (`RUN_TAXONOMY=1`)
- Inspect `test_multi_cate` context_leak

| Outcome | Rule |
|---------|------|
| **Kill WTI v1.5** | leak ↓ **< 2 pt** → do **not** proceed to v2 |
| **Partial success** | leak ↓ clearly but gIoU ≤ 42.45 → analyze threshold / convergence / repeat cost |
| **Proceed to 5000-step** | leak ↓ ≥ **2 pt** |

Must inspect small-target and K≥3 buckets.

## 5000-step gate

| Signal | Rule |
|--------|------|
| Positive task signal | multi_cate **> 42.45** |
| Consider v2 | multi_cate **≥ 43.0** **and** leak clearly down |
| Regression guard | single / explicit each lose ≤ **0.5** gIoU pt |

## Full 8w (80000 steps)

Only after 5000-step passes mechanism + task gates.

Default script `MAX_STEPS=80000` — **do not run full 8w until short gates pass**.

## Taxonomy (automated)

`run_train_merge_test_tgswin.sh` runs `audit_lasers_error_taxonomy.py` when `RUN_EVAL=1` and `RUN_TAXONOMY=1`.

Output:
- `/root/rivermind-data/huangziyi/reseg/plans/baseline_diagnostics/tgswin_wti_multi_cate_error_taxonomy.json`
- `.md` sibling

## Kill playbook

leak < 2pt at 1000-step → **kill WTI v1.5**; analyze router/WTI stats; **do not** stack PES / CS-DEG / SET++.
