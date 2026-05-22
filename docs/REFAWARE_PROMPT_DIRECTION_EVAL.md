# Refaware prompt direction eval (Phase 2.5)

**Diagnostic only** — not standard benchmark inference.

- **model_path**: `/home/wangchengjun/huangziyi/reseg/output/source/rrsisd_public_semantic_v2_refaware_exclusion_only_28w/rrsisd_public_semantic_v2_refaware_exclusion_only_28w_merged_best`
- **split**: test
- **sample_count (paired)**: 3480
- **pilot_only**: False

## Highest-priority result

- **overall delta mIoU** (refaware − raw): **-0.000268**
- **decision**: **C** — prior_direction_weak_or_negligible
- **stage0_recommendation**: **do_not_enter_stage0**

> **do_not_enter_stage0** — overall delta mIoU ≤ 0 per project gate.

## Overall metrics (excerpt)

| metric | raw | refaware | delta |
|--------|-----|----------|-------|
| mIoU | 0.668132 | 0.667864 | -0.000268 |
| gIoU | 0.668132 | 0.667864 | -0.000268 |
| oIoU | 0.780625 | 0.780967 | 0.000342 |
| cIoU | 0.780625 | 0.780967 | 0.000342 |
| mDice | 0.744682 | 0.744314 | -0.000368 |
| mPrecision | 0.757200 | 0.756067 | -0.001133 |
| mRecall | 0.762184 | 0.761503 | -0.000681 |
| Pr@0.5 | 0.775862 | 0.775287 | -0.000575 |
| box_level_giou | 0.616859 | 0.615029 | -0.001830 |

## Per-prior-group delta mIoU

- **mixed**: n=30, delta_mIoU=0.006671, improved/degraded/stable=5/2/23
- **no_prior**: n=2153, delta_mIoU=-0.001034, improved/degraded/stable=126/115/1912
- **reference_only**: n=95, delta_mIoU=0.001206, improved/degraded/stable=10/6/79
- **target_prior**: n=1202, delta_mIoU=0.000814, improved/degraded/stable=137/94/971

## Sample direction counts

- improved / degraded / stable: **278** / **217** / **2985**
- strong_improved / strong_degraded: **86** / **75**

## Outputs

- raw preds: `outputs/source/refaware_prompt_direction_eval/raw_prompt/test_results`
- refaware preds: `outputs/source/refaware_prompt_direction_eval/refaware_prompt/test_results`
- metrics: `outputs/source/refaware_prompt_direction_eval/metrics`
- per-sample: `outputs/source/refaware_prompt_direction_eval/per_sample_direction_diff.csv`
- summary json: `outputs/source/refaware_prompt_direction_eval/direction_eval_summary.json`
