#!/usr/bin/env python3
"""V1: SEG embedding vs taxonomy metrics using stage3 analysis_results (54 samples). CPU-only."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import List, Optional

import numpy as np


def pearson(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 3 or len(x) != len(y):
        return None
    xs, ys = np.array(x, dtype=float), np.array(y, dtype=float)
    if np.std(xs) < 1e-9 or np.std(ys) < 1e-9:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def r2_score(y: List[float], yhat: List[float]) -> Optional[float]:
    if len(y) < 3:
        return None
    yv, pv = np.array(y, dtype=float), np.array(yhat, dtype=float)
    ss_res = np.sum((yv - pv) ** 2)
    ss_tot = np.sum((yv - np.mean(yv)) ** 2)
    if ss_tot < 1e-9:
        return None
    return float(1 - ss_res / ss_tot)


def classify_v1(r_tiou: Optional[float], r_ctx: Optional[float], n: int) -> str:
    if r_tiou is None:
        return "INCONCLUSIVE (zero variance or n<3)"
    if abs(r_tiou) > 0.6:
        return "SEG confusion likely strong — consider SEG refinement before or parallel to CS-DEG"
    if abs(r_tiou) > 0.35:
        return "MODERATE — CS-DEG-core OK but keep SEG refinement as parallel branch"
    return "WEAK — CS-DEG-core can proceed; SEG confusion not dominant on this subset"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-json", required=True)
    parser.add_argument("--join-json", default="", help="Optional output from audit_csdeg_v1_existing_probe_join.json")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    with open(args.probe_json, encoding="utf-8") as f:
        probe = json.load(f)

    rows = []
    for rec in probe:
        seg_sim = rec.get("seg_embedding_sim", {})
        diff = seg_sim.get("diff_category_pairs", {}).get("mean")
        seg_cos = diff if diff is not None and not (isinstance(diff, float) and math.isnan(diff)) else seg_sim.get("pairwise_cos_sim_mean")
        base_iou = rec.get("baseline_iou", {}).get("iou_mean")
        rows.append({
            "sample_id": rec["sample_id"],
            "bucket": rec.get("bucket"),
            "seg_cos": seg_cos,
            "baseline_iou_mean": base_iou,
            "pd_l0": rec.get("pd_linear_probe", {}).get("pd_level_0", {}).get("iou_mean"),
            "gt_phrases": rec.get("gt_phrases"),
        })

    join_map = {}
    if args.join_json and Path(args.join_json).is_file():
        with open(args.join_json, encoding="utf-8") as f:
            j = json.load(f)
        for s in j.get("joined_samples", []):
            tax = s.get("taxonomy") or {}
            join_map[s["sample_id"]] = {
                "target_iou_mean": tax.get("target_iou_mean"),
                "context_leak_ratio_mean": tax.get("context_leak_ratio_mean"),
            }
        for r in rows:
            extra = join_map.get(r["sample_id"], {})
            r.update(extra)

    valid = [r for r in rows if r.get("seg_cos") is not None and r.get("baseline_iou_mean") is not None]
    seg = [float(r["seg_cos"]) for r in valid]
    biou = [float(r["baseline_iou_mean"]) for r in valid]

    ctx_vals = [float(r["context_leak_ratio_mean"]) for r in valid if r.get("context_leak_ratio_mean") is not None]
    seg_ctx = [float(r["seg_cos"]) for r in valid if r.get("context_leak_ratio_mean") is not None]

    # high confusion: seg_cos > median among diff-category pairs
    med = statistics.median(seg) if seg else 0
    high_conf = [r for r in valid if float(r["seg_cos"]) > med]
    low_conf = [r for r in valid if float(r["seg_cos"]) <= med]
    high_success = sum(1 for r in high_conf if float(r["baseline_iou_mean"]) >= 0.3)
    low_success = sum(1 for r in low_conf if float(r["baseline_iou_mean"]) >= 0.3)

    report = {
        "n_probe": len(probe),
        "n_valid": len(valid),
        "seg_cos_median": med,
        "correlations": {
            "corr_seg_cos_baseline_iou": pearson(seg, biou),
            "corr_seg_cos_context_leak": pearson(seg_ctx, ctx_vals) if len(seg_ctx) == len(ctx_vals) else None,
        },
        "high_confusion_group": {
            "n": len(high_conf),
            "success_rate_iou_ge_0.3": high_success / len(high_conf) if high_conf else None,
            "mean_baseline_iou": statistics.mean([float(r["baseline_iou_mean"]) for r in high_conf]) if high_conf else None,
        },
        "low_confusion_group": {
            "n": len(low_conf),
            "success_rate_iou_ge_0.3": low_success / len(low_conf) if low_conf else None,
            "mean_baseline_iou": statistics.mean([float(r["baseline_iou_mean"]) for r in low_conf]) if low_conf else None,
        },
        "v1_classification": classify_v1(pearson(seg, biou), pearson(seg_ctx, ctx_vals) if seg_ctx else None, len(valid)),
        "full_728_export_needed": True,
        "full_728_reason": "54-sample curated probe cannot establish population-level SEG confusion vs context_leak (728 targets).",
        "per_sample": rows,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    md = [
        "# CS-DEG V1 SEG Embedding vs Taxonomy (54-sample)",
        "",
        f"- n_probe: **{report['n_probe']}**",
        f"- corr(seg_cos, baseline_iou): **{report['correlations']['corr_seg_cos_baseline_iou']}**",
        f"- corr(seg_cos, context_leak): **{report['correlations']['corr_seg_cos_context_leak']}**",
        f"- High-confusion success rate (IoU≥0.3): **{report['high_confusion_group']['success_rate_iou_ge_0.3']}** (n={report['high_confusion_group']['n']})",
        f"- Low-confusion success rate: **{report['low_confusion_group']['success_rate_iou_ge_0.3']}** (n={report['low_confusion_group']['n']})",
        "",
        f"**Classification:** {report['v1_classification']}",
        "",
        f"**Full 728 export needed:** {report['full_728_export_needed']} — {report['full_728_reason']}",
    ]
    Path(args.out_md).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"correlations": report["correlations"], "classification": report["v1_classification"]}, indent=2))


if __name__ == "__main__":
    main()
