#!/usr/bin/env python3
"""Post-500 report for DGP v6.1 Stage A from-base."""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from segearth_r2.train.dgp_wandb_monitor import check_stage_a_stop_conditions

BASE_METRICS = "/root/rivermind-data/huangziyi/reseg/output/base/standard-base-siglip1-28w-gd4/rrsisd_test_metrics.json"
BASE_GIOU = 0.6749559358451052


def load_json(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def pct(x):
    return "n/a" if x is None else f"{100 * x:.2f}%"


def qseg_matches_base(qseg_m: dict | None, base_m: dict | None, tol: float = 1e-6) -> tuple[bool, dict]:
    if not qseg_m or not base_m:
        return False, {}
    keys = ["mIoU", "gIoU", "cIoU", "Pr@0.5", "Pr@0.8", "Pr@0.9"]
    diffs = {k: float(qseg_m.get(k, 0)) - float(base_m.get(k, 0)) for k in keys}
    ok = all(abs(v) < tol for v in diffs.values())
    return ok, diffs


def verdict(
    qseg_ok: bool,
    health: dict | None,
    metrics: dict | None,
    base_giou: float = BASE_GIOU,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    ob = (metrics or {}).get("overall_baseline_metrics") or {}
    giou = float((metrics or {}).get("gIoU") or 0)
    delta = giou - base_giou
    bad_flip = ob.get("bad_flip_count")
    rescued = ob.get("rescued_count")

    stop = check_stage_a_stop_conditions(
        health or {},
        eval_metrics={
            "eval_score": giou,
            "base_good_to_model_zero_count": ob.get("base_good_to_model_zero_count", 0),
            "base_good_to_model_zero_rate": ob.get("base_good_to_model_zero_rate", 0),
        },
    )

    if not qseg_ok:
        return "NEED_FIX", ["Q_seg-only metrics drifted from baseline after 500-step training"]

    if stop.get("should_stop"):
        triggers = stop.get("triggers") or []
        return "NEED_FIX", triggers or ["Stage A stop criteria triggered"]

    cos_pg_pl = float((health or {}).get("cos_pg_pl") or 0)
    ent_pl = float((health or {}).get("entropy_pl_attention_norm") or 0)

    if delta >= 0.005 and (bad_flip is None or bad_flip <= 40):
        reasons.append(f"Q_ref gIoU +{delta:.4f} vs base with acceptable bad_flip={bad_flip}")
        if cos_pg_pl < 0.92 and ent_pl >= 0.05:
            reasons.append("health: cos(P_g,P_l) decoupling and P_l entropy not collapsed")
        return "CONTINUE_TO_2K", reasons

    if delta >= 0.002:
        reasons.append(f"modest Q_ref gain (+{delta:.4f}); extend to 2k with monitoring")
        return "CONTINUE_TO_2K", reasons

    if delta < 0:
        reasons.append(f"Q_ref regressed ({delta:.4f})")
        return "STOP", reasons

    reasons.append(f"Q_ref flat (+{delta:.4f}); insufficient signal at 500 steps")
    return "STOP", reasons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tag", default="step500")
    args = parser.parse_args()

    out = args.out_dir
    eval_dir = os.path.join(out, "eval_2k", args.tag)
    metrics_path = os.path.join(eval_dir, "rrsisd_test_metrics_class_group_with_base.json")
    health_path = os.path.join(out, "health_500", f"{args.tag}.json")
    qseg_sanity_path = os.path.join(out, "post_train_qseg_sanity", "rrsisd_test_metrics_qseg_post500.json")

    base_m = load_json(BASE_METRICS)
    qseg_m = load_json(qseg_sanity_path)
    metrics = load_json(metrics_path)
    health = load_json(health_path)

    qseg_ok, qseg_diffs = qseg_matches_base(qseg_m, base_m)
    ob = (metrics or {}).get("overall_baseline_metrics") or {}
    giou = (metrics or {}).get("gIoU")
    cg = (metrics or {}).get("class_group_metrics") or {}
    pc = (metrics or {}).get("per_class_metrics") or {}

    v, rationale = verdict(qseg_ok, health, metrics)

    lines = [
        "# DGP v6.1 Stage A from-base — 500-step report",
        "",
        f"**Verdict:** `{v}`",
        "",
        "## Rationale",
        *[f"- {r}" for r in rationale],
        "",
        "## 1. Q_seg-only sanity vs historical base",
        f"- Match: **{qseg_ok}**",
        f"- Base gIoU: {pct(BASE_GIOU)}",
    ]
    if qseg_m:
        lines.append(f"- Post-train Q_seg gIoU: {pct(qseg_m.get('gIoU'))}")
        for k, d in qseg_diffs.items():
            lines.append(f"  - Δ{k}: {d:+.2e}")
    else:
        lines.append("- Post-train Q_seg metrics: **missing**")

    lines.extend([
        "",
        "## 2. Q_ref model metrics (test)",
        f"- gIoU/mIoU: {pct(giou)}",
        f"- cIoU: {pct((metrics or {}).get('cIoU'))}",
        f"- Pr@0.5: {pct((metrics or {}).get('Pr@0.5'))}",
        f"- Pr@0.8: {pct((metrics or {}).get('Pr@0.8'))}",
        f"- Pr@0.9: {pct((metrics or {}).get('Pr@0.9'))}",
        "",
        "## 3. Q_ref − Q_seg delta (overall)",
        f"- ΔgIoU: {fmt(ob.get('delta_gIoU') or ob.get('delta_mIoU'))}",
        f"- ΔcIoU: {fmt(ob.get('delta_cIoU'))}",
        f"- ΔPr@0.5: {fmt(ob.get('delta_Pr@0.5'))}",
        f"- ΔPr@0.8: {fmt(ob.get('delta_Pr@0.8'))}",
        "",
        "## 4. class_group_metrics",
        "| group | n | mIoU | cIoU | Pr@0.5 | Pr@0.8 | ΔmIoU | bad_flip | rescued |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, row in sorted(cg.items()):
        lines.append(
            f"| {name} | {row.get('n','')} | {fmt(row.get('mIoU'))} | {fmt(row.get('cIoU'))} | "
            f"{fmt(row.get('Pr@0.5'))} | {fmt(row.get('Pr@0.8'))} | {fmt(row.get('delta_mIoU'))} | "
            f"{row.get('bad_flip_count','n/a')} | {row.get('rescued_count','n/a')} |"
        )

    lines.extend([
        "",
        "## 5. per_class_metrics",
        "| class | n | mIoU | cIoU | Pr@0.5 | Pr@0.8 | ΔmIoU | bad_flip | rescued |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, row in sorted(pc.items()):
        lines.append(
            f"| {name} | {row.get('n','')} | {fmt(row.get('mIoU'))} | {fmt(row.get('cIoU'))} | "
            f"{fmt(row.get('Pr@0.5'))} | {fmt(row.get('Pr@0.8'))} | {fmt(row.get('delta_mIoU'))} | "
            f"{row.get('bad_flip_count','n/a')} | {row.get('rescued_count','n/a')} |"
        )

    lines.extend([
        "",
        "## 6. bad_flip / rescued (overall)",
        f"- bad_flip_count: {ob.get('bad_flip_count', 'n/a')}",
        f"- rescued_count: {ob.get('rescued_count', 'n/a')}",
        "",
        "## 7. current_risk / regular_easy delta",
    ])
    for g in ("current_risk", "regular_easy"):
        row = cg.get(g, {})
        lines.append(
            f"- **{g}**: ΔmIoU={fmt(row.get('delta_mIoU'))}, ΔcIoU={fmt(row.get('delta_cIoU'))}, "
            f"bad_flip={row.get('bad_flip_count','n/a')}, rescued={row.get('rescued_count','n/a')}"
        )

    lines.extend([
        "",
        "## 8. Health metrics",
        f"- gate_g: {fmt((health or {}).get('query_refiner_gate_g'))}",
        f"- gate_l: {fmt((health or {}).get('query_refiner_gate_l'))}",
        f"- cos(P_g,P_l): {fmt((health or {}).get('cos_pg_pl'))}",
        f"- entropy_pl_attention_norm: {fmt((health or {}).get('entropy_pl_attention_norm'))}",
        f"- delta_l_norm: {fmt((health or {}).get('delta_l_norm'))}",
        f"- cos_qref_qseg: {fmt((health or {}).get('cos_qref_qseg'))}",
        "",
        "## 9. Conclusion",
        f"**{v}**",
    ])

    report_path = os.path.join(out, "report_500_from_base.md")
    text = "\n".join(lines) + "\n"
    with open(report_path, "w") as f:
        f.write(text)
    print(text)
    print(f"[OK] wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
