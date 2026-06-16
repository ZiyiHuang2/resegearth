#!/usr/bin/env python3
"""Generate DGP Stage A 2k validation report from health probes + test eval metrics."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime


def load_json(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def health_row(h: dict | None) -> dict:
    if not h:
        return {}
    return {
        "cos_pg_pl": h.get("cos_pg_pl"),
        "entropy_pl_norm": h.get("entropy_pl_attention_norm"),
        "gate_g": h.get("query_refiner_gate_g"),
        "gate_l": h.get("query_refiner_gate_l"),
        "cos_qref_qseg": h.get("cos_qref_qseg"),
        "delta_l_norm": h.get("delta_l_norm"),
    }


def metrics_summary(m: dict | None) -> dict:
    if not m:
        return {}
    ob = m.get("overall_baseline_metrics") or m.get("overall_baseline") or {}
    score = m.get("gIoU") if m.get("gIoU") is not None else m.get("mIoU")
    base_score = ob.get("base_gIoU") if ob.get("base_gIoU") is not None else ob.get("base_mIoU")
    return {
        "eval_score": score,
        "mdice": m.get("mDice"),
        "mrecall": m.get("mRecall"),
        "mprecision": m.get("mPrecision"),
        "base_score": base_score,
        "bad_flip": ob.get("bad_flip_count"),
        "rescued": ob.get("rescued_count"),
        "delta_giou": ob.get("delta_gIoU") or ob.get("delta_mIoU"),
    }


def class_deltas(m: dict | None, classes: list[str]) -> dict:
    if not m:
        return {}
    per = m.get("per_class_metrics", {})
    out = {}
    for c in classes:
        key = c.lower().replace("-", "").replace("_", "")
        row = {}
        for k, v in per.items():
            if k.lower().replace("-", "").replace("_", "") == key:
                row = v
                break
        out[c] = {
            "model_score": row.get("mIoU"),
            "base_score": row.get("base_mIoU"),
            "delta": row.get("delta_mIoU"),
            "bad_flip": row.get("bad_flip_count"),
        }
    return out


def verdict(health_500, health_1000, health_final, m1000, mfinal) -> tuple[str, str]:
    reasons = []
    h500 = health_row(health_500)
    hf = health_row(health_final)
    cos_ok = (h500.get("cos_pg_pl") or 1) > (hf.get("cos_pg_pl") or 0) + 0.05 or (hf.get("cos_pg_pl") or 0) < 0.85
    ent = hf.get("entropy_pl_norm")
    if ent is not None and ent < 0.05:
        reasons.append("P_l attention entropy collapsed at final checkpoint")
    s1000 = metrics_summary(m1000).get("eval_score")
    sf = metrics_summary(mfinal).get("eval_score")
    base = 0.0132
    if s1000 and s1000 > base * 1.5:
        reasons.append(f"test eval_score improved at 1000 ({s1000:.4f} vs base {base:.4f})")
    if sf and s1000 and sf >= s1000 * 0.95:
        reasons.append("final checkpoint maintains test gain vs 1000")
    elif sf and s1000 and sf < s1000 * 0.9:
        reasons.append("test score regressed after step 1000")

    bad = metrics_summary(mfinal).get("bad_flip")
    if bad is not None and bad <= 2:
        reasons.append(f"bad_flip low ({bad})")

    if sf and sf > base * 1.8 and (hf.get("cos_pg_pl") or 1) < 0.88 and (ent or 0) > 0.05:
        return "CONTINUE_TO_5K", "; ".join(reasons) or "metrics and health acceptable"
    if sf and sf > base * 1.2:
        return "CONTINUE_TO_5K", "; ".join(reasons) or "modest test gain; extend training"
    if (ent or 0) < 0.05 or (hf.get("cos_pg_pl") or 0) > 0.95:
        return "NEED_FIX", "; ".join(reasons) or "attention collapse or no P_g/P_l decoupling"
    return "STOP", "; ".join(reasons) or "insufficient test gain for extended training"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = args.out_dir

    health = {
        "500": load_json(os.path.join(out, "health_2k/step500.json")),
        "1000": load_json(os.path.join(out, "health_2k/step1000.json")),
        "1300": load_json(os.path.join(out, "health_2k/step1300.json")),
    }
    m1000 = load_json(os.path.join(out, "eval_2k/step1000/rrsisd_test_metrics_class_group_with_base.json"))
    m1300 = load_json(os.path.join(out, "eval_2k/step1300/rrsisd_test_metrics_class_group_with_base.json"))

    # val eval from training log (quick reference)
    val_1000 = 0.02709
    val_500 = 0.01559

    problem = ["bridge", "vehicle", "ship", "tenniscourt", "Expressway-toll-station", "chimney"]
    v, rationale = verdict(health["500"], health["1000"], health["1300"], m1000, m1300)

    lines = [
        "# DGP v6.1 Stage A — 2k Validation Report (early stop ~1400)",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Run summary",
        "",
        "- **Output dir:** `" + out + "`",
        "- **Planned:** 2000 steps from checkpoint-500",
        "- **Actual:** training stopped at **~1400 steps** (SIGTERM); latest saved checkpoint **1300**",
        "- **Stage:** A only (prompt_adapter + query_refiner, 14 trainable params)",
        "- **Val eval (in-training):** step 500 eval_score≈0.0156, step 1000 eval_score≈0.0271",
        "",
        "## Health probe (8 train batches, offline)",
        "",
        "| Step | cos(P_g,P_l) | entropy_pl_norm | gate_g | gate_l | cos(Q_ref,Q_seg) | delta_l_norm |",
        "|------|-------------|-----------------|--------|--------|------------------|--------------|",
    ]
    for step in ("500", "1000", "1300"):
        r = health_row(health[step])
        label = step if step != "1300" else "1300 (~1400 stop)"
        lines.append(
            f"| {label} | {fmt(r.get('cos_pg_pl'))} | {fmt(r.get('entropy_pl_norm'))} | "
            f"{fmt(r.get('gate_g'))} | {fmt(r.get('gate_l'))} | {fmt(r.get('cos_qref_qseg'))} | {fmt(r.get('delta_l_norm'))} |"
        )

    lines += [
        "",
        "## RRSISD test metrics (full eval, gIoU = mIoU)",
        "",
        "| Step | gIoU | mDice | mRecall | mPrecision | bad_flip | rescued | base_gIoU | ΔgIoU |",
        "|------|------|-------|---------|------------|----------|---------|-----------|-------|",
    ]
    for step, m in [("1000", m1000), ("1300", m1300)]:
        s = metrics_summary(m)
        lines.append(
            f"| {step} | {fmt(s.get('eval_score'))} | {fmt(s.get('mdice'))} | {fmt(s.get('mrecall'))} | "
            f"{fmt(s.get('mprecision'))} | {fmt(s.get('bad_flip'), 0)} | {fmt(s.get('rescued'), 0)} | "
            f"{fmt(s.get('base_score'))} | {fmt(s.get('delta_giou'), nd=4)} |"
        )

    if not health["500"]:
        health["500"] = {
            "cos_pg_pl": 0.806,
            "entropy_pl_attention_norm": 0.781,
            "query_refiner_gate_g": 0.026,
            "query_refiner_gate_l": 0.037,
            "cos_qref_qseg": 0.993,
            "_note": "from 500-step diagnostic probe (checkpoint deleted)",
        }

    lines += ["", "## Problem classes (test, step 1300)", ""]
    cd = class_deltas(m1300, problem)
    if cd:
        lines.append("| Class | model_score | base_score | delta |")
        lines.append("|-------|-------------|------------|-------|")
        for c, row in cd.items():
            lines.append(f"| {c} | {fmt(row.get('model_score'))} | {fmt(row.get('base_score'))} | {fmt(row.get('delta'))} |")
    else:
        lines.append("_Per-class metrics pending full eval completion._")

    lines += [
        "",
        "## Verdict",
        "",
        f"**{v}**",
        "",
        rationale,
        "",
        "### Notes",
        "",
        "- Trainer `health/cos_*` and gate logs unreliable under DeepSpeed ZeRO; offline probe used.",
        "- `stage_a/stop_pl_attn_collapse` may fire when health dict empty during training.",
        "- Untrained DGP baseline test eval_score ≈ 0.0132.",
        "- v5 full pipeline reference: `output/dgp/dgp-qdti-stage3-v5/rrsisd_test_metrics_class_group_with_base.json`.",
    ]

    report_path = os.path.join(out, "report_2k.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {report_path}")
    print(f"Verdict: {v}")


if __name__ == "__main__":
    main()
