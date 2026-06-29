#!/usr/bin/env python3
"""Compare short-train gate runs: loss curve, eval, step time, WTI param drift."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)


def _load_log(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _train_loss_series(log_history: List[Dict]) -> List[tuple]:
    rows = []
    for row in log_history:
        if "loss" in row and "step" in row:
            rows.append((int(row["step"]), float(row["loss"])))
    return rows


def _eval_series(log_history: List[Dict]) -> List[tuple]:
    rows = []
    for row in log_history:
        if "eval_score" in row and "step" in row:
            rows.append((int(row["step"]), float(row["eval_score"])))
        elif "eval_giou" in row and "step" in row:
            giou = float(row.get("eval_giou", 0))
            ciou = float(row.get("eval_ciou", 0))
            rows.append((int(row["step"]), 0.5 * giou + 0.5 * ciou))
    return rows


def _step_time_stats(log_history: List[Dict]) -> Optional[Dict[str, float]]:
    times = [float(r["train_runtime"]) / max(float(r.get("train_samples_per_second", 1)), 1e-8)
             for r in log_history if "train_runtime" in r]
    if not times:
        # fallback: derive from consecutive steps if available
        return None
    return {"mean_s": sum(times) / len(times)}


def _summarize_run(name: str, state_path: str) -> Dict[str, Any]:
    st = _load_log(state_path)
    hist = st.get("log_history", [])
    train = _train_loss_series(hist)
    evals = _eval_series(hist)
    out = {
        "name": name,
        "global_step": st.get("global_step"),
        "best_metric": st.get("best_metric"),
        "best_checkpoint": st.get("best_model_checkpoint"),
        "train_loss_first": train[0][1] if train else None,
        "train_loss_last": train[-1][1] if train else None,
        "train_loss_delta": (train[-1][1] - train[0][1]) if len(train) >= 2 else None,
        "eval_points": evals,
        "eval_last": evals[-1][1] if evals else None,
    }
    return out


def _wti_drift(init_path: str, ckpt_path: str) -> Dict[str, float]:
    init_sd = torch.load(init_path, map_location="cpu")
    ckpt_sd = torch.load(ckpt_path, map_location="cpu")
    if "module" in ckpt_sd:
        ckpt_sd = ckpt_sd["module"]
    keys = [k for k in init_sd if k in ckpt_sd and init_sd[k].shape == ckpt_sd[k].shape]
    if not keys:
        return {}
    deltas = []
    per_key = {}
    for k in sorted(keys):
        d = float((ckpt_sd[k].float() - init_sd[k].float()).abs().mean().item())
        per_key[k] = d
        deltas.append(d)
    out = {"mean_abs_delta": sum(deltas) / len(deltas), "max_abs_delta": max(deltas)}
    # highlight mixer/gate/projector keys
    for pat in ("head_mixer_gamma", "alpha", "visual_proj", "text_proj", "head_mixer"):
        pat_keys = [k for k in per_key if pat in k]
        if pat_keys:
            out[f"{pat}_mean"] = sum(per_key[k] for k in pat_keys) / len(pat_keys)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v15-state", required=True)
    parser.add_argument("--v2-state", required=True)
    parser.add_argument("--v15-init", default="")
    parser.add_argument("--v15-ckpt", default="")
    parser.add_argument("--v2-init", default="")
    parser.add_argument("--v2-ckpt", default="")
    args = parser.parse_args()

    v15 = _summarize_run("v1.5+SET++", args.v15_state)
    v2 = _summarize_run("Enhanced-v2+SET++", args.v2_state)

    print("=" * 72)
    print("Short-train gate comparison")
    print("=" * 72)
    for run in (v15, v2):
        print(f"\n[{run['name']}]")
        print(f"  steps={run['global_step']} best_eval={run['best_metric']}")
        print(f"  train loss: {run['train_loss_first']:.4f} -> {run['train_loss_last']:.4f} "
              f"(delta {run['train_loss_delta']:+.4f})" if run['train_loss_first'] is not None else "  train loss: n/a")
        if run["eval_points"]:
            print(f"  eval score trail: {run['eval_points']}")
            print(f"  eval last: {run['eval_last']:.4f}")
        print(f"  best ckpt: {run['best_checkpoint']}")

    if v15["eval_last"] is not None and v2["eval_last"] is not None:
        diff = v2["eval_last"] - v15["eval_last"]
        print(f"\n[delta] Enhanced-v2 eval_last - v1.5 eval_last = {diff:+.4f}")
        if diff < -0.05:
            print("[WARN] Enhanced v2 early eval notably worse than v1.5 (>0.05)")
        else:
            print("[OK] early eval not notably worse than v1.5 baseline")

    for label, init_p, ckpt_p in (
        ("v1.5", args.v15_init, args.v15_ckpt),
        ("Enhanced-v2", args.v2_init, args.v2_ckpt),
    ):
        if init_p and ckpt_p and os.path.isfile(init_p) and os.path.isfile(ckpt_p):
            drift = _wti_drift(init_p, ckpt_p)
            print(f"\n[{label} WTI param drift]")
            for k, v in drift.items():
                print(f"  {k}: {v:.6e}")

    print("\n[PASS] report_short_train_compare")


if __name__ == "__main__":
    main()
