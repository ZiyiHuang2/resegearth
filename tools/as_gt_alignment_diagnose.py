#!/usr/bin/env python3
"""
Quantitative A_S vs GT alignment diagnostics (debug outputs only).
Does not load overlay PNGs; does not modify model code.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

VARIANTS = ("last_layer_as", "mean_layer_as", "loss_style_as")


def _load_gt_binary(path: str) -> np.ndarray:
    im = np.array(Image.open(path).convert("L"))
    return (im > 127).astype(np.float32)


def _centroid_2d(mask: np.ndarray) -> Tuple[float, float]:
    """(cy, cx) in pixel coordinates."""
    ys, xs = np.where(mask > 0.5)
    if ys.size == 0:
        return float("nan"), float("nan")
    return float(ys.mean()), float(xs.mean())


def _map_centroid_to_grid(cy: float, cx: float, h: int, w: int) -> Tuple[int, int]:
    """Nearest grid cell (row, col) in 0..26 for 27x27."""
    gy = (cy + 0.5) / h * 27.0 - 0.5
    gx = (cx + 0.5) / w * 27.0 - 0.5
    r = int(round(np.clip(gy, 0, 26)))
    c = int(round(np.clip(gx, 0, 26)))
    return r, c


def _downsample_gt_to_27(gt_hw: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Returns gt27 float in {0,1}, and gt_disappeared_after_downsample.
    """
    h, w = gt_hw.shape
    small = cv2.resize(gt_hw, (27, 27), interpolation=cv2.INTER_AREA)
    gt27 = (small >= 0.5).astype(np.float32)
    disappeared = bool(gt27.sum() < 1e-6)
    if disappeared:
        cy, cx = _centroid_2d(gt_hw)
        if math.isnan(cy):
            gt27 = np.zeros((27, 27), dtype=np.float32)
        else:
            r, c = _map_centroid_to_grid(cy, cx, h, w)
            gt27 = np.zeros((27, 27), dtype=np.float32)
            gt27[r, c] = 1.0
    return gt27, disappeared


def _dilate27(mask: np.ndarray) -> np.ndarray:
    """One grid-cell dilation (8-neighborhood) on 27x27."""
    m = (mask > 0.5).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    d = cv2.dilate(m, kernel, iterations=1)
    return (d > 0).astype(np.float32)


def _normalize_as(as_flat: np.ndarray) -> np.ndarray:
    x = as_flat.astype(np.float64).reshape(-1)
    x = x - x.min()
    s = x.sum()
    if s <= 0 or not np.isfinite(s):
        raise ValueError("A_S is constant or invalid after subtracting min; cannot normalize.")
    p = x / s
    return p.reshape(27, 27)


def _gt_centroid_grid(gt27: np.ndarray) -> Tuple[float, float]:
    """Continuous (row, col) centroid on 27x27 grid."""
    ys, xs = np.where(gt27 > 0.5)
    if ys.size == 0:
        return 13.0, 13.0
    return float(ys.mean()), float(xs.mean())


def _cell_center(r: int, c: int) -> Tuple[float, float]:
    return r + 0.5, c + 0.5


def _dist_grid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _topk_indices(p: np.ndarray, k: int) -> List[Tuple[int, int]]:
    flat = p.reshape(-1)
    idx = np.argsort(-flat)[:k]
    out = []
    for ii in idx:
        r, c = divmod(int(ii), 27)
        out.append((r, c))
    return out


def _border_cell(r: int, c: int) -> bool:
    return r == 0 or r == 26 or c == 0 or c == 26


def _entropy(p: np.ndarray) -> float:
    p = p.reshape(-1).astype(np.float64)
    p = np.clip(p, 1e-30, 1.0)
    return float(-(p * np.log(p)).sum())


def compute_row(
    p: np.ndarray,
    gt27: np.ndarray,
    gt_dil: np.ndarray,
    gt_centroid_rc: Tuple[float, float],
) -> Dict[str, object]:
    flat = p.reshape(-1)
    top1 = int(np.argmax(flat))
    r1, c1 = divmod(top1, 27)

    top1_hit = bool(gt27[r1, c1] > 0.5)
    top5 = _topk_indices(p, 5)
    top10 = _topk_indices(p, 10)
    top5_hit = any(gt27[r, c] > 0.5 for r, c in top5)
    top10_hit = any(gt27[r, c] > 0.5 for r, c in top10)

    mass_in_gt = float((p * gt27).sum())
    mass_in_dil = float((p * gt_dil).sum())

    c1_pos = _cell_center(r1, c1)
    d_top1 = _dist_grid(c1_pos, gt_centroid_rc)

    d5_min = min(_dist_grid(_cell_center(r, c), gt_centroid_rc) for r, c in top5)
    border_ct = sum(1 for r, c in top5 if _border_cell(r, c))
    ent = _entropy(p)

    return {
        "top1_hit_gt": top1_hit,
        "top5_hit_gt": top5_hit,
        "top10_hit_gt": top10_hit,
        "mass_in_gt": mass_in_gt,
        "mass_in_dilated_gt": mass_in_dil,
        "top1_distance_to_gt_centroid": d_top1,
        "top5_min_distance_to_gt_centroid": d5_min,
        "border_top5_count": border_ct,
        "entropy": ent,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-dir",
        type=str,
        default="/home/wangchengjun/huangziyi/reseg/segearth+cross/outputs/debug_as_quality/test_10_small",
    )
    ap.add_argument(
        "--output-csv",
        type=str,
        default=None,
    )
    args = ap.parse_args()
    in_dir = os.path.abspath(args.input_dir)
    out_csv = args.output_csv or os.path.join(in_dir, "as_gt_alignment_metrics.csv")
    summary_path = os.path.join(in_dir, "summary.jsonl")

    if not os.path.isfile(summary_path):
        print(f"[error] missing {summary_path}", file=sys.stderr)
        sys.exit(1)

    meta_rows: List[dict] = []
    with open(summary_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                meta_rows.append(json.loads(line))
    meta_rows.sort(key=lambda r: r["sample_id"])

    csv_rows: List[dict] = []
    by_variant: Dict[str, List[dict]] = defaultdict(list)

    for meta in meta_rows:
        sid = meta["sample_id"]
        pred_iou = meta["pred_iou"]
        fg = meta["foreground_ratio"]
        gt_path = os.path.join(in_dir, f"{sid}_gt_mask.png")
        if not os.path.isfile(gt_path):
            print(f"[error] missing GT: {gt_path}", file=sys.stderr)
            sys.exit(1)

        gt_hw = _load_gt_binary(gt_path)
        gt27, disappeared = _downsample_gt_to_27(gt_hw)
        gt_dil = _dilate27(gt27)
        gc = _gt_centroid_grid(gt27)

        for var in VARIANTS:
            npy_path = os.path.join(in_dir, f"{sid}_{var}.npy")
            if not os.path.isfile(npy_path):
                print(f"[error] missing npy: {npy_path}", file=sys.stderr)
                sys.exit(1)
            as_flat = np.load(npy_path)
            if as_flat.size != 729:
                print(f"[error] expected 729 values, got {as_flat.size}: {npy_path}", file=sys.stderr)
                sys.exit(1)
            p = _normalize_as(as_flat)
            m = compute_row(p, gt27, gt_dil, gc)
            row = {
                "sample_id": sid,
                "variant": var,
                "pred_iou": pred_iou,
                "foreground_ratio": fg,
                "top1_hit_gt": m["top1_hit_gt"],
                "top5_hit_gt": m["top5_hit_gt"],
                "top10_hit_gt": m["top10_hit_gt"],
                "mass_in_gt": m["mass_in_gt"],
                "mass_in_dilated_gt": m["mass_in_dilated_gt"],
                "top1_distance_to_gt_centroid": m["top1_distance_to_gt_centroid"],
                "top5_min_distance_to_gt_centroid": m["top5_min_distance_to_gt_centroid"],
                "border_top5_count": m["border_top5_count"],
                "entropy": m["entropy"],
                "gt_disappeared_after_downsample": disappeared,
            }
            csv_rows.append(row)
            by_variant[var].append(row)

    fields = [
        "sample_id",
        "variant",
        "pred_iou",
        "foreground_ratio",
        "top1_hit_gt",
        "top5_hit_gt",
        "top10_hit_gt",
        "mass_in_gt",
        "mass_in_dilated_gt",
        "top1_distance_to_gt_centroid",
        "top5_min_distance_to_gt_centroid",
        "border_top5_count",
        "entropy",
        "gt_disappeared_after_downsample",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in csv_rows:
            w.writerow(row)

    print(f"Wrote {out_csv}")
    print()
    print("=== Per-variant summary (numeric only) ===")
    for var in VARIANTS:
        rs = by_variant[var]
        n = len(rs)
        r_top1 = sum(1 for r in rs if r["top1_hit_gt"]) / n
        r_top5 = sum(1 for r in rs if r["top5_hit_gt"]) / n
        r_top10 = sum(1 for r in rs if r["top10_hit_gt"]) / n
        mm_gt = float(np.mean([r["mass_in_gt"] for r in rs]))
        mm_d = float(np.mean([r["mass_in_dilated_gt"] for r in rs]))
        md1 = float(np.mean([r["top1_distance_to_gt_centroid"] for r in rs]))
        mb = float(np.mean([r["border_top5_count"] for r in rs]))
        me = float(np.mean([r["entropy"] for r in rs]))
        print(f"variant={var}  n={n}")
        print(f"  top1_hit_gt_rate={r_top1:.4f}")
        print(f"  top5_hit_gt_rate={r_top5:.4f}")
        print(f"  top10_hit_gt_rate={r_top10:.4f}")
        print(f"  mean_mass_in_gt={mm_gt:.6f}")
        print(f"  mean_mass_in_dilated_gt={mm_d:.6f}")
        print(f"  mean_top1_distance_to_gt_centroid={md1:.6f}")
        print(f"  mean_border_top5_count={mb:.4f}")
        print(f"  mean_entropy={me:.6f}")
        print()

    # Risk notes (numeric thresholds only)
    print("=== Risk notes (numeric thresholds; no visual judgment) ===")
    border_means = {var: float(np.mean([r["border_top5_count"] for r in by_variant[var]])) for var in VARIANTS}
    mass_means = {var: float(np.mean([r["mass_in_gt"] for r in by_variant[var]])) for var in VARIANTS}
    mass_d_means = {var: float(np.mean([r["mass_in_dilated_gt"] for r in by_variant[var]])) for var in VARIANTS}

    border_thr = 2.0
    mass_thr = 0.02
    mass_d_thr = 0.04
    pred_thr = 0.5
    mass_weak_thr = 0.03

    for var in VARIANTS:
        if border_means[var] >= border_thr:
            print(
                f"[{var}] mean_border_top5_count={border_means[var]:.4f} >= {border_thr}: "
                "top-5 中边界格点占比较高，可能存在固定边界热点（数值现象，非视觉结论）。"
            )
        if mass_means[var] < mass_thr and mass_d_means[var] < mass_d_thr:
            print(
                f"[{var}] mean_mass_in_gt={mass_means[var]:.6f} (<{mass_thr}) 且 "
                f"mean_mass_in_dilated_gt={mass_d_means[var]:.6f} (<{mass_d_thr}): "
                "A_S 归一化后落在 GT（及 1 格膨胀）内的 mass 偏低，与 GT 对齐弱（数值结论）。"
            )

    # "A_S 不准但 pred 准": pred_iou 高，但三种 variant 的 mass_in_gt 均值仍低
    risk_samples: List[str] = []
    for meta in meta_rows:
        sid = meta["sample_id"]
        if meta["pred_iou"] < pred_thr:
            continue
        migs = [r["mass_in_gt"] for r in csv_rows if r["sample_id"] == sid and r["variant"] in VARIANTS]
        mds = [r["mass_in_dilated_gt"] for r in csv_rows if r["sample_id"] == sid and r["variant"] in VARIANTS]
        if len(migs) != 3:
            continue
        mean_mig = float(np.mean(migs))
        mean_md = float(np.mean(mds))
        if mean_mig < mass_weak_thr and mean_md < mass_d_thr * 1.5:
            risk_samples.append(sid)
    if risk_samples:
        print(
            f"pred_iou>={pred_thr} 且三 variant 平均 mass_in_gt<{mass_weak_thr}、"
            f"平均 mass_in_dilated_gt<{mass_d_thr * 1.5} 的 sample_id（A_S 与 GT 对齐弱但 pred 较高）: {sorted(set(risk_samples))}"
        )
        for sid in sorted(set(risk_samples)):
            meta = next(m for m in meta_rows if m["sample_id"] == sid)
            migs = [r["mass_in_gt"] for r in csv_rows if r["sample_id"] == sid]
            mds = [r["mass_in_dilated_gt"] for r in csv_rows if r["sample_id"] == sid]
            print(
                f"  detail: {sid} pred_iou={meta['pred_iou']} "
                f"mean_mass_in_gt={float(np.mean(migs)):.6f} mean_mass_in_dilated_gt={float(np.mean(mds)):.6f}"
            )
    else:
        print(
            f"未命中“pred_iou>={pred_thr} 且三 variant 平均 mass 偏低”组合；无“A_S 不准但 pred 准”风险列表。"
        )


if __name__ == "__main__":
    main()
