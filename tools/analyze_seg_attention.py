#!/usr/bin/env python3
import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze [SEG] -> image token attention audit dumps.")
    parser.add_argument("--audit-dir", type=str, required=True, help="Directory containing attention_audit_step*.pt")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for csv summaries")
    parser.add_argument("--area-small", type=float, default=0.05, help="Small bin upper bound (ratio)")
    parser.add_argument("--area-medium", type=float, default=0.20, help="Medium bin upper bound (ratio)")
    parser.add_argument("--export-attn-dir", type=str, default=None, help="Optional per-sample attention export dir")
    return parser.parse_args()


def load_audit_files(audit_dir):
    files = sorted(glob.glob(os.path.join(audit_dir, "**", "*.pt"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .pt files found in {audit_dir}")
    return files


def area_bin(area_ratio, small_thr, medium_thr):
    if area_ratio < small_thr:
        return "small"
    if area_ratio < medium_thr:
        return "medium"
    return "large"


def topk_iou(attn_vec, fg_mask):
    fg_mask = fg_mask.astype(np.bool_)
    fg_count = int(fg_mask.sum())
    if fg_count <= 0:
        return np.nan
    k = min(max(fg_count, 1), attn_vec.shape[0])
    topk_idx = np.argpartition(attn_vec, -k)[-k:]
    pred = np.zeros_like(fg_mask, dtype=np.bool_)
    pred[topk_idx] = True
    inter = np.logical_and(pred, fg_mask).sum()
    union = np.logical_or(pred, fg_mask).sum()
    return float(inter / (union + 1e-8))


def masked_mean(values, mask):
    if mask.sum() == 0:
        return np.nan
    return float(values[mask].mean())


def safe_mean(values):
    if len(values) == 0:
        return np.nan
    arr = np.array(values, dtype=np.float64)
    if np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmean(arr))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.export_attn_dir:
        os.makedirs(args.export_attn_dir, exist_ok=True)

    files = load_audit_files(args.audit_dir)

    rows = []
    grouped = defaultdict(lambda: {"iou": [], "fg_bg": [], "bo": []})

    for file_path in files:
        payload = torch.load(file_path, map_location="cpu")
        layer_attn = payload.get("seg_to_image_attn", [])
        fg_mask = payload.get("fg_mask", None)
        boundary_map = payload.get("boundary_map", None)
        outer_ring_map = payload.get("outer_ring_map", None)
        meta = payload.get("meta", [])

        if fg_mask is None or len(layer_attn) == 0:
            continue

        fg_mask = fg_mask.detach().float().cpu().numpy().astype(np.bool_)
        boundary_np = boundary_map.detach().float().cpu().numpy().astype(np.bool_) if boundary_map is not None else None
        outer_np = outer_ring_map.detach().float().cpu().numpy().astype(np.bool_) if outer_ring_map is not None else None
        sample_count = fg_mask.shape[0]

        for sample_idx in range(sample_count):
            sample_meta = meta[sample_idx] if sample_idx < len(meta) else {}
            sample_fg = fg_mask[sample_idx]
            sample_bg = ~sample_fg
            sample_area = float(sample_fg.mean())
            sample_bin = area_bin(sample_area, args.area_small, args.area_medium)

            for layer_idx, layer_tensor in enumerate(layer_attn):
                layer_np = layer_tensor.detach().float().cpu().numpy()
                if sample_idx >= layer_np.shape[0]:
                    continue
                head_count = layer_np.shape[1]
                for head_idx in range(head_count):
                    attn_vec = layer_np[sample_idx, head_idx]
                    iou = topk_iou(attn_vec, sample_fg)
                    fg_mean = masked_mean(attn_vec, sample_fg)
                    bg_mean = masked_mean(attn_vec, sample_bg)
                    fg_bg_contrast = fg_mean - bg_mean if not np.isnan(fg_mean) and not np.isnan(bg_mean) else np.nan

                    bo_contrast = np.nan
                    if boundary_np is not None and outer_np is not None:
                        b_mean = masked_mean(attn_vec, boundary_np[sample_idx])
                        o_mean = masked_mean(attn_vec, outer_np[sample_idx])
                        if not np.isnan(b_mean) and not np.isnan(o_mean):
                            bo_contrast = b_mean - o_mean

                    row = {
                        "file": os.path.basename(file_path),
                        "image_id": sample_meta.get("image_id", ""),
                        "data_id": sample_meta.get("data_id", ""),
                        "mask_id": sample_meta.get("mask_id", ""),
                        "area_ratio": sample_area,
                        "size_bin": sample_bin,
                        "layer": layer_idx,
                        "head": head_idx,
                        "iou_topk": iou,
                        "fg_bg_contrast": fg_bg_contrast,
                        "boundary_outer_contrast": bo_contrast,
                    }
                    rows.append(row)

                    key_all = ("all", layer_idx, head_idx)
                    key_bin = (sample_bin, layer_idx, head_idx)
                    grouped[key_all]["iou"].append(iou)
                    grouped[key_all]["fg_bg"].append(fg_bg_contrast)
                    grouped[key_all]["bo"].append(bo_contrast)
                    grouped[key_bin]["iou"].append(iou)
                    grouped[key_bin]["fg_bg"].append(fg_bg_contrast)
                    grouped[key_bin]["bo"].append(bo_contrast)

                    if args.export_attn_dir:
                        export_name = (
                            f"{os.path.splitext(os.path.basename(file_path))[0]}"
                            f"_sample{sample_idx:03d}_layer{layer_idx:02d}_head{head_idx:02d}.npy"
                        )
                        np.save(os.path.join(args.export_attn_dir, export_name), attn_vec)

    detail_csv = os.path.join(args.output_dir, "seg_attention_detail.csv")
    summary_csv = os.path.join(args.output_dir, "seg_attention_summary.csv")

    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "image_id",
                "data_id",
                "mask_id",
                "area_ratio",
                "size_bin",
                "layer",
                "head",
                "iou_topk",
                "fg_bg_contrast",
                "boundary_outer_contrast",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for (bin_name, layer_idx, head_idx), values in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        summary_rows.append(
            {
                "size_bin": bin_name,
                "layer": layer_idx,
                "head": head_idx,
                "count": len(values["iou"]),
                "iou_topk_mean": safe_mean(values["iou"]),
                "fg_bg_contrast_mean": safe_mean(values["fg_bg"]),
                "boundary_outer_contrast_mean": safe_mean(values["bo"]),
            }
        )

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "size_bin",
                "layer",
                "head",
                "count",
                "iou_topk_mean",
                "fg_bg_contrast_mean",
                "boundary_outer_contrast_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved detail: {detail_csv}")
    print(f"Saved summary: {summary_csv}")


if __name__ == "__main__":
    main()
