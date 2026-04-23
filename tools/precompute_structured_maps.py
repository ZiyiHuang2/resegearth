#!/usr/bin/env python3
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute boundary/outer ring maps from binary masks.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing binary mask files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save structured maps")
    parser.add_argument("--pattern", type=str, default="**/*.png", help="Glob pattern under input-dir")
    return parser.parse_args()


def load_mask(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        mask = np.load(path)
    else:
        mask = np.array(Image.open(path))
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    return (mask > 0).astype(np.uint8)


def build_structured_maps(mask_np):
    fg = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
    dilated = (F.max_pool2d(fg, kernel_size=3, stride=1, padding=1) > 0).float()
    eroded = (1.0 - F.max_pool2d(1.0 - fg, kernel_size=3, stride=1, padding=1)).clamp(min=0.0, max=1.0)
    boundary = (fg - eroded).clamp(min=0.0, max=1.0)
    outer_ring = (dilated - fg).clamp(min=0.0, max=1.0)
    return boundary.squeeze().numpy().astype(np.uint8), outer_ring.squeeze().numpy().astype(np.uint8)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern), recursive=True))
    if len(files) == 0:
        raise FileNotFoundError(f"No masks found with pattern={args.pattern} under {args.input_dir}")

    meta_records = []
    for path in files:
        rel = os.path.relpath(path, args.input_dir)
        stem = os.path.splitext(rel)[0]
        stem_base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(args.output_dir, f"{stem}.npz")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        mask = load_mask(path)
        boundary_map, outer_ring_map = build_structured_maps(mask)
        np.savez_compressed(
            out_path,
            boundary_map=boundary_map,
            outer_ring_map=outer_ring_map,
            fg_map=mask.astype(np.uint8),
        )
        record = {
            "input_mask": path,
            "output_npz": out_path,
            "structured_map_key": stem_base,
            "height": int(mask.shape[0]),
            "width": int(mask.shape[1]),
            "fg_ratio": float(mask.mean()),
        }
        meta_records.append(record)

    meta_path = os.path.join(args.output_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_records, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(meta_records)} structured map files.")
    print(f"Saved meta: {meta_path}")


if __name__ == "__main__":
    main()
