#!/usr/bin/env python3
"""Check if context_leak pixels overlap masks in non-multi_cate LaSeRS test annotations."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pycocotools.mask as mask_utils
except ImportError:
    raise SystemExit("pycocotools required")

try:
    from tifffile import imread as tiff_imread
except ImportError:
    tiff_imread = None


def load_pred(path: Path) -> np.ndarray:
    if tiff_imread is not None:
        arr = np.squeeze(tiff_imread(str(path)))
    else:
        from PIL import Image
        arr = np.array(Image.open(path))
    return (np.squeeze(arr) > 0).astype(np.uint8)


def parse_pred(name: str) -> Optional[Tuple[str, str, int]]:
    stem = Path(name).stem
    token = "_test_multi_cate_"
    if token not in stem:
        return None
    prefix, mask_part = stem.rsplit(token, 1)
    if not mask_part.isdigit():
        return None
    image, data_id = prefix.rsplit("_", 1)
    return image, data_id, int(mask_part)


def decode_masks(entry: dict) -> List[np.ndarray]:
    out = []
    for rle in entry.get("mask", []):
        m = mask_utils.decode(rle)
        if m.ndim == 3:
            m = m[..., 0]
        out.append((m > 0).astype(np.uint8))
    return out


def load_all_annotations(ann_dir: Path) -> Tuple[Dict[str, dict], Dict[str, np.ndarray]]:
    """image_stem -> list of entries; image_stem -> union mask from ALL test json files."""
    by_image: Dict[str, List[dict]] = defaultdict(list)
    union_by_image: Dict[str, np.ndarray] = {}
    for path in sorted(ann_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            stem = Path(entry["image_name"]).stem
            by_image[stem].append(entry)
    for stem, entries in by_image.items():
        union = None
        for e in entries:
            for m in decode_masks(e):
                union = m if union is None else np.logical_or(union, m > 0).astype(np.uint8)
        if union is not None:
            union_by_image[stem] = union.astype(np.uint8)
    return by_image, union_by_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy-json", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--ann-dir", required=True)
    parser.add_argument("--multi-cate-ann", required=True)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    with open(args.taxonomy_json, encoding="utf-8") as f:
        tax = json.load(f)
    cases = tax.get("top_cases", {}).get("context_leak", [])
    cases = sorted(cases, key=lambda x: x.get("context_leak_ratio", 0), reverse=True)[: args.top_n]

    with open(args.multi_cate_ann, encoding="utf-8") as f:
        mc_ann = {str(x["id"]): x for x in json.load(f)}

    _, full_union = load_all_annotations(Path(args.ann_dir))
    pred_dir = Path(args.pred_dir)

    results = []
    inflated = 0
    for c in cases:
        pred_path = Path(c["pred_path"])
        if not pred_path.is_file():
            pred_path = pred_dir / Path(c["pred_path"]).name
        parsed = parse_pred(pred_path.name)
        if parsed is None:
            continue
        image, data_id, mask_id = parsed
        entry = mc_ann.get(data_id)
        if entry is None:
            continue
        gt_masks = decode_masks(entry)
        target = gt_masks[mask_id]
        mc_union = np.zeros_like(target)
        for gm in gt_masks:
            mc_union = np.logical_or(mc_union, gm > 0)
        mc_union = mc_union.astype(np.uint8)

        full = full_union.get(image)
        if full is None or full.shape != target.shape:
            continue

        pred = load_pred(pred_path)
        if pred.shape != target.shape:
            import cv2
            pred = cv2.resize(pred, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_NEAREST)
            pred = (pred > 0).astype(np.uint8)

        pred_a = float(pred.sum())
        mc_overlap = float(np.logical_and(pred, mc_union).sum())
        full_overlap = float(np.logical_and(pred, full).sum())
        context_mc = pred_a - mc_overlap
        context_full = pred_a - full_overlap
        inflation_ratio = (context_mc - context_full) / (context_mc + 1e-7)
        possibly_inflated = inflation_ratio > 0.15 and context_mc > 0

        if possibly_inflated:
            inflated += 1
        results.append({
            **{k: c[k] for k in ["image", "data_id", "mask_id", "context_leak_ratio", "target_iou", "pred_path"]},
            "pred_area": pred_a,
            "context_leak_mc": context_mc / (pred_a + 1e-7),
            "context_leak_full_ann": context_full / (pred_a + 1e-7),
            "inflation_ratio": inflation_ratio,
            "possibly_inflated": possibly_inflated,
        })

    n = len(results)
    rate = inflated / n if n else 0
    verdict = "RELIABLE" if rate < 0.15 else "DISCOUNT context_leak"
    report = {
        "top_n_requested": args.top_n,
        "cases_analyzed": n,
        "possibly_inflated_count": inflated,
        "inflation_rate": rate,
        "verdict": verdict,
        "cases": results,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md = [
        "# CS-DEG Annotation Completeness Spot-Check",
        "",
        f"- Top context_leak cases analyzed: **{n}**",
        f"- Possibly inflated (leak into other-benchmark GT): **{inflated}** ({100*rate:.1f}%)",
        f"- **Verdict:** {verdict}",
        "",
        "Method: compare pred pixels outside multi_cate GT union vs outside union of **all** test annotation masks on same image.",
    ]
    Path(args.out_md).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"inflation_rate": rate, "verdict": verdict, "n": n}, indent=2))


if __name__ == "__main__":
    main()
