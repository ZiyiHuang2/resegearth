#!/usr/bin/env python3
"""CPU-only sibling target-swap audit for LaSeRS prediction TIF artifacts.

Does not import torch/CUDA or training entrypoints. Reads predicted masks from
eval output TIFs and GT masks from LaSeRS annotation JSON (RLE decode via pycocotools).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pycocotools.mask as mask_utils
except ImportError as exc:
    print("ERROR: pycocotools is required for RLE mask decode.", file=sys.stderr)
    raise SystemExit(2) from exc

try:
    from tifffile import imread
except ImportError:
    imread = None


PRED_TIF_RE = re.compile(
    r"^(?P<image>[^_]+)_(?P<data_id>\d+)_(?P<split>[^_]+(?:_[^_]+)*)_(?P<mask_id>\d+)\.tif$"
)

LASERS_SPLIT_STEMS = (
    "test_multi_cate",
    "test_single_cate",
    "test_instance_level",
    "test_sematic_level",
    "test_part_level",
    "test_long_query",
    "test_short_query",
    "test_explicit",
    "test_implicit",
    "train",
    "val",
    "test",
)


def parse_pred_filename(name: str) -> Optional[Tuple[str, str, str, int]]:
    stem = Path(name).stem
    for split_stem in LASERS_SPLIT_STEMS:
        token = f"_{split_stem}_"
        if token not in stem:
            continue
        prefix, mask_part = stem.rsplit(token, 1)
        if not mask_part.isdigit():
            continue
        if "_" not in prefix:
            continue
        image, data_id = prefix.rsplit("_", 1)
        if not data_id.isdigit():
            continue
        return image, data_id, split_stem, int(mask_part)

    m = PRED_TIF_RE.match(name)
    if not m:
        return None
    return m.group("image"), m.group("data_id"), m.group("split"), int(m.group("mask_id"))


@dataclass
class SampleAudit:
    image: str
    data_id: str
    mask_id: int
    k: int
    target_iou: float
    max_sibling_iou: float
    mean_sibling_iou: float
    sibling_swap: bool
    pred_path: str


def decode_gt_masks(entry: dict) -> List[np.ndarray]:
    masks = []
    for rle in entry.get("mask", []):
        m = mask_utils.decode(rle)
        if m.ndim == 3:
            m = m[..., 0]
        masks.append((m > 0).astype(np.uint8))
    return masks


def load_pred_mask(path: Path) -> np.ndarray:
    if imread is not None:
        arr = imread(str(path))
    else:
        from PIL import Image

        arr = np.array(Image.open(path))
    arr = np.squeeze(arr)
    return (arr > 0).astype(np.uint8)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / (union + 1e-7)


def build_annotation_index(ann_paths: List[Path]) -> Dict[Tuple[str, str], dict]:
    index: Dict[Tuple[str, str], dict] = {}
    for path in ann_paths:
        with open(path, encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            image_stem = Path(entry["image_name"]).stem
            key = (image_stem, str(entry["id"]))
            index[key] = entry
    return index


def audit_predictions(
    pred_dir: Path,
    ann_paths: List[Path],
    margin: float = 0.05,
    split_filter: Optional[str] = None,
    max_files: Optional[int] = None,
) -> Tuple[dict, List[SampleAudit], List[str]]:
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")

    ann_index = build_annotation_index(ann_paths)
    pred_files = sorted(pred_dir.glob("*.tif"))
    if split_filter:
        pred_files = [p for p in pred_files if split_filter in p.name]
    if max_files:
        pred_files = pred_files[:max_files]

    errors: List[str] = []
    audits: List[SampleAudit] = []
    swap_cases: List[SampleAudit] = []

    for pred_path in pred_files:
        parsed = parse_pred_filename(pred_path.name)
        if parsed is None:
            errors.append(f"Unparseable prediction filename: {pred_path.name}")
            continue
        image, data_id, split_stem, mask_id = parsed
        entry = ann_index.get((image, data_id))
        if entry is None:
            errors.append(f"No annotation for {pred_path.name} key=({image}, {data_id})")
            continue

        gt_masks = decode_gt_masks(entry)
        k = len(gt_masks)
        if k <= 1:
            continue
        if mask_id >= k:
            errors.append(f"mask_id {mask_id} out of range (K={k}) for {pred_path.name}")
            continue

        try:
            pred = load_pred_mask(pred_path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Failed to read {pred_path}: {exc}")
            continue

        target_gt = gt_masks[mask_id]
        if pred.shape != target_gt.shape:
            try:
                import cv2

                pred = cv2.resize(
                    pred,
                    (target_gt.shape[1], target_gt.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                pred = (pred > 0).astype(np.uint8)
            except ImportError:
                errors.append(
                    f"Shape mismatch pred={pred.shape} gt={target_gt.shape} for {pred_path.name} (install opencv-python or match shapes)"
                )
                continue

        target_iou = iou(pred, target_gt)
        sibling_ious = [iou(pred, gt_masks[j]) for j in range(k) if j != mask_id]
        max_sib = max(sibling_ious) if sibling_ious else 0.0
        mean_sib = float(np.mean(sibling_ious)) if sibling_ious else 0.0
        swapped = max_sib > target_iou + margin

        rec = SampleAudit(
            image=image,
            data_id=data_id,
            mask_id=mask_id,
            k=k,
            target_iou=target_iou,
            max_sibling_iou=max_sib,
            mean_sibling_iou=mean_sib,
            sibling_swap=swapped,
            pred_path=str(pred_path),
        )
        audits.append(rec)
        if swapped:
            swap_cases.append(rec)

    total_preds = len(pred_files)
    valid = len(audits)
    summary = {
        "pred_dir": str(pred_dir),
        "annotation_files": [str(p) for p in ann_paths],
        "split_filter": split_filter,
        "margin": margin,
        "total_prediction_files": total_preds,
        "valid_k_gt_1_samples": valid,
        "parse_or_lookup_errors": len(errors),
        "swap_rate": (sum(1 for a in audits if a.sibling_swap) / valid) if valid else None,
        "target_iou_mean": float(np.mean([a.target_iou for a in audits])) if valid else None,
        "sibling_iou_mean": float(np.mean([a.max_sibling_iou for a in audits])) if valid else None,
        "sibling_iou_max": float(max((a.max_sibling_iou for a in audits), default=0.0)),
        "sibling_wins_count": sum(1 for a in audits if a.sibling_swap),
    }
    return summary, audits, errors


def write_artifact_requirements(path: Path, reason: str) -> None:
    content = f"""# LaSeRS binding audit artifact requirements

Generated because audit could not run: **{reason}**

## Required inputs

1. **Prediction directory** (`--pred-dir`): eval output folder containing per-target TIF masks.
   - Filename pattern: `{{image_stem}}_{{data_id}}_{{split_stem}}_{{mask_id}}.tif`
   - Example: `17494_0_test_multi_cate_1.tif`
   - Produced by `segearth_r2/eval/eval.py` (`do_eval`, lines ~293-331).

2. **LaSeRS annotation JSON** (`--ann-dir` or `--ann-files`):
   - Under `{{LaSeRS}}/test/annotations/*.json`
   - Each entry needs: `id`, `image_name`, `mask` (list of COCO RLE dicts)
   - Multi-target samples: `len(mask) == answer.count('[SEG]') > 1`

3. **Optional filters**:
   - `--split-filter test_multi_cate` to restrict to a benchmark slice
   - `--margin 0.05` for sibling_swap threshold

## Outputs

- JSON summary with swap_rate, target/sibling IoU stats
- Optional `--out-cases` listing sibling-win cases

## Not supported (yet)

- Prediction-only JSON without dense masks
- Attention map artifacts (separate script)
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit sibling target-swap on LaSeRS preds (CPU-only)")
    parser.add_argument("--pred-dir", required=True, help="Directory of predicted mask TIF files")
    parser.add_argument("--ann-dir", default="", help="LaSeRS test/annotations directory")
    parser.add_argument("--ann-files", nargs="*", default=[], help="Explicit annotation JSON paths")
    parser.add_argument("--split-filter", default="test_multi_cate", help="Substring filter on TIF names")
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--max-files", type=int, default=0, help="Limit files (0 = all)")
    parser.add_argument("--out-json", default="", help="Write summary JSON here")
    parser.add_argument("--out-cases", default="", help="Write sibling-win cases JSON")
    parser.add_argument("--requirements-md", default="", help="Write artifact requirements on failure")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    ann_paths: List[Path] = [Path(p) for p in args.ann_files]
    if args.ann_dir:
        ann_dir = Path(args.ann_dir)
        ann_paths.extend(sorted(ann_dir.glob("*.json")))
    ann_paths = [p for p in ann_paths if p.is_file()]

    if not ann_paths:
        msg = "No annotation JSON files provided"
        if args.requirements_md:
            write_artifact_requirements(Path(args.requirements_md), msg)
        print(f"ERROR: {msg}", file=sys.stderr)
        raise SystemExit(2)

    try:
        summary, audits, errors = audit_predictions(
            pred_dir,
            ann_paths,
            margin=args.margin,
            split_filter=args.split_filter or None,
            max_files=args.max_files or None,
        )
    except FileNotFoundError as exc:
        if args.requirements_md:
            write_artifact_requirements(Path(args.requirements_md), str(exc))
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    if summary["valid_k_gt_1_samples"] == 0:
        msg = "No valid K>1 samples matched predictions to annotations"
        if args.requirements_md:
            write_artifact_requirements(Path(args.requirements_md), msg)
        print(f"ERROR: {msg}", file=sys.stderr)
        if errors[:5]:
            print("Sample errors:", *errors[:5], sep="\n  ")
        raise SystemExit(2)

    print(json.dumps(summary, indent=2))
    if errors:
        print(f"\nWarnings: {len(errors)} parse/lookup/shape errors (first 3):")
        for e in errors[:3]:
            print(" ", e)

    swap_cases = [asdict(a) for a in audits if a.sibling_swap]
    swap_cases.sort(key=lambda x: x["max_sibling_iou"] - x["target_iou"], reverse=True)

    if args.out_json:
        out = {"summary": summary, "errors_sample": errors[:20], "top_swap_cases": swap_cases[:30]}
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {args.out_json}")

    if args.out_cases:
        Path(args.out_cases).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_cases, "w", encoding="utf-8") as f:
            json.dump(swap_cases, f, indent=2)
        print(f"Wrote {args.out_cases} ({len(swap_cases)} cases)")


if __name__ == "__main__":
    main()
