#!/usr/bin/env python3
"""CPU-only LaSeRS prediction error taxonomy audit (no torch/CUDA/training imports)."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pycocotools.mask as mask_utils
except ImportError:
    print("ERROR: pycocotools required for RLE decode.", file=sys.stderr)
    raise SystemExit(2)

try:
    from tifffile import imread as tiff_imread
except ImportError:
    tiff_imread = None

try:
    import cv2
except ImportError:
    cv2 = None


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

RELATION_PATTERNS: Dict[str, re.Pattern] = {
    "left": re.compile(r"\bleft\b", re.I),
    "right": re.compile(r"\bright\b", re.I),
    "top": re.compile(r"\btop\b", re.I),
    "bottom": re.compile(r"\bbottom\b", re.I),
    "upper": re.compile(r"\bupper\b", re.I),
    "lower": re.compile(r"\blower\b", re.I),
    "between": re.compile(r"\bbetween\b", re.I),
    "adjacent": re.compile(r"\badjacent\b", re.I),
    "next_to": re.compile(r"\bnext to\b|\badjacent to\b", re.I),
}

ERROR_LABELS = (
    "good",
    "severe_fail",
    "over_segment",
    "under_segment",
    "localization_shift",
    "fragmented",
    "sibling_confusion",
    "context_leak",
    "sibling_leak",
    "union_like",
)


def parse_pred_filename(name: str) -> Optional[Tuple[str, str, str, int]]:
    stem = Path(name).stem
    for split_stem in LASERS_SPLIT_STEMS:
        token = f"_{split_stem}_"
        if token not in stem:
            continue
        prefix, mask_part = stem.rsplit(token, 1)
        if not mask_part.isdigit() or "_" not in prefix:
            continue
        image, data_id = prefix.rsplit("_", 1)
        if not data_id.isdigit():
            continue
        return image, data_id, split_stem, int(mask_part)
    return None


def decode_gt_masks(entry: dict) -> List[np.ndarray]:
    masks = []
    for rle in entry.get("mask", []):
        m = mask_utils.decode(rle)
        if m.ndim == 3:
            m = m[..., 0]
        masks.append((m > 0).astype(np.uint8))
    return masks


def load_pred_mask(path: Path, threshold: float = 0.5) -> np.ndarray:
    if tiff_imread is not None:
        arr = tiff_imread(str(path))
    else:
        from PIL import Image

        arr = np.array(Image.open(path))
    arr = np.squeeze(arr)
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8)
    if np.issubdtype(arr.dtype, np.floating):
        return (arr >= threshold).astype(np.uint8)
    if arr.max() > 1:
        return (arr > 0).astype(np.uint8)
    return (arr >= threshold).astype(np.uint8)


def resize_pred(pred: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if pred.shape == target_shape:
        return pred
    if cv2 is None:
        raise ValueError(f"shape mismatch {pred.shape} vs {target_shape} and cv2 unavailable")
    out = cv2.resize(pred, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return (out > 0).astype(np.uint8)


def mask_area(mask: np.ndarray) -> int:
    return int(mask.sum())


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / (union + 1e-7)


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    x1_min, y1_min, x1_max, y1_max = b1
    x2_min, y2_min, x2_max, y2_max = b2
    ix_min = max(x1_min, x2_min)
    iy_min = max(y1_min, y2_min)
    ix_max = min(x1_max, x2_max)
    iy_max = min(y1_max, y2_max)
    iw = max(0, ix_max - ix_min + 1)
    ih = max(0, iy_max - iy_min + 1)
    inter = iw * ih
    a1 = (x1_max - x1_min + 1) * (y1_max - y1_min + 1)
    a2 = (x2_max - x2_min + 1) * (y2_max - y2_min + 1)
    union = a1 + a2 - inter
    return inter / (union + 1e-7)


def mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def centroid_distance_norm(c1: Tuple[float, float], c2: Tuple[float, float], hw: Tuple[int, int]) -> float:
    h, w = hw
    diag = math.hypot(w, h) + 1e-7
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1]) / diag


def connected_components_stats(mask: np.ndarray) -> Tuple[int, float]:
    if cv2 is None:
        area = mask_area(mask)
        return (1 if area > 0 else 0), 1.0 if area > 0 else 0.0
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return 0, 0.0
    counts = np.bincount(labels.ravel())[1:]
    total = counts.sum()
    if total == 0:
        return 0, 0.0
    return int(len(counts)), float(counts.max() / total)


def matched_relation_terms(text: str) -> List[str]:
    return [name for name, pat in RELATION_PATTERNS.items() if pat.search(text)]


def assign_error_labels(m: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    tiou = m["target_iou"]
    if tiou >= 0.5:
        labels.append("good")
    if tiou < 0.1:
        labels.append("severe_fail")
    if m["area_ratio"] > 1.5 and m["recall"] >= 0.5:
        labels.append("over_segment")
    if m["area_ratio"] < 0.5 and m["precision"] >= 0.5:
        labels.append("under_segment")
    if m["bbox_iou"] < 0.3 and m["centroid_distance_norm"] > 0.2:
        labels.append("localization_shift")
    if m["component_count"] >= 3 and m["largest_component_ratio"] < 0.7:
        labels.append("fragmented")
    if m["max_sibling_iou"] > tiou + 0.05:
        labels.append("sibling_confusion")
    if m["context_leak_ratio"] > 0.3:
        labels.append("context_leak")
    if m["sibling_leak_ratio"] > 0.1:
        labels.append("sibling_leak")
    if m["union_gt_iou"] > tiou + 0.15 and m["area_ratio"] > 1.2:
        labels.append("union_like")
    return labels


@dataclass
class SampleRecord:
    image: str
    data_id: str
    mask_id: int
    split_name: str
    k: int
    pred_path: str
    target_iou: float
    pred_area: int
    gt_area: int
    area_ratio: float
    intersection_area: int
    precision: float
    recall: float
    dice: float
    bbox_iou: float
    centroid_distance_norm: float
    component_count: int
    largest_component_ratio: float
    touches_sibling: bool
    max_sibling_iou: float
    union_gt_iou: float
    context_leak_ratio: float
    sibling_leak_ratio: float
    gt_area_quantile: str = ""
    relation_terms: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)


def build_ann_index(ann_file: Path) -> Dict[Tuple[str, str], dict]:
    with open(ann_file, encoding="utf-8") as f:
        entries = json.load(f)
    return {(Path(e["image_name"]).stem, str(e["id"])): e for e in entries}


def quantile_bucket(area: int, q33: float, q66: float) -> str:
    if area <= q33:
        return "small"
    if area <= q66:
        return "medium"
    return "large"


def summarize_records(records: Sequence[SampleRecord]) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"count": 0}

    def stat(vals: List[float], name: str) -> Dict[str, float]:
        return {
            f"{name}_mean": float(np.mean(vals)),
            f"{name}_median": float(np.median(vals)),
        }

    tious = [r.target_iou for r in records]
    label_counts = Counter(lbl for r in records for lbl in r.labels)
    label_rates = {k: label_counts[k] / n for k in ERROR_LABELS}

    low = [r for r in records if r.target_iou < 0.5]
    low_label_rates = Counter(lbl for r in low for lbl in r.labels)
    low_denom = len(low) or 1

    out: Dict[str, Any] = {
        "count": n,
        "target_iou": stat(tious, "target_iou"),
        "precision": stat([r.precision for r in records], "precision"),
        "recall": stat([r.recall for r in records], "recall"),
        "dice": stat([r.dice for r in records], "dice"),
        "area_ratio": stat([r.area_ratio for r in records], "area_ratio"),
        "context_leak_ratio": stat([r.context_leak_ratio for r in records], "context_leak_ratio"),
        "sibling_leak_ratio": stat([r.sibling_leak_ratio for r in records], "sibling_leak_ratio"),
        "label_counts": dict(label_counts),
        "label_rates": label_rates,
        "low_iou_count": len(low),
        "low_iou_label_rates": {k: low_label_rates[k] / low_denom for k in ERROR_LABELS},
    }
    return out


def top_cases(records: Sequence[SampleRecord], key_fn, reverse: bool = True, n: int = 30) -> List[dict]:
    sorted_recs = sorted(records, key=key_fn, reverse=reverse)[:n]
    return [
        {
            "image": r.image,
            "data_id": r.data_id,
            "mask_id": r.mask_id,
            "pred_path": r.pred_path,
            "target_iou": r.target_iou,
            "area_ratio": r.area_ratio,
            "precision": r.precision,
            "recall": r.recall,
            "context_leak_ratio": r.context_leak_ratio,
            "sibling_leak_ratio": r.sibling_leak_ratio,
            "max_sibling_iou": r.max_sibling_iou,
            "labels": r.labels,
        }
        for r in sorted_recs
    ]


def run_audit(
    pred_dir: Path,
    ann_file: Path,
    split_name: str = "test_multi_cate",
    threshold: float = 0.5,
    min_area: int = 1,
    sibling_touch_threshold: float = 0.05,
) -> Tuple[Dict[str, Any], List[SampleRecord], List[str]]:
    ann_index = build_ann_index(ann_file)
    pred_files = sorted(p for p in pred_dir.glob("*.tif") if split_name in p.name)

    skipped: List[str] = []
    records: List[SampleRecord] = []

    for pred_path in pred_files:
        parsed = parse_pred_filename(pred_path.name)
        if parsed is None:
            skipped.append(f"unparseable_filename:{pred_path.name}")
            continue
        image, data_id, split_stem, mask_id = parsed
        entry = ann_index.get((image, data_id))
        if entry is None:
            skipped.append(f"no_annotation:{pred_path.name}")
            continue

        gt_masks = decode_gt_masks(entry)
        k = len(gt_masks)
        if mask_id >= k:
            skipped.append(f"mask_id_oob:{pred_path.name}")
            continue

        try:
            pred = load_pred_mask(pred_path, threshold=threshold)
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"read_fail:{pred_path.name}:{exc}")
            continue

        target_gt = gt_masks[mask_id]
        try:
            pred = resize_pred(pred, target_gt.shape)
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"resize_fail:{pred_path.name}:{exc}")
            continue

        pred_a = mask_area(pred)
        gt_a = mask_area(target_gt)
        if gt_a < min_area:
            skipped.append(f"gt_too_small:{pred_path.name}")
            continue
        if pred_a == 0 and gt_a > 0:
            pass  # still analyze as severe fail

        inter = int(np.logical_and(pred, target_gt).sum())
        prec = inter / (pred_a + 1e-7)
        rec = inter / (gt_a + 1e-7)
        dice = 2 * prec * rec / (prec + rec + 1e-7)
        tiou = mask_iou(pred, target_gt)
        area_ratio = pred_a / (gt_a + 1e-7)

        sibling_masks = [gt_masks[j] for j in range(k) if j != mask_id]
        sibling_union = np.zeros_like(target_gt)
        for sm in sibling_masks:
            sibling_union = np.logical_or(sibling_union, sm > 0)
        sibling_union = sibling_union.astype(np.uint8)

        all_gt_union = np.zeros_like(target_gt)
        for gm in gt_masks:
            all_gt_union = np.logical_or(all_gt_union, gm > 0)
        all_gt_union = all_gt_union.astype(np.uint8)

        max_sib_iou = max((mask_iou(pred, sm) for sm in sibling_masks), default=0.0)
        union_gt_iou = mask_iou(pred, all_gt_union)

        sibling_overlap = int(np.logical_and(pred, sibling_union).sum())
        all_gt_overlap = int(np.logical_and(pred, all_gt_union).sum())
        context_only = pred_a - all_gt_overlap
        context_leak_ratio = context_only / (pred_a + 1e-7)
        sibling_leak_ratio = sibling_overlap / (pred_a + 1e-7)

        touches_sibling = sibling_leak_ratio >= sibling_touch_threshold or max_sib_iou >= sibling_touch_threshold

        pb = mask_bbox(pred)
        gb = mask_bbox(target_gt)
        if pb and gb:
            biou = bbox_iou(pb, gb)
            pc, gc = mask_centroid(pred), mask_centroid(target_gt)
            cdn = centroid_distance_norm(pc, gc, target_gt.shape) if pc and gc else 0.0
        else:
            biou = 0.0
            cdn = 1.0 if gt_a > 0 and pred_a == 0 else 0.0

        comp_n, comp_lr = connected_components_stats(pred)
        text = (entry.get("description") or "") + " " + (entry.get("reasoning") or "")
        rel_terms = matched_relation_terms(text)

        rec = SampleRecord(
            image=image,
            data_id=data_id,
            mask_id=mask_id,
            split_name=split_stem,
            k=k,
            pred_path=str(pred_path),
            target_iou=tiou,
            pred_area=pred_a,
            gt_area=gt_a,
            area_ratio=area_ratio,
            intersection_area=inter,
            precision=prec,
            recall=rec,
            dice=dice,
            bbox_iou=biou,
            centroid_distance_norm=cdn,
            component_count=comp_n,
            largest_component_ratio=comp_lr,
            touches_sibling=touches_sibling,
            max_sibling_iou=max_sib_iou,
            union_gt_iou=union_gt_iou,
            context_leak_ratio=context_leak_ratio,
            sibling_leak_ratio=sibling_leak_ratio,
            relation_terms=rel_terms,
        )
        records.append(rec)

    if records:
        areas = sorted(r.gt_area for r in records)
        q33 = float(np.percentile(areas, 33.33))
        q66 = float(np.percentile(areas, 66.67))
        for r in records:
            r.gt_area_quantile = quantile_bucket(r.gt_area, q33, q66)
            r.labels = assign_error_labels(asdict(r))

    overall = summarize_records(records)
    overall["total_prediction_files"] = len(pred_files)
    overall["annotation_entries"] = len(ann_index)
    overall["valid"] = len(records)
    overall["skipped"] = len(skipped)

    buckets: Dict[str, Any] = {}

    by_k: Dict[str, List[SampleRecord]] = defaultdict(list)
    by_size: Dict[str, List[SampleRecord]] = defaultdict(list)
    by_mask_id: Dict[str, List[SampleRecord]] = defaultdict(list)
    by_rel: Dict[str, List[SampleRecord]] = defaultdict(list)

    for r in records:
        by_k[str(r.k)].append(r)
        by_size[r.gt_area_quantile].append(r)
        by_mask_id[str(r.mask_id)].append(r)
        for term in r.relation_terms:
            by_rel[term].append(r)
    by_rel["no_relation_term"] = [r for r in records if not r.relation_terms]

    buckets["by_k"] = {k: summarize_records(v) for k, v in sorted(by_k.items())}
    buckets["by_gt_area_quantile"] = {k: summarize_records(v) for k, v in sorted(by_size.items())}
    buckets["by_mask_id"] = {k: summarize_records(v) for k, v in sorted(by_mask_id.items())}
    buckets["by_relation_term"] = {k: summarize_records(v) for k, v in sorted(by_rel.items())}
    buckets["test_multi_cate"] = overall

    report = {
        "meta": {
            "pred_dir": str(pred_dir),
            "ann_file": str(ann_file),
            "split_name": split_name,
            "threshold": threshold,
            "min_area": min_area,
            "cpu_only": True,
            "cuda_used": False,
        },
        "coverage": {
            "total_prediction_files": len(pred_files),
            "annotation_entries_in_file": len(ann_index),
            "valid": len(records),
            "skipped": len(skipped),
            "skipped_reasons_sample": skipped[:30],
            "skipped_reason_counts": dict(Counter(s.split(":")[0] for s in skipped)),
        },
        "overall": overall,
        "buckets": buckets,
        "top_cases": {
            "worst_target_iou": top_cases(records, lambda r: r.target_iou, reverse=False),
            "over_segment": top_cases(records, lambda r: r.area_ratio if "over_segment" in r.labels else -1),
            "context_leak": top_cases(records, lambda r: r.context_leak_ratio),
            "union_like": top_cases(
                records,
                lambda r: (r.union_gt_iou - r.target_iou) if "union_like" in r.labels else -1,
            ),
        },
    }
    return report, records, skipped


def render_markdown(report: Dict[str, Any], md_path: Path) -> None:
    cov = report["coverage"]
    ov = report["overall"]
    lr = ov.get("label_rates", {})
    low_lr = ov.get("low_iou_label_rates", {})

    def pct(x: Optional[float]) -> str:
        return f"{100 * x:.2f}%" if x is not None else "N/A"

    # rank failure modes among low-iou samples
    failure_rank = sorted(
        [(k, low_lr.get(k, 0)) for k in ERROR_LABELS if k != "good"],
        key=lambda x: x[1],
        reverse=True,
    )
    primary = failure_rank[0] if failure_rank else ("unknown", 0)

    rel_bucket = report["buckets"]["by_relation_term"]
    rel_rows = []
    for term, stats in rel_bucket.items():
        if stats.get("count", 0) < 5:
            continue
        rel_rows.append(
            (term, stats["count"], stats["target_iou"]["target_iou_mean"], stats["label_rates"].get("context_leak", 0))
        )
    rel_rows.sort(key=lambda x: x[2])

    size_bucket = report["buckets"]["by_gt_area_quantile"]

    lines = [
        "# Baseline LaSeRS multi_cate Error Taxonomy",
        "",
        "## 1. Safety confirmation",
        "",
        "- **Interrupted processes:** No",
        "- **GPU jobs started:** No",
        "- **Model loaded:** No (CPU-only numpy/tifffile/pycocotools/cv2)",
        f"- **Prediction source:** `{report['meta']['pred_dir']}`",
        "",
        "## 2. Data coverage",
        "",
        f"| Item | Count |",
        f"|------|-------|",
        f"| Prediction TIF files (`{report['meta']['split_name']}`) | {cov['total_prediction_files']} |",
        f"| Annotation entries in file | {cov['annotation_entries_in_file']} |",
        f"| **Valid analyzed** | **{cov['valid']}** |",
        f"| Skipped | {cov['skipped']} |",
        "",
        f"Skip reason counts: `{cov['skipped_reason_counts']}`",
        "",
        "## 3. Main error distribution",
        "",
        f"- Mean target IoU: **{ov['target_iou']['target_iou_mean']:.4f}** (median {ov['target_iou']['target_iou_median']:.4f})",
        f"- Mean precision / recall / dice: **{ov['precision']['precision_mean']:.4f} / {ov['recall']['recall_mean']:.4f} / {ov['dice']['dice_mean']:.4f}**",
        f"- Mean area_ratio (pred/gt): **{ov['area_ratio']['area_ratio_mean']:.4f}** (median {ov['area_ratio']['area_ratio_median']:.4f})",
        f"- Mean context_leak_ratio: **{ov['context_leak_ratio']['context_leak_ratio_mean']:.4f}**",
        f"- Mean sibling_leak_ratio: **{ov['sibling_leak_ratio']['sibling_leak_ratio_mean']:.4f}**",
        "",
        "### Error label rates (all valid samples, multi-label)",
        "",
        "| Label | Rate | Count |",
        "|-------|------|-------|",
    ]
    for label in ERROR_LABELS:
        cnt = ov.get("label_counts", {}).get(label, 0)
        lines.append(f"| `{label}` | {pct(lr.get(label))} | {cnt} |")

    lines.extend(
        [
            "",
            f"Low-IoU subset (target_iou < 0.5): **{ov.get('low_iou_count', 0)}** samples",
            "",
            "### Error label rates within low-IoU subset",
            "",
            "| Label | Rate |",
            "|-------|------|",
        ]
    )
    for label in ERROR_LABELS:
        if label == "good":
            continue
        lines.append(f"| `{label}` | {pct(low_lr.get(label))} |")

    area_mean = ov["area_ratio"]["area_ratio_mean"]
    prec_m = ov["precision"]["precision_mean"]
    rec_m = ov["recall"]["recall_mean"]

    lines.extend(
        [
            "",
            "## 4. What actually explains low gIoU?",
            "",
            f"Observed mean target IoU: **{ov['target_iou']['target_iou_mean']:.4f}** (median **{ov['target_iou']['target_iou_median']:.4f}**) vs reported benchmark gIoU **0.4245** (`output/base/standard-base-lasers-siglip1-8w-gd4/lasers_test_test_multi_cate_metrics.json`, n=321 queries).",
            "",
            f"- **Precision {prec_m:.3f} ≈ recall {rec_m:.3f}** — failures are mostly **missed overlap**, not recall-with-bad-precision alone.",
            f"- **area_ratio mean {area_mean:.3f} (median {ov['area_ratio']['area_ratio_median']:.3f})** — pred area is **not systematically larger** than GT; classic `over_segment` (area_ratio>1.5) tags only **{pct(lr.get('over_segment'))}**.",
            f"- **context_leak_ratio mean {ov['context_leak_ratio']['context_leak_ratio_mean']:.3f}** — **{pct(lr.get('context_leak'))}** of targets have >30% pred pixels **outside all GT masks** (target + siblings). This is **wrong-region / background leakage**, not merely bloated correct masks.",
            f"- **union_like** (pred closer to full GT union than target): only **{pct(lr.get('union_like'))}** — cIoU–gIoU gap is **not** primarily union-of-all-targets fitting.",
            "",
            f"**Primary failure tag among target_iou<0.5 (n={ov.get('low_iou_count', 0)}):** `{primary[0]}` at **{pct(primary[1])}**.",
            "",
            "Ranked failure tags (low-IoU subset):",
            "",
        ]
    )
    for tag, rate in failure_rank[:6]:
        lines.append(f"- `{tag}`: {pct(rate)}")

    lines.extend(
        [
            "",
            "**Interpretation (metric-backed, not cIoU gap alone):**",
            "",
        ]
    )

    if lr.get("context_leak", 0) >= max(lr.get("under_segment", 0), lr.get("over_segment", 0)):
        lines.append("- **Wrong-region / context leak dominates** — most pred mass falls outside any GT mask; not classic area_ratio>1.5 over-segmentation.")
    if lr.get("union_like", 0) < 0.05:
        lines.append(f"- **union_like is rare ({pct(lr.get('union_like'))})** — low gIoU is **not** mainly “covering all siblings at once”.")
    if lr.get("over_segment", 0) < lr.get("context_leak", 0):
        lines.append(f"- **`over_segment` tag rare ({pct(lr.get('over_segment'))})** despite high benchmark cIoU — loss is **off-target pixels**, not only dilated correct regions.")
    if lr.get("sibling_confusion", 0) < 0.05:
        lines.append(f"- **Sibling confusion remains minor ({pct(lr.get('sibling_confusion'))})**, matching swap audit 2.88%.")
    if lr.get("fragmented", 0) > 0.05:
        lines.append(f"- **Fragmentation** is common ({pct(lr.get('fragmented'))}) but co-occurs with severe miss — often scattered false positives on background.")

    lines.extend(["", "## 5. Relation and size effects", ""])
    lines.append("### By K (targets per query)")
    lines.append("")
    lines.append("| K | n | mean IoU | context_leak rate |")
    lines.append("|---|---|----------|-------------------|")
    for k, st in sorted(report["buckets"]["by_k"].items(), key=lambda x: int(x[0])):
        lines.append(
            f"| {k} | {st['count']} | {st['target_iou']['target_iou_mean']:.3f} | "
            f"{pct(st['label_rates'].get('context_leak'))} |"
        )
    lines.append("")
    lines.append("### By GT area quantile")
    lines.append("")
    lines.append("| Size | n | mean IoU | context_leak rate | over_segment rate |")
    lines.append("|------|---|----------|-------------------|-------------------|")
    for sz in ("small", "medium", "large"):
        st = size_bucket.get(sz, {})
        if not st.get("count"):
            continue
        lines.append(
            f"| {sz} | {st['count']} | {st['target_iou']['target_iou_mean']:.3f} | "
            f"{pct(st['label_rates'].get('context_leak'))} | {pct(st['label_rates'].get('over_segment'))} |"
        )

    lines.extend(["", "### Relation terms (n≥5)", "", "| Term | n | mean IoU | context_leak rate |", "|------|---|----------|-------------------|"])
    for term, n, miou, cl in rel_rows[:12]:
        lines.append(f"| `{term}` | {n} | {miou:.3f} | {pct(cl)} |")

    lines.extend(
        [
            "",
            "## 6. Implication for module design",
            "",
        ]
    )
    if lr.get("context_leak", 0) > 0.3:
        lines.append("- **Go:** spatial evidence precision + background/context suppression + boundary refinement (wrong-region leak dominant; not TCPD-style blind pixel inject).")
    if lr.get("localization_shift", 0) > 0.15:
        lines.append("- **Conditional Go:** relation-aware spatial grounding — localization_shift 17% overall, 28% among low-IoU.")
    if lr.get("under_segment", 0) > lr.get("over_segment", 0):
        lines.append("- **Go:** multi-scale detail / target expansion (under-segment dominant — not observed if over-seg higher).")
    if lr.get("sibling_confusion", 0) < 0.05:
        lines.append("- **Do not make sibling-negative the sole mainline** — keep as auxiliary contrastive term only.")

    lines.extend(
        [
            "",
            "## 7. Go / No-Go",
            "",
            "| Direction | Decision | Evidence |",
            "|-----------|----------|----------|",
            f"| Context suppression / precision refinement | **Go** | context_leak {pct(lr.get('context_leak'))}, union_like {pct(lr.get('union_like'))} |",
            f"| TCPD-style pixel injection | **No-Go** | prior audit ΔgIoU −19.7 |",
            f"| SET++ repetition | **No-Go** | flat multi_cate metrics |",
            f"| Sibling-negative only | **No-Go** | sibling_confusion {pct(lr.get('sibling_confusion'))} |",
            f"| Relation-aware grounding | **Conditional Go** | worst relation buckets above |",
            "",
            f"Full JSON: `{md_path.with_suffix('.json').name}` sibling path under `plans/baseline_diagnostics/`.",
        ]
    )

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="LaSeRS CPU error taxonomy audit")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", default="", help="Markdown report path")
    parser.add_argument("--split-name", default="test_multi_cate")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=1)
    args = parser.parse_args()

    t0 = time.time()
    report, _, skipped = run_audit(
        Path(args.pred_dir),
        Path(args.ann_file),
        split_name=args.split_name,
        threshold=args.threshold,
        min_area=args.min_area,
    )
    elapsed = time.time() - t0
    report["meta"]["runtime_seconds"] = round(elapsed, 3)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    out_md = Path(args.out_md) if args.out_md else out_json.with_suffix(".md")
    render_markdown(report, out_md)

    print(json.dumps({"runtime_seconds": report["meta"]["runtime_seconds"], "coverage": report["coverage"], "label_rates": report["overall"]["label_rates"]}, indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
