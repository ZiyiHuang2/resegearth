#!/usr/bin/env python3
"""CPU-only: join stage3 probe (54) with taxonomy/logit dump. No model imports."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pycocotools.mask as mask_utils
except ImportError:
    print("ERROR: pycocotools required", file=sys.stderr)
    raise SystemExit(2)

try:
    from tifffile import imread as tiff_imread
except ImportError:
    tiff_imread = None


def load_pred_mask(path: Path) -> np.ndarray:
    if tiff_imread is not None:
        arr = np.squeeze(tiff_imread(str(path)))
    else:
        from PIL import Image
        arr = np.array(Image.open(path))
    return (np.squeeze(arr) > 0).astype(np.uint8)

LASERS_SPLIT_STEMS = ("test_multi_cate",)


def parse_pred_filename(name: str) -> Optional[Tuple[str, str, int]]:
    stem = Path(name).stem
    for split_stem in LASERS_SPLIT_STEMS:
        token = f"_{split_stem}_"
        if token not in stem:
            continue
        prefix, mask_part = stem.rsplit(token, 1)
        if not mask_part.isdigit() or "_" not in prefix:
            continue
        image, data_id = prefix.rsplit("_", 1)
        if data_id.isdigit():
            return image, data_id, int(mask_part)
    return None


def pearson(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 3 or len(y) < 3 or len(x) != len(y):
        return None
    xs = np.array(x, dtype=float)
    ys = np.array(y, dtype=float)
    if np.std(xs) < 1e-9 or np.std(ys) < 1e-9:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def safe_mean(vals: List[float]) -> Optional[float]:
    return float(statistics.mean(vals)) if vals else None


def load_probe(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_logit_by_sample_id(path: Path) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sid = int(r["sample_id"])
            out[sid] = r
    return out


def decode_masks(entry: dict) -> List[np.ndarray]:
    masks = []
    for rle in entry.get("mask", []):
        m = mask_utils.decode(rle)
        if m.ndim == 3:
            m = m[..., 0]
        masks.append((m > 0).astype(np.uint8))
    return masks


def compute_target_metrics(pred_dir: Path, ann_by_id: Dict[str, dict], data_ids: List[str]) -> Dict[str, dict]:
    """Per query (data_id) aggregate target_iou and context_leak_ratio."""
    per_id: Dict[str, List[dict]] = defaultdict(list)
    for p in pred_dir.glob("*test_multi_cate*.tif"):
        parsed = parse_pred_filename(p.name)
        if parsed is None:
            continue
        image, data_id, mask_id = parsed
        if data_id not in data_ids:
            continue
        entry = ann_by_id.get(data_id)
        if entry is None:
            continue
        gt_masks = decode_masks(entry)
        if mask_id >= len(gt_masks):
            continue
        if tiff_imread is None and True:
            pass
        pred = load_pred_mask(p)
        target_gt = gt_masks[mask_id]
        if pred.shape != target_gt.shape:
            import cv2
            pred = cv2.resize(pred, (target_gt.shape[1], target_gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            pred = (pred > 0).astype(np.uint8)
        inter = float(np.logical_and(pred, target_gt).sum())
        union = float(np.logical_or(pred, target_gt).sum())
        tiou = inter / (union + 1e-7)
        pred_a = float(pred.sum())
        all_gt = np.zeros_like(target_gt)
        for gm in gt_masks:
            all_gt = np.logical_or(all_gt, gm > 0)
        all_gt = all_gt.astype(np.uint8)
        all_overlap = float(np.logical_and(pred, all_gt).sum())
        ctx_leak = (pred_a - all_overlap) / (pred_a + 1e-7)
        per_id[data_id].append({"target_iou": tiou, "context_leak_ratio": ctx_leak, "mask_id": mask_id})
    agg = {}
    for did, rows in per_id.items():
        agg[did] = {
            "n_targets": len(rows),
            "target_iou_mean": safe_mean([r["target_iou"] for r in rows]),
            "context_leak_ratio_mean": safe_mean([r["context_leak_ratio"] for r in rows]),
        }
    return agg


def extract_probe_fields(rec: dict) -> dict:
    seg_sim = rec.get("seg_embedding_sim", {})
    diff_cat = seg_sim.get("diff_category_pairs", {}).get("mean")
    if diff_cat is None or (isinstance(diff_cat, float) and math.isnan(diff_cat)):
        seg_cos = seg_sim.get("pairwise_cos_sim_mean")
    else:
        seg_cos = diff_cat
    pd0 = rec.get("pd_linear_probe", {}).get("pd_level_0", {}).get("iou_mean")
    base_iou = rec.get("baseline_iou", {}).get("iou_mean")
    return {
        "sample_id": rec["sample_id"],
        "bucket": rec.get("bucket"),
        "seg_cos": seg_cos,
        "pairwise_cos_mean": seg_sim.get("pairwise_cos_sim_mean"),
        "diff_category_cos_mean": diff_cat,
        "pd_l0_iou": pd0,
        "probe_baseline_iou_mean": base_iou,
        "gt_phrases": rec.get("gt_phrases"),
        "num_masks": rec.get("num_masks"),
    }


def render_md(report: dict, path: Path) -> None:
    j = report["join_summary"]
    c = report["correlations"]
    lines = [
        "# CS-DEG V1 Existing Probe Join",
        "",
        "## Probe fields (analysis_results.json)",
        "",
        f"- Records: **{j['probe_records']}**",
        f"- Fields present: `{', '.join(report['probe_field_inventory'])}`",
        f"- Has sample_id: **{report['field_checks']['has_sample_id']}**",
        f"- Has seg_embedding_sim.diff_category_pairs.mean: **{report['field_checks']['has_diff_category_cos']}**",
        f"- Has baseline_iou: **{report['field_checks']['has_baseline_iou']}**",
        f"- Has pd_linear_probe.pd_level_0: **{report['field_checks']['has_pd_l0']}**",
        "",
        "## Join results",
        "",
        f"| Join type | Matched | Failed |",
        f"|-----------|---------|--------|",
        f"| Probe → taxonomy (by data_id={j['join_key']}) | {j['taxonomy_joined']} | {j['taxonomy_join_failures']} |",
        f"| Probe → logit dump (by sample_id) | {j['logit_joined']} | {j['logit_join_failures']} |",
        "",
        "## Correlations (n={})".format(j["correlation_n"]),
        "",
    ]
    for name, val in c.items():
        lines.append(f"- `{name}`: **{val if val is not None else 'N/A (zero variance)'}**")
    lines.extend(["", "## Verdict", "", report["verdict"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-json", required=True)
    parser.add_argument("--taxonomy-json", required=True)
    parser.add_argument("--logit-jsonl", required=True)
    parser.add_argument("--logit-summary", default="")
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    probe = load_probe(Path(args.probe_json))
    with open(args.ann_file, encoding="utf-8") as f:
        ann_list = json.load(f)
    ann_by_id = {str(x["id"]): x for x in ann_list}
    logit_map = load_logit_by_sample_id(Path(args.logit_jsonl))

    probe_fields = extract_probe_fields(probe[0]) if probe else {}
    field_inventory = list(probe[0].keys()) if probe else []

    data_ids = [str(x["sample_id"]) for x in probe]
    tax_agg = compute_target_metrics(Path(args.pred_dir), ann_by_id, set(data_ids))

    joined: List[dict] = []
    tax_fail = []
    logit_fail = []
    for rec in probe:
        pf = extract_probe_fields(rec)
        did = str(rec["sample_id"])
        tax = tax_agg.get(did)
        if tax is None:
            tax_fail.append({"sample_id": did, "reason": "no_pred_tif_or_ann"})
        logit = logit_map.get(int(rec["sample_id"]))
        if logit is None:
            logit_fail.append({"sample_id": did, "reason": "missing_in_logit_jsonl"})
        row = {**pf, "taxonomy": tax, "logit_merged_iou": logit.get("merged_mask_IoU") if logit else None}
        joined.append(row)

    corr_rows = [r for r in joined if r.get("taxonomy") and r.get("seg_cos") is not None]
    seg_cos = [float(r["seg_cos"]) for r in corr_rows]
    tiou = [float(r["taxonomy"]["target_iou_mean"]) for r in corr_rows]
    ctx = [float(r["taxonomy"]["context_leak_ratio_mean"]) for r in corr_rows]
    pd0 = [float(r["pd_l0_iou"]) for r in corr_rows if r.get("pd_l0_iou") is not None]
    pd0_tiou = [float(r["taxonomy"]["target_iou_mean"]) for r in corr_rows if r.get("pd_l0_iou") is not None]
    pd0_ctx = [float(r["taxonomy"]["context_leak_ratio_mean"]) for r in corr_rows if r.get("pd_l0_iou") is not None]

    correlations = {
        "corr_seg_cos_target_iou": pearson(seg_cos, tiou),
        "corr_seg_cos_context_leak": pearson(seg_cos, ctx),
        "corr_pd_l0_target_iou": pearson(pd0, pd0_tiou),
        "corr_pd_l0_context_leak": pearson(pd0, pd0_ctx),
    }

    # bucket stats
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for r in corr_rows:
        buckets[r.get("bucket") or "unknown"].append(r)
    bucket_stats = {}
    for b, rows in buckets.items():
        bucket_stats[b] = {
            "n": len(rows),
            "seg_cos_mean": safe_mean([float(x["seg_cos"]) for x in rows]),
            "target_iou_mean": safe_mean([float(x["taxonomy"]["target_iou_mean"]) for x in rows]),
            "context_leak_mean": safe_mean([float(x["taxonomy"]["context_leak_ratio_mean"]) for x in rows]),
            "pd_l0_mean": safe_mean([float(x["pd_l0_iou"]) for x in rows if x.get("pd_l0_iou") is not None]),
        }

    # V1 substitute verdict
    r_seg_tiou = correlations["corr_seg_cos_target_iou"]
    r_seg_ctx = correlations["corr_seg_cos_context_leak"]
    if r_seg_tiou is None:
        verdict = "INSUFFICIENT: cannot compute seg_cos vs target_iou correlation (n too small or zero variance)."
    elif abs(r_seg_tiou) > 0.6 and (r_seg_ctx is not None and abs(r_seg_ctx) > 0.5):
        verdict = (
            "NOT sufficient to replace full V1 as Go signal alone: moderate-strong seg_cos correlation "
            f"(r={r_seg_tiou:.3f}) on n=54 suggests SEG refinement may help, but sample size is too small."
        )
    elif abs(r_seg_tiou) < 0.35:
        verdict = (
            f"Partial support for CS-DEG-core: weak seg_cos↔target_iou correlation (r={r_seg_tiou:.3f}) on n=54; "
            "54-sample join cannot replace full 728-target V1 but does not indicate SEG confusion as sole blocker."
        )
    else:
        verdict = (
            f"INCONCLUSIVE for full V1 replacement: moderate seg_cos↔target_iou r={r_seg_tiou:.3f} on n=54 only. "
            "Recommend optional 728-target embedding export before implementation."
        )

    e3_e4 = {
        "E3_seg_embedding_probe_proves": "On 54 curated samples, SEG pairwise cosine differs by bucket; representation-level phrase/SEG separation is imperfect on bad buckets.",
        "E3_cannot_prove": "Cannot generalize to all 728 targets; cannot prove SEG confusion causes context_leak at population level without full join.",
        "E4_probe_e_proves": "Probe E verdict FAIL: SEG/phrase spatial top-100 alignment to GT is near random on bad buckets.",
        "E4_cannot_prove": "Does not test decoder-internal evidence loop feasibility (that's V2); uses fixed random projection not CS-DEG heads.",
        "logit_dump_proves": "Strong logit margin but low raw IoU on multi_cate — calibration/post-threshold not primary issue.",
        "logit_dump_cannot_prove": "Does not isolate SEG embedding vs spatial evidence formation inside decoder.",
    }

    report = {
        "probe_field_inventory": field_inventory,
        "field_checks": {
            "has_sample_id": all("sample_id" in r for r in probe),
            "has_diff_category_cos": all(
                r.get("seg_embedding_sim", {}).get("diff_category_pairs") is not None for r in probe
            ),
            "has_baseline_iou": all("baseline_iou" in r for r in probe),
            "has_pd_l0": all(
                r.get("pd_linear_probe", {}).get("pd_level_0") is not None for r in probe
            ),
        },
        "join_key": "sample_id == LaSeRS annotation id (data_id)",
        "taxonomy_json_limitation": "baseline_multi_cate_error_taxonomy.json stores aggregates/top_cases only; per-query metrics recomputed from pred TIF for probe sample_ids.",
        "join_summary": {
            "probe_records": len(probe),
            "taxonomy_joined": len(probe) - len(tax_fail),
            "taxonomy_join_failures": len(tax_fail),
            "logit_joined": len(probe) - len(logit_fail),
            "logit_join_failures": len(logit_fail),
            "correlation_n": len(corr_rows),
            "join_key": "sample_id",
        },
        "join_failures": {"taxonomy": tax_fail[:10], "logit": logit_fail[:10]},
        "correlations": correlations,
        "bucket_stats": bucket_stats,
        "joined_samples": joined,
        "e3_e4_interpretation": e3_e4,
        "verdict": verdict,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    render_md(report, Path(args.out_md))
    print(json.dumps({"join_summary": report["join_summary"], "correlations": correlations, "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
