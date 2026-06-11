#!/usr/bin/env python3
"""
Compare LaSeRS baseline vs A3 frozen SET diagnostic JSONL outputs.

Reads existing per-sample generation diagnostic records only (no re-eval).
All baseline vs SET comparisons use intersection keys: (subset, sample_id).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

TINY_MASK_THRESHOLD = 10
FOCUS_SUBSETS = ("test_multi_cate", "test_long_query", "test_instance_level")
COUNT_CATEGORIES = ("no_seg", "under_generate", "equal_count", "over_generate")


@dataclass
class LoadedDiagnostic:
    records: Dict[Tuple[str, int], dict] = field(default_factory=dict)
    raw_count: int = 0
    duplicate_keys: int = 0
    subset_counts: Counter = field(default_factory=Counter)


def load_jsonl(path: str) -> LoadedDiagnostic:
    out = LoadedDiagnostic()
    seen: Counter = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.raw_count += 1
            key = (rec["subset"], rec["sample_id"])
            seen[key] += 1
            out.subset_counts[rec["subset"]] += 1
            out.records[key] = rec
    out.duplicate_keys = sum(1 for c in seen.values() if c > 1)
    return out


def count_category(rec: dict) -> str:
    gen = rec.get("generated_SEG_count", 0)
    gt = rec.get("gt_mask_count", 0)
    if gen == 0:
        return "no_seg"
    if gen < gt:
        return "under_generate"
    if gen == gt:
        return "equal_count"
    return "over_generate"


def safe_mean(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(statistics.mean(vals))


def safe_median(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(statistics.median(vals))


def merged_iou(rec: dict) -> Optional[float]:
    v = rec.get("merged_mask_IoU")
    if v is None:
        return None
    return float(v)


def has_duplicate_best_match(rec: dict) -> bool:
    idxs = rec.get("best_matching_GT_index") or []
    if len(idxs) <= 1:
        return False
    valid = [i for i in idxs if i is not None and i >= 0]
    return len(valid) != len(set(valid))


def gt_coverage_stats(rec: dict, iou_thresh: float = 0.5) -> Tuple[bool, float]:
    gt_count = rec.get("gt_mask_count", 0)
    if gt_count == 0:
        return True, 1.0
    iou_matrix = rec.get("per_SEG_IoU_with_each_GT") or []
    if not iou_matrix:
        return False, 0.0
    covered = 0
    for gt_j in range(gt_count):
        best = max((row[gt_j] for row in iou_matrix if gt_j < len(row)), default=0.0)
        if best >= iou_thresh:
            covered += 1
    ratio = covered / gt_count
    return covered == gt_count, ratio


def empty_or_tiny_mask_rate(rec: dict) -> Tuple[float, float]:
    areas = rec.get("per_SEG_mask_area") or []
    if not areas:
        return 0.0, 0.0
    empty = sum(1 for a in areas if a == 0) / len(areas)
    tiny = sum(1 for a in areas if a < TINY_MASK_THRESHOLD) / len(areas)
    return empty, tiny


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def fmt_pct(v: Optional[float], digits: int = 1) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}%"


def fmt_float(v: Optional[float], digits: int = 3) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"


def fmt_delta(v: Optional[float], digits: int = 3) -> str:
    if v is None:
        return "N/A"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.{digits}f}"


def filter_records(
    records: Dict[Tuple[str, int], dict],
    keys: Iterable[Tuple[str, int]],
    subset: Optional[str] = None,
) -> List[dict]:
    out = []
    for k in keys:
        if subset is not None and k[0] != subset:
            continue
        out.append(records[k])
    return out


def count_stats(records: List[dict]) -> Dict[str, Any]:
    n = len(records)
    cats = Counter(count_category(r) for r in records)
    return {
        "n": n,
        **{c: cats.get(c, 0) for c in COUNT_CATEGORIES},
        **{f"{c}_rate": pct(cats.get(c, 0), n) for c in COUNT_CATEGORIES},
    }


def iou_stats(records: List[dict]) -> Dict[str, Any]:
    ious = [merged_iou(r) for r in records]
    ious_valid = [x for x in ious if x is not None]
    n = len(ious_valid)
    return {
        "n": len(records),
        "mean_merged_iou": safe_mean(ious_valid),
        "median_merged_iou": safe_median(ious_valid),
        "iou_lt_0.5_rate": pct(sum(1 for x in ious_valid if x < 0.5), n),
        "iou_ge_0.5_rate": pct(sum(1 for x in ious_valid if x >= 0.5), n),
        "iou_ge_0.8_rate": pct(sum(1 for x in ious_valid if x >= 0.8), n),
    }


def multi_cate_extra_stats(records: List[dict]) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {}
    dup_rate = pct(sum(1 for r in records if has_duplicate_best_match(r)), n)
    all_cov = pct(sum(1 for r in records if gt_coverage_stats(r)[0]), n)
    cov_ratios = [gt_coverage_stats(r)[1] for r in records]
    empty_rates = [100.0 * empty_or_tiny_mask_rate(r)[0] for r in records]
    tiny_rates = [100.0 * empty_or_tiny_mask_rate(r)[1] for r in records]

    by_cat: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        by_cat[count_category(r)].append(r)

    cat_iou = {}
    for c in COUNT_CATEGORIES:
        cat_iou[f"{c}_mean_merged_iou"] = safe_mean([merged_iou(r) for r in by_cat[c]])

    return {
        "duplicate_best_match_rate": dup_rate,
        "all_gt_covered_rate": all_cov,
        "avg_covered_gt_ratio": safe_mean(cov_ratios),
        "empty_mask_rate": safe_mean(empty_rates),
        "tiny_mask_rate": safe_mean(tiny_rates),
        **cat_iou,
    }


def assign_label(b_iou: Optional[float], s_iou: Optional[float], delta: Optional[float]) -> str:
    if b_iou is not None and s_iou is not None:
        if b_iou < 0.5 and s_iou >= 0.5:
            return "rescued"
        if b_iou >= 0.5 and s_iou < 0.5:
            return "damaged"
    if delta is not None:
        if delta >= 0.1:
            return "big_gain"
        if delta <= -0.1:
            return "big_loss"
    return "unchanged"


def load_standard_metrics(metrics_dir: str) -> Dict[str, dict]:
    table_path = os.path.join(metrics_dir, "lasers_test_table2_metrics.json")
    metrics: Dict[str, dict] = {}
    if os.path.isfile(table_path):
        with open(table_path, "r", encoding="utf-8") as f:
            for row in json.load(f):
                stem = row.get("stem")
                if stem:
                    metrics[stem] = row
    for fname in os.listdir(metrics_dir):
        if fname.startswith("lasers_test_") and fname.endswith("_metrics.json"):
            stem = fname.replace("lasers_test_", "").replace("_metrics.json", "")
            path = os.path.join(metrics_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                metrics[stem] = json.load(f)
    return metrics


def write_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out = {}
            for k in fieldnames:
                v = row.get(k)
                if isinstance(v, (list, dict)):
                    out[k] = json.dumps(v, ensure_ascii=False)
                else:
                    out[k] = v
            w.writerow(out)


def build_case_rows(
    baseline: LoadedDiagnostic,
    set_data: LoadedDiagnostic,
    keys: List[Tuple[str, int]],
) -> List[dict]:
    rows = []
    for key in keys:
        b = baseline.records[key]
        s = set_data.records[key]
        b_iou = merged_iou(b)
        s_iou = merged_iou(s)
        delta = (s_iou - b_iou) if (b_iou is not None and s_iou is not None) else None
        rows.append({
            "subset": key[0],
            "sample_id": key[1],
            "query": b.get("query", ""),
            "gt_answer": b.get("gt_answer", ""),
            "baseline_model_answer": b.get("model_answer", ""),
            "set_model_answer": s.get("model_answer", ""),
            "gt_mask_count": b.get("gt_mask_count"),
            "baseline_generated_SEG_count": b.get("generated_SEG_count"),
            "set_generated_SEG_count": s.get("generated_SEG_count"),
            "baseline_pred_mask_count": b.get("pred_mask_count"),
            "set_pred_mask_count": s.get("pred_mask_count"),
            "baseline_merged_IoU": b_iou,
            "set_merged_IoU": s_iou,
            "delta_merged_IoU": delta,
            "baseline_best_matching_GT_index": b.get("best_matching_GT_index"),
            "set_best_matching_GT_index": s.get("best_matching_GT_index"),
            "baseline_per_SEG_IoU_with_each_GT": b.get("per_SEG_IoU_with_each_GT"),
            "set_per_SEG_IoU_with_each_GT": s.get("per_SEG_IoU_with_each_GT"),
            "baseline_per_SEG_mask_area": b.get("per_SEG_mask_area"),
            "set_per_SEG_mask_area": s.get("per_SEG_mask_area"),
            "baseline_per_SEG_mask_path": b.get("per_SEG_mask_path"),
            "set_per_SEG_mask_path": s.get("per_SEG_mask_path"),
            "label": assign_label(b_iou, s_iou, delta),
        })
    return rows


def delta_count_rate(set_stats: dict, base_stats: dict, cat: str) -> Optional[float]:
    k = f"{cat}_rate"
    if set_stats.get("n", 0) == 0 or base_stats.get("n", 0) == 0:
        return None
    return set_stats[k] - base_stats[k]


def analyze_long_query_decomposition(
    baseline: LoadedDiagnostic,
    set_data: LoadedDiagnostic,
    keys: List[Tuple[str, int]],
) -> Dict[str, Any]:
    subset_keys = [k for k in keys if k[0] == "test_long_query"]
    rows = []
    for k in subset_keys:
        b, s = baseline.records[k], set_data.records[k]
        b_iou, s_iou = merged_iou(b), merged_iou(s)
        delta = (s_iou - b_iou) if (b_iou is not None and s_iou is not None) else None
        b_cat, s_cat = count_category(b), count_category(s)
        rows.append({
            "delta": delta,
            "count_changed": b_cat != s_cat,
            "b_cat": b_cat,
            "s_cat": s_cat,
            "iou_worse_count_same": b_cat == s_cat and delta is not None and delta < -0.05,
            "over_or_no_seg_increase": s_cat in ("over_generate", "no_seg") and b_cat not in ("over_generate", "no_seg"),
            "b_dup": has_duplicate_best_match(b),
            "s_dup": has_duplicate_best_match(s),
            "b_cov": gt_coverage_stats(b)[1],
            "s_cov": gt_coverage_stats(s)[1],
        })

    worsened = [r for r in rows if r["delta"] is not None and r["delta"] < 0]
    n = len(rows)
    return {
        "n": n,
        "mean_delta_iou": safe_mean([r["delta"] for r in rows if r["delta"] is not None]),
        "worsened_n": len(worsened),
        "worsened_from_count_change": sum(1 for r in worsened if r["count_changed"]),
        "worsened_from_iou_only": sum(1 for r in worsened if r["iou_worse_count_same"]),
        "worsened_from_over_no_seg": sum(1 for r in worsened if r["over_or_no_seg_increase"]),
        "worsened_from_dup_increase": sum(1 for r in worsened if (not r["b_dup"] and r["s_dup"])),
        "worsened_from_cov_drop": sum(1 for r in worsened if r["s_cov"] < r["b_cov"] - 0.05),
    }


def build_subset_table_rows(
    baseline: LoadedDiagnostic,
    set_data: LoadedDiagnostic,
    keys: List[Tuple[str, int]],
    subsets: List[str],
) -> List[dict]:
    rows = []
    all_b = filter_records(baseline.records, keys)
    all_s = filter_records(set_data.records, keys)
    b_all = count_stats(all_b)
    s_all = count_stats(all_s)
    bi_all = iou_stats(all_b)
    si_all = iou_stats(all_s)

    def add_row(scope: str, b_recs: List[dict], s_recs: List[dict]) -> None:
        bc = count_stats(b_recs)
        sc = count_stats(s_recs)
        bi = iou_stats(b_recs)
        si = iou_stats(s_recs)
        rows.append({
            "scope": scope,
            "intersection_n": bc["n"],
            "baseline_no_seg_rate": bc["no_seg_rate"],
            "set_no_seg_rate": sc["no_seg_rate"],
            "delta_no_seg_rate": sc["no_seg_rate"] - bc["no_seg_rate"],
            "baseline_under_generate_rate": bc["under_generate_rate"],
            "set_under_generate_rate": sc["under_generate_rate"],
            "delta_under_generate_rate": sc["under_generate_rate"] - bc["under_generate_rate"],
            "baseline_equal_count_rate": bc["equal_count_rate"],
            "set_equal_count_rate": sc["equal_count_rate"],
            "delta_equal_count_rate": sc["equal_count_rate"] - bc["equal_count_rate"],
            "baseline_over_generate_rate": bc["over_generate_rate"],
            "set_over_generate_rate": sc["over_generate_rate"],
            "delta_over_generate_rate": sc["over_generate_rate"] - bc["over_generate_rate"],
            "baseline_mean_merged_iou": bi["mean_merged_iou"],
            "set_mean_merged_iou": si["mean_merged_iou"],
            "delta_mean_merged_iou": (si["mean_merged_iou"] - bi["mean_merged_iou"])
            if si["mean_merged_iou"] is not None and bi["mean_merged_iou"] is not None
            else None,
            "baseline_iou_lt_0.5_rate": bi["iou_lt_0.5_rate"],
            "set_iou_lt_0.5_rate": si["iou_lt_0.5_rate"],
            "rescued_n": sum(
                1
                for r_b, r_s in zip(b_recs, s_recs)
                if (merged_iou(r_b) or 0) < 0.5 and (merged_iou(r_s) or 0) >= 0.5
            ),
            "damaged_n": sum(
                1
                for r_b, r_s in zip(b_recs, s_recs)
                if (merged_iou(r_b) or 0) >= 0.5 and (merged_iou(r_s) or 0) < 0.5
            ),
        })

    add_row("overall_intersection", all_b, all_s)
    for subset in subsets:
        sub_keys = [k for k in keys if k[0] == subset]
        add_row(subset, filter_records(baseline.records, sub_keys), filter_records(set_data.records, sub_keys))
    return rows


def build_count_failure_table(
    baseline: LoadedDiagnostic,
    set_data: LoadedDiagnostic,
    keys: List[Tuple[str, int]],
    subsets: List[str],
) -> List[dict]:
    rows = []
    for subset in ["overall"] + list(subsets):
        if subset == "overall":
            b_recs = filter_records(baseline.records, keys)
            s_recs = filter_records(set_data.records, keys)
        else:
            sub_keys = [k for k in keys if k[0] == subset]
            b_recs = filter_records(baseline.records, sub_keys)
            s_recs = filter_records(set_data.records, sub_keys)
        bc, sc = count_stats(b_recs), count_stats(s_recs)
        for model_name, stats in [("baseline", bc), ("set", sc)]:
            for cat in COUNT_CATEGORIES:
                rows.append({
                    "subset": subset,
                    "model": model_name,
                    "category": cat,
                    "count": stats[cat],
                    "rate_pct": stats[f"{cat}_rate"],
                    "intersection_n": stats["n"],
                })
        for cat in COUNT_CATEGORIES:
            rows.append({
                "subset": subset,
                "model": "delta_set_minus_baseline",
                "category": cat,
                "count": sc[cat] - bc[cat],
                "rate_pct": sc[f"{cat}_rate"] - bc[f"{cat}_rate"],
                "intersection_n": bc["n"],
            })
    return rows


def build_multi_cate_failure_table(
    baseline: LoadedDiagnostic,
    set_data: LoadedDiagnostic,
    keys: List[Tuple[str, int]],
) -> List[dict]:
    subset = "test_multi_cate"
    sub_keys = [k for k in keys if k[0] == subset]
    b_recs = filter_records(baseline.records, sub_keys)
    s_recs = filter_records(set_data.records, sub_keys)
    rows = []
    for model_name, recs in [("baseline", b_recs), ("set", s_recs)]:
        cs = count_stats(recs)
        extra = multi_cate_extra_stats(recs)
        row = {"model": model_name, "intersection_n": cs["n"], **cs, **extra}
        rows.append(row)

    bc, sc = count_stats(b_recs), count_stats(s_recs)
    be, se = multi_cate_extra_stats(b_recs), multi_cate_extra_stats(s_recs)
    delta = {"model": "delta_set_minus_baseline", "intersection_n": bc["n"]}
    for cat in COUNT_CATEGORIES:
        delta[f"{cat}_rate"] = sc[f"{cat}_rate"] - bc[f"{cat}_rate"]
    for k in (
        "duplicate_best_match_rate",
        "all_gt_covered_rate",
        "avg_covered_gt_ratio",
        "empty_mask_rate",
        "tiny_mask_rate",
        "equal_count_mean_merged_iou",
        "under_generate_mean_merged_iou",
    ):
        if k in be and k in se and be[k] is not None and se[k] is not None:
            delta[k] = se[k] - be[k]
    rows.append(delta)
    return rows


def generate_markdown_report(
    baseline: LoadedDiagnostic,
    set_data: LoadedDiagnostic,
    keys: List[Tuple[str, int]],
    baseline_metrics: Dict[str, dict],
    set_metrics: Dict[str, dict],
    subset_table: List[dict],
    case_rows: List[dict],
    long_decomp: Dict[str, Any],
    multi_extra_b: Dict[str, Any],
    multi_extra_s: Dict[str, Any],
    multi_b_count: dict,
    multi_s_count: dict,
) -> str:
    expected_n = baseline.raw_count
    set_n = set_data.raw_count
    inter_n = len(keys)

    complete_subsets = []
    partial_subsets = []
    missing_subsets = []
    for subset, exp in sorted(baseline.subset_counts.items()):
        got = set_data.subset_counts.get(subset, 0)
        if got == 0:
            missing_subsets.append(f"{subset} (0/{exp})")
        elif got < exp:
            partial_subsets.append(f"{subset} ({got}/{exp})")
        else:
            complete_subsets.append(f"{subset} ({got}/{exp})")

    lines = [
        "# A3 Frozen SET 5w vs LaSeRS Baseline — Diagnostic Comparison",
        "",
        "> **Scope warning:** All conclusions below are based on **intersection samples only** "
        f"({inter_n} pairs). SET diagnostic is **incomplete** ({set_n}/{expected_n}). "
        "Do not treat as full-benchmark conclusions.",
        "",
        "## A. File completeness",
        "",
        f"| Item | Baseline | SET 5w |",
        f"|------|----------|--------|",
        f"| Total diagnostic records | {baseline.raw_count} | {set_data.raw_count} |",
        f"| Unique keys | {len(baseline.records)} | {len(set_data.records)} |",
        f"| Duplicate (subset, sample_id) | {baseline.duplicate_keys} | {set_data.duplicate_keys} |",
        f"| Expected total (baseline ref) | {expected_n} | {expected_n} |",
        f"| SET completion | — | **{set_n}/{expected_n} ({pct(set_n, expected_n):.1f}%)** |",
        f"| Intersection sample count | — | **{inter_n}** |",
        "",
        "**SET completed subsets:** " + (", ".join(complete_subsets) if complete_subsets else "none"),
        "",
        "**SET partial subsets:** " + (", ".join(partial_subsets) if partial_subsets else "none"),
        "",
        "**SET missing subsets:** " + (", ".join(missing_subsets) if missing_subsets else "none"),
        "",
        "**Per-subset intersection counts:**",
        "",
    ]

    for subset in sorted(set(k[0] for k in keys)):
        n = sum(1 for k in keys if k[0] == subset)
        exp = baseline.subset_counts.get(subset, 0)
        lines.append(f"- `{subset}`: {n} (baseline has {exp})")

    overall_row = next(r for r in subset_table if r["scope"] == "overall_intersection")
    multi_row = next((r for r in subset_table if r["scope"] == "test_multi_cate"), None)
    long_row = next((r for r in subset_table if r["scope"] == "test_long_query"), None)
    inst_row = next((r for r in subset_table if r["scope"] == "test_instance_level"), None)

    label_counts = Counter(r["label"] for r in case_rows)
    seg_count_changed = sum(
        1 for k in keys
        if baseline.records[k].get("generated_SEG_count") != set_data.records[k].get("generated_SEG_count")
    )
    answer_changed = sum(
        1 for k in keys
        if baseline.records[k].get("model_answer") != set_data.records[k].get("model_answer")
    )
    iou_changed = sum(
        1 for k in keys
        if merged_iou(baseline.records[k]) != merged_iou(set_data.records[k])
    )

    lines.extend([
        "",
        f"**Sample-level deltas on intersection:** generated_SEG_count changed **{seg_count_changed}/{inter_n}**; "
        f"model_answer changed {answer_changed}/{inter_n}; merged_mask_IoU changed {iou_changed}/{inter_n}.",
        "",
        "## B. Generation-side [SEG] count (intersection)",
        "",
        "| Scope | Baseline under_gen | SET under_gen | Δ | Baseline equal | SET equal | Δ | Baseline no_seg | SET no_seg | Δ |",
        "|-------|-------------------|---------------|---|----------------|-----------|---|-----------------|------------|---|",
    ])
    for row in subset_table:
        if row["scope"] not in ("overall_intersection", *FOCUS_SUBSETS):
            continue
        lines.append(
            f"| {row['scope']} | {row['baseline_under_generate_rate']:.1f}% | {row['set_under_generate_rate']:.1f}% | "
            f"{row['delta_under_generate_rate']:+.1f}pp | {row['baseline_equal_count_rate']:.1f}% | "
            f"{row['set_equal_count_rate']:.1f}% | {row['delta_equal_count_rate']:+.1f}pp | "
            f"{row['baseline_no_seg_rate']:.1f}% | {row['set_no_seg_rate']:.1f}% | {row['delta_no_seg_rate']:+.1f}pp |"
        )

    lines.extend([
        "",
        "## C. Mask-side merged IoU (intersection)",
        "",
        "| Scope | Baseline mean IoU | SET mean IoU | Δ | rescued | damaged |",
        "|-------|-------------------|--------------|---|---------|---------|",
    ])
    for row in subset_table:
        if row["scope"] not in ("overall_intersection", *FOCUS_SUBSETS):
            continue
        d = row["delta_mean_merged_iou"]
        lines.append(
            f"| {row['scope']} | {fmt_float(row['baseline_mean_merged_iou'])} | "
            f"{fmt_float(row['set_mean_merged_iou'])} | {fmt_delta(d)} | {row['rescued_n']} | {row['damaged_n']} |"
        )

    lines.extend([
        "",
        f"Case labels (intersection): rescued={label_counts.get('rescued', 0)}, "
        f"damaged={label_counts.get('damaged', 0)}, big_gain={label_counts.get('big_gain', 0)}, "
        f"big_loss={label_counts.get('big_loss', 0)}, unchanged={label_counts.get('unchanged', 0)}",
        "",
        "## D. Multiple (`test_multi_cate`) deep dive",
        "",
        f"Intersection n={multi_row['intersection_n'] if multi_row else 0} "
        f"(baseline full subset n={baseline.subset_counts.get('test_multi_cate', 0)}, "
        f"SET completed {set_data.subset_counts.get('test_multi_cate', 0)})",
        "",
        "| Metric | Baseline | SET | Δ |",
        "|--------|----------|-----|---|",
        f"| under_generate rate | {multi_b_count['under_generate_rate']:.1f}% | {multi_s_count['under_generate_rate']:.1f}% | {multi_s_count['under_generate_rate'] - multi_b_count['under_generate_rate']:+.1f}pp |",
        f"| equal_count rate | {multi_b_count['equal_count_rate']:.1f}% | {multi_s_count['equal_count_rate']:.1f}% | {multi_s_count['equal_count_rate'] - multi_b_count['equal_count_rate']:+.1f}pp |",
        f"| over_generate rate | {multi_b_count['over_generate_rate']:.1f}% | {multi_s_count['over_generate_rate']:.1f}% | {multi_s_count['over_generate_rate'] - multi_b_count['over_generate_rate']:+.1f}pp |",
        f"| equal_count mean merged IoU | {fmt_float(multi_extra_b.get('equal_count_mean_merged_iou'))} | {fmt_float(multi_extra_s.get('equal_count_mean_merged_iou'))} | {fmt_delta((multi_extra_s.get('equal_count_mean_merged_iou') or 0) - (multi_extra_b.get('equal_count_mean_merged_iou') or 0) if multi_extra_b.get('equal_count_mean_merged_iou') is not None else None)} |",
        f"| under_generate mean merged IoU | {fmt_float(multi_extra_b.get('under_generate_mean_merged_iou'))} | {fmt_float(multi_extra_s.get('under_generate_mean_merged_iou'))} | {fmt_delta((multi_extra_s.get('under_generate_mean_merged_iou') or 0) - (multi_extra_b.get('under_generate_mean_merged_iou') or 0) if multi_extra_b.get('under_generate_mean_merged_iou') is not None else None)} |",
        f"| duplicate best-match rate | {multi_extra_b.get('duplicate_best_match_rate', 0):.1f}% | {multi_extra_s.get('duplicate_best_match_rate', 0):.1f}% | {(multi_extra_s.get('duplicate_best_match_rate', 0) - multi_extra_b.get('duplicate_best_match_rate', 0)):+.1f}pp |",
        f"| all_gt_covered rate | {multi_extra_b.get('all_gt_covered_rate', 0):.1f}% | {multi_extra_s.get('all_gt_covered_rate', 0):.1f}% | {(multi_extra_s.get('all_gt_covered_rate', 0) - multi_extra_b.get('all_gt_covered_rate', 0)):+.1f}pp |",
        f"| avg covered GT ratio | {fmt_float(multi_extra_b.get('avg_covered_gt_ratio'))} | {fmt_float(multi_extra_s.get('avg_covered_gt_ratio'))} | {fmt_delta((multi_extra_s.get('avg_covered_gt_ratio') or 0) - (multi_extra_b.get('avg_covered_gt_ratio') or 0) if multi_extra_b.get('avg_covered_gt_ratio') is not None else None)} |",
        f"| tiny mask (<{TINY_MASK_THRESHOLD}px) rate | {multi_extra_b.get('tiny_mask_rate', 0):.1f}% | {multi_extra_s.get('tiny_mask_rate', 0):.1f}% | {(multi_extra_s.get('tiny_mask_rate', 0) - multi_extra_b.get('tiny_mask_rate', 0)):+.1f}pp |",
        "",
        "## E. Long Query (`test_long_query`)",
        "",
    ])

    def metric_delta(stem: str, field: str) -> str:
        b_row, s_row = baseline_metrics.get(stem, {}), set_metrics.get(stem, {})
        pct_field = f"{field}_percent"
        if pct_field in b_row and pct_field in s_row:
            b, s = b_row[pct_field], s_row[pct_field]
        else:
            b, s = b_row.get(field), s_row.get(field)
            if b is not None and s is not None and b <= 1.0 and s <= 1.0:
                b, s = 100.0 * b, 100.0 * s
        if b is None or s is None:
            return "N/A"
        return f"{b:.1f} → {s:.1f} ({s - b:+.2f} pp)"

    lines.extend([
        f"- Teacher-forced standard metrics (full eval, not diagnostic): gIoU {metric_delta('test_long_query', 'gIoU')}, cIoU {metric_delta('test_long_query', 'cIoU')}",
        f"- Diagnostic merged IoU (intersection n={long_row['intersection_n'] if long_row else 0}): "
        f"{fmt_float(long_row['baseline_mean_merged_iou'] if long_row else None)} → "
        f"{fmt_float(long_row['set_mean_merged_iou'] if long_row else None)} "
        f"({fmt_delta(long_row['delta_mean_merged_iou'] if long_row else None)})",
        "",
        "Worsened-case decomposition (samples with Δ merged IoU < 0):",
        f"- worsened n = {long_decomp.get('worsened_n', 0)} / {long_decomp.get('n', 0)}",
        f"- from generated_SEG_count change: {long_decomp.get('worsened_from_count_change', 0)}",
        f"- count same but merged IoU worse: {long_decomp.get('worsened_from_iou_only', 0)}",
        f"- new over_generate / no_seg: {long_decomp.get('worsened_from_over_no_seg', 0)}",
        f"- duplicate match appeared: {long_decomp.get('worsened_from_dup_increase', 0)}",
        f"- GT coverage ratio dropped (>0.05): {long_decomp.get('worsened_from_cov_drop', 0)}",
        "",
        "## F. Instance (`test_instance_level`) positive signal",
        "",
        f"- Teacher-forced standard metrics: gIoU {metric_delta('test_instance_level', 'gIoU')}, cIoU {metric_delta('test_instance_level', 'cIoU')}",
        f"- Diagnostic merged IoU: {fmt_float(inst_row['baseline_mean_merged_iou'] if inst_row else None)} → "
        f"{fmt_float(inst_row['set_mean_merged_iou'] if inst_row else None)} "
        f"({fmt_delta(inst_row['delta_mean_merged_iou'] if inst_row else None)})",
        f"- under_generate rate: {inst_row['baseline_under_generate_rate']:.1f}% → {inst_row['set_under_generate_rate']:.1f}% "
        f"({inst_row['delta_under_generate_rate']:+.1f}pp)" if inst_row else "",
        f"- rescued / big_gain on intersection: {inst_row['rescued_n'] if inst_row else 0} / "
        f"{sum(1 for r in case_rows if r['subset'] == 'test_instance_level' and r['label'] == 'big_gain')}",
        "",
        "## G–I. Answers to key questions",
        "",
    ])

    # Derive strict conclusions
    du_under = overall_row["delta_under_generate_rate"]
    du_equal = overall_row["delta_equal_count_rate"]
    du_iou = overall_row["delta_mean_merged_iou"] or 0
    multi_du = (multi_s_count["under_generate_rate"] - multi_b_count["under_generate_rate"]) if multi_row else 0
    multi_de = (multi_s_count["equal_count_rate"] - multi_b_count["equal_count_rate"]) if multi_row else 0
    multi_diou = (multi_row["delta_mean_merged_iou"] or 0) if multi_row else 0
    multi_dup_d = (multi_extra_s.get("duplicate_best_match_rate", 0) - multi_extra_b.get("duplicate_best_match_rate", 0))
    multi_cov_d = (multi_extra_s.get("all_gt_covered_rate", 0) - multi_extra_b.get("all_gt_covered_rate", 0))

    q1 = (
        f"SET diagnostic is **incomplete** ({set_n}/{expected_n}, {pct(set_n, expected_n):.1f}%). "
        f"Complete: {', '.join(s.split(' ')[0] for s in complete_subsets) or 'none'}. "
        f"Partial: {', '.join(partial_subsets) or 'none'}. "
        f"Missing: {', '.join(m.split(' ')[0] for m in missing_subsets) or 'none'}."
    )

    if seg_count_changed == 0:
        q2 = (
            f"On {inter_n} intersection samples, **generated_SEG_count is identical for every pair** "
            f"(0/{inter_n} changed; model_answer changed only {answer_changed}/{inter_n}). "
            "Aggregate count rates are therefore tied. SET does **not** improve target-set planning on generation."
        )
    elif abs(du_under) < 1.0 and abs(du_equal) < 1.0:
        q2 = (
            f"On {inter_n} intersection samples, SET does **not** materially change generation-side target count "
            f"(Δunder_generate={du_under:+.1f}pp, Δequal={du_equal:+.1f}pp). "
            "Cannot claim SET solved target-set planning."
        )
    elif du_under < -1.0:
        q2 = (
            f"SET **modestly reduces** under_generate ({du_under:+.1f}pp) on intersection, "
            f"but effect must be validated after full diagnostic completes."
        )
    else:
        q2 = (
            f"SET does **not** reduce under_generate overall (Δ={du_under:+.1f}pp); "
            f"equal_count Δ={du_equal:+.1f}pp."
        )

    mask_evidence = []
    if abs(du_iou) >= 0.01:
        mask_evidence.append(f"overall Δmean merged IoU={du_iou:+.3f}")
    if multi_dup_d < -1:
        mask_evidence.append(f"Multiple duplicate match ↓{abs(multi_dup_d):.1f}pp")
    elif multi_dup_d > 1:
        mask_evidence.append(f"Multiple duplicate match ↑{multi_dup_d:.1f}pp")
    if multi_cov_d > 1:
        mask_evidence.append(f"Multiple all_gt_covered ↑{multi_cov_d:.1f}pp")
    elif multi_cov_d < -1:
        mask_evidence.append(f"Multiple all_gt_covered ↓{abs(multi_cov_d):.1f}pp")

    if not mask_evidence or all(abs(float(x.split("=")[-1].replace("+", "")) if "=" in x else 0) < 0.01 for x in mask_evidence):
        q3 = "Weak or no mask-side improvement on intersection; any IoU/duplicate/coverage shift is within noise band."
    else:
        q3 = "Partial mask-side signals on intersection: " + "; ".join(mask_evidence) + "."

    std_multi_g = metric_delta("test_multi_cate", "gIoU")
    q4 = (
        f"Standard Multiple gIoU {std_multi_g} (full teacher-forced eval). "
        f"On partial diagnostic intersection (n={multi_row['intersection_n'] if multi_row else 0}), "
        f"Δunder_generate={multi_du:+.1f}pp, Δequal={multi_de:+.1f}pp, Δmean merged IoU={multi_diou:+.3f}. "
    )
    if abs(multi_du) < 1 and abs(multi_diou) < 0.01:
        q4 += "Diagnostic shows **no meaningful generation or mask gain** — small standard metric bump is likely noise / incomplete overlap."
    elif multi_de > 1 and multi_du >= 0 and abs(multi_extra_s.get("equal_count_mean_merged_iou", 0) - multi_extra_b.get("equal_count_mean_merged_iou", 0)) < 0.02:
        q4 += "SET may slightly improve equal-count cases but **not** under-generate failures."
    else:
        q4 += "Mixed/local effects only within completed multi_cate subset."

    q5 = (
        f"Long standard gIoU {metric_delta('test_long_query', 'gIoU')} drops slightly. "
        f"Diagnostic intersection mean merged IoU Δ={fmt_delta(long_row['delta_mean_merged_iou'] if long_row else None)}. "
        f"Among worsened cases, count-change drives {long_decomp.get('worsened_from_count_change', 0)}, "
        f"IoU-only {long_decomp.get('worsened_from_iou_only', 0)}, over/no_seg {long_decomp.get('worsened_from_over_no_seg', 0)}."
    )

    inst_diou = inst_row["delta_mean_merged_iou"] if inst_row else None
    inst_du = inst_row["delta_under_generate_rate"] if inst_row else 0
    q6 = (
        f"Instance standard gIoU +0.72 / cIoU +2.11 (teacher-forced). "
        f"Diagnostic: Δmean merged IoU={fmt_delta(inst_diou)}, Δunder_generate={inst_du:+.1f}pp, "
        f"rescued={inst_row['rescued_n'] if inst_row else 0}. "
    )
    if inst_diou is not None and inst_diou > 0.02:
        q6 += "Gain aligns with **merged IoU improvement** on generation diagnostic."
    elif inst_du < -1:
        q6 += "Gain partly from **fewer under_generate / no_seg**."
    else:
        q6 += "Diagnostic uplift is **modest** vs standard metrics — may reflect teacher-forced vs generation gap."

    if seg_count_changed == 0:
        q7 = (
            f"A3 SET changes **mask execution** ({iou_changed}/{inter_n} merged IoU shifts) with **zero** [SEG] count changes — "
            "weak mask-side modulation, not set planning."
        )
    elif abs(du_under) < 1 and abs(du_equal) < 1 and (inst_diou or 0) > 0.01:
        q7 = "A3 SET on current evidence looks like **weak mask execution modulation** (Instance IoU↑) rather than set planning — generation counts barely move."
    elif du_under < -1:
        q7 = "Some set-planning signal (lower under_generate), but mask-side gains are mixed."
    else:
        q7 = "Neither strong set planning nor strong mask execution — **local Instance gains only**."

    if set_n < expected_n * 0.95:
        q8 = (
            f"**Do not** continue 8w as-is without finishing diagnostic on full 1157 samples. "
            f"Current SET diagnostic stopped at {set_n}/{expected_n}; conclusions are partial."
        )
    elif abs(du_under) < 1 and (inst_diou or 0) < 0.03:
        q8 = "Full diagnostic not done; on available intersection, 8w continuation is **not supported** by generation-side evidence."
    else:
        q8 = "Need completed diagnostic before 8w decision; current partial data insufficient."

    if abs(du_under) >= 1:
        q9 = "If pursuing 8w: keep SET but add explicit target-count supervision or decoding constraint; under_generate still dominant failure."
    elif (inst_diou or 0) > 0.02 and abs(inst_du) < 1:
        q9 = "Minimal next step: **finish diagnostic eval** on remaining subsets; consider strengthening mask head / SET gate only on Instance-like queries rather than blanket 8w."
    else:
        q9 = (
            "Minimal next step: (1) resume diagnostic for missing subsets + remaining 56 multi_cate; "
            "(2) do **not** scale to 8w until under_generate drops on Multiple; "
            "(3) if continuing training, add count-aware loss or constrained [SEG] decoding."
        )

    for i, ans in enumerate([q1, q2, q3, q4, q5, q6, q7, q8, q9], start=1):
        lines.append(f"{i}. {ans}")
        lines.append("")

    lines.extend([
        "## Final strict conclusion",
        "",
    ])

    if abs(du_under) < 1 and abs(du_iou) < 0.01:
        lines.append(
            "**No robust overall improvement** on the available intersection. "
            "Instance subset shows localized merged-IoU / standard-metric uplift; "
            "Multiple and Long show no diagnostic evidence matching small standard-metric swings. "
            "SET has **not** demonstrated target-set planning fixes (under_generate unchanged)."
        )
    else:
        lines.append(
            "Mixed/localized effects only within completed subsets. "
            "Treat any positive signal as **hypothesis** until full diagnostic completes."
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze A3 SET vs baseline LaSeRS diagnostic JSONL")
    parser.add_argument("--baseline_jsonl", required=True)
    parser.add_argument("--set_jsonl", required=True)
    parser.add_argument("--baseline_metrics_dir", required=True)
    parser.add_argument("--set_metrics_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    baseline = load_jsonl(args.baseline_jsonl)
    set_data = load_jsonl(args.set_jsonl)
    intersection_keys = sorted(set(baseline.records.keys()) & set(set_data.records.keys()))
    all_subsets = sorted(set(k[0] for k in intersection_keys))

    baseline_metrics = load_standard_metrics(args.baseline_metrics_dir)
    set_metrics = load_standard_metrics(args.set_metrics_dir)

    subset_table = build_subset_table_rows(baseline, set_data, intersection_keys, all_subsets)
    case_rows = build_case_rows(baseline, set_data, intersection_keys)
    count_failure = build_count_failure_table(baseline, set_data, intersection_keys, all_subsets)
    multi_failure = build_multi_cate_failure_table(baseline, set_data, intersection_keys)

    multi_keys = [k for k in intersection_keys if k[0] == "test_multi_cate"]
    multi_b_recs = filter_records(baseline.records, multi_keys)
    multi_s_recs = filter_records(set_data.records, multi_keys)
    multi_b_count = count_stats(multi_b_recs)
    multi_s_count = count_stats(multi_s_recs)
    multi_extra_b = multi_cate_extra_stats(multi_b_recs)
    multi_extra_s = multi_cate_extra_stats(multi_s_recs)
    long_decomp = analyze_long_query_decomposition(baseline, set_data, intersection_keys)

    rescued = sorted(
        [r for r in case_rows if r["label"] == "rescued"],
        key=lambda r: r["delta_merged_IoU"] or 0,
        reverse=True,
    )[:50]
    damaged = sorted(
        [r for r in case_rows if r["label"] == "damaged"],
        key=lambda r: r["delta_merged_IoU"] or 0,
    )[:50]

    case_fields = [
        "subset", "sample_id", "query", "gt_answer",
        "baseline_model_answer", "set_model_answer",
        "gt_mask_count", "baseline_generated_SEG_count", "set_generated_SEG_count",
        "baseline_pred_mask_count", "set_pred_mask_count",
        "baseline_merged_IoU", "set_merged_IoU", "delta_merged_IoU",
        "baseline_best_matching_GT_index", "set_best_matching_GT_index",
        "baseline_per_SEG_IoU_with_each_GT", "set_per_SEG_IoU_with_each_GT",
        "baseline_per_SEG_mask_area", "set_per_SEG_mask_area",
        "baseline_per_SEG_mask_path", "set_per_SEG_mask_path",
        "label",
    ]

    subset_fields = list(subset_table[0].keys()) if subset_table else []
    count_fields = ["subset", "model", "category", "count", "rate_pct", "intersection_n"]

    write_csv(os.path.join(args.output_dir, "diagnostic_subset_table.csv"), subset_table, subset_fields)
    write_csv(os.path.join(args.output_dir, "diagnostic_case_deltas.csv"), case_rows, case_fields)
    write_csv(os.path.join(args.output_dir, "rescued_cases_top50.csv"), rescued, case_fields)
    write_csv(os.path.join(args.output_dir, "damaged_cases_top50.csv"), damaged, case_fields)
    write_csv(os.path.join(args.output_dir, "count_failure_table.csv"), count_failure, count_fields)
    if multi_failure:
        write_csv(
            os.path.join(args.output_dir, "multi_cate_failure_table.csv"),
            multi_failure,
            list(multi_failure[0].keys()),
        )

    report = generate_markdown_report(
        baseline=baseline,
        set_data=set_data,
        keys=intersection_keys,
        baseline_metrics=baseline_metrics,
        set_metrics=set_metrics,
        subset_table=subset_table,
        case_rows=case_rows,
        long_decomp=long_decomp,
        multi_extra_b=multi_extra_b,
        multi_extra_s=multi_extra_s,
        multi_b_count=multi_b_count,
        multi_s_count=multi_s_count,
    )
    report_path = os.path.join(args.output_dir, "diagnostic_comparison_summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Baseline records: {baseline.raw_count}")
    print(f"SET records: {set_data.raw_count}")
    print(f"Intersection: {len(intersection_keys)}")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
