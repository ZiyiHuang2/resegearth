#!/usr/bin/env python3
"""
只读分析：RRSIS-D explicitization 产出的 enhancement_results.jsonl（及可选分割指标 JSON）。

不调用 API；不修改任何库 schema；不运行 process_units_jsonl。

用法示例：
  python analyze_enhancement_ablation_jsonl.py \\
    --jsonl path/to/intermediate/enhancement_results.jsonl

  python analyze_enhancement_ablation_jsonl.py \\
    --baseline-jsonl outputs/ablation_baseline_20/intermediate/enhancement_results.jsonl \\
    --variant-jsonl outputs/ablation_public_v0_20/intermediate/enhancement_results.jsonl \\
    --variant-label public_v0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

# 与 pipeline 中 PROMPT_LEAKAGE_* 对齐（仅用于输出侧统计，不 import pipeline）
FORBIDDEN_FIELD_NAMES = (
    "forbidden_auto_infer_tokens",
    "visual_form_options",
    "mandatory_high_risk_terms",
    "must_put_in_forbidden",
    "do_not_put_in_forbidden",
    "generation_hint",
    "slot_guidance",
)
POLLUTION_TERMS = (
    "irregular",
    "elongated",
    "clustered",
    "patches",
    "narrow",
    "dense",
    "sparse",
    "shape",
    "texture",
)
SUBTYPE_TOKENS = (
    "river",
    "lake",
    "pond",
    "canal",
    "reservoir",
    "harbor",
    "car",
    "truck",
    "runway",
    "airplane",
    "forest",
    "grassland",
)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
    return rows


def _get_triplet(rec: Dict[str, Any]) -> Tuple[str, str, str]:
    raw = str(rec.get("raw") if rec.get("raw") is not None else rec.get("raw_expression", "") or "")
    enh = str(rec.get("enhanced") if rec.get("enhanced") is not None else rec.get("enhanced_expression", "") or "")
    cmp_ = str(rec.get("compressed") if rec.get("compressed") is not None else rec.get("compressed_expression", "") or "")
    return raw, enh, cmp_


def _word_re(token: str) -> re.Pattern:
    return re.compile(r"\b" + re.escape(token) + r"\b", re.IGNORECASE)


def _contains_word(text: str, token: str) -> bool:
    return bool(_word_re(token).search(text or ""))


def _count_word_occurrences(text: str, token: str) -> int:
    return len(_word_re(token).findall(text or ""))


def _count_substring_insensitive(haystack: str, needle: str) -> int:
    if not needle:
        return 0
    h = haystack.lower()
    n = needle.lower()
    total = 0
    start = 0
    while True:
        i = h.find(n, start)
        if i < 0:
            break
        total += 1
        start = i + len(n)
    return total


def _avg_len(xs: Iterable[str]) -> float:
    items = [x for x in xs if isinstance(x, str)]
    if not items:
        return 0.0
    return sum(len(s) for s in items) / len(items)


def analyze_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    status_breakdown: Counter = Counter()
    success_count = 0
    unchanged_count = 0
    failed_postcheck_count = 0
    failed_api_error_count = 0

    raws: List[str] = []
    enhs: List[str] = []
    cmps: List[str] = []

    subtype_new_row_hits: Counter = Counter()  # 每条记录每个 token 最多计 1
    subtype_new_occurrences: Counter = Counter()  # enh+cmp 中新增 token 出现次数之和

    forbidden_leak_hits = 0
    pollution_hits = 0

    for rec in rows:
        st = str(rec.get("status") or "missing_status")
        status_breakdown[st] += 1
        if st == "success":
            success_count += 1
        elif st == "unchanged":
            unchanged_count += 1
        elif st == "failed_postcheck":
            failed_postcheck_count += 1
        elif st == "failed_api_error":
            failed_api_error_count += 1

        raw, enh, cmp_ = _get_triplet(rec)
        raws.append(raw)
        enhs.append(enh)
        cmps.append(cmp_)

        notes = str(rec.get("notes") or "")
        # 仅统计输出侧文本，不把 semantic_context_summary 内合法 JSON 键当作泄露。
        leak_haystack = f"{enh} {cmp_} {notes}"

        for fname in FORBIDDEN_FIELD_NAMES:
            forbidden_leak_hits += _count_substring_insensitive(leak_haystack, fname)

        for term in POLLUTION_TERMS:
            pollution_hits += _count_word_occurrences(f"{enh} {cmp_}", term)
            pollution_hits += _count_word_occurrences(notes, term)

        comb = f"{enh} {cmp_}"
        for tok in SUBTYPE_TOKENS:
            in_raw = _contains_word(raw, tok)
            if in_raw:
                continue
            occ = _count_word_occurrences(comb, tok)
            if occ > 0:
                subtype_new_row_hits[tok] += 1
                subtype_new_occurrences[tok] += occ

    return {
        "num_rows": n,
        "success_count": success_count,
        "unchanged_count": unchanged_count,
        "failed_postcheck_count": failed_postcheck_count,
        "failed_api_error_count": failed_api_error_count,
        "status_breakdown": dict(status_breakdown),
        "avg_raw_len": _avg_len(raws),
        "avg_enhanced_len": _avg_len(enhs),
        "avg_compressed_len": _avg_len(cmps),
        "subtype_new_rows_by_token": dict(subtype_new_row_hits),
        "subtype_new_occurrences_by_token": dict(subtype_new_occurrences),
        "subtype_new_rows_total": int(sum(subtype_new_row_hits.values())),
        "forbidden_field_leak_substring_count": forbidden_leak_hits,
        "pollution_term_occurrence_count": pollution_hits,
    }


def _load_seg_metrics(path: Optional[str]) -> Optional[Dict[str, float]]:
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, float] = {}
    for k in ("gIoU", "cIoU", "mIoU", "oIoU"):
        if k in data and isinstance(data[k], (int, float)):
            out[k] = float(data[k])
    return out or None


def _print_block(title: str, payload: Dict[str, Any]) -> None:
    print(title)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Read-only stats for enhancement_results.jsonl (ablation A/B).")
    ap.add_argument("--jsonl", default=None, help="Single enhancement_results.jsonl path.")
    ap.add_argument("--baseline-jsonl", default=None, help="Baseline arm jsonl (e.g. no public library).")
    ap.add_argument("--variant-jsonl", default=None, help="Variant arm jsonl (e.g. public_v0).")
    ap.add_argument("--variant-label", default="variant", help="Label printed for variant arm.")
    ap.add_argument(
        "--baseline-seg-metrics-json",
        default=None,
        help="Optional RRSIS-D aggregate metrics JSON (top-level gIoU/cIoU).",
    )
    ap.add_argument("--variant-seg-metrics-json", default=None, help="Optional metrics JSON for variant arm.")
    args = ap.parse_args()

    if args.jsonl:
        rows = _read_jsonl(args.jsonl)
        report = analyze_rows(rows)
        report["source_jsonl"] = args.jsonl
        m = _load_seg_metrics(args.baseline_seg_metrics_json or args.variant_seg_metrics_json)
        if m:
            report["segmentation_metrics_top_level"] = m
        _print_block("single_arm", report)
        return

    if args.baseline_jsonl and args.variant_jsonl:
        b_rows = _read_jsonl(args.baseline_jsonl)
        v_rows = _read_jsonl(args.variant_jsonl)
        b_rep = analyze_rows(b_rows)
        v_rep = analyze_rows(v_rows)
        b_rep["source_jsonl"] = args.baseline_jsonl
        v_rep["source_jsonl"] = args.variant_jsonl
        v_rep["arm_label"] = args.variant_label

        b_m = _load_seg_metrics(args.baseline_seg_metrics_json)
        v_m = _load_seg_metrics(args.variant_seg_metrics_json)
        if b_m:
            b_rep["segmentation_metrics_top_level"] = b_m
        if v_m:
            v_rep["segmentation_metrics_top_level"] = v_m

        delta: Dict[str, Any] = {
            "variant_label": args.variant_label,
            "delta_failed_postcheck": v_rep["failed_postcheck_count"] - b_rep["failed_postcheck_count"],
            "delta_failed_api_error": v_rep["failed_api_error_count"] - b_rep["failed_api_error_count"],
            "delta_unchanged": v_rep["unchanged_count"] - b_rep["unchanged_count"],
            "delta_success": v_rep["success_count"] - b_rep["success_count"],
            "delta_subtype_new_rows_total": v_rep["subtype_new_rows_total"] - b_rep["subtype_new_rows_total"],
            "delta_avg_enhanced_len": v_rep["avg_enhanced_len"] - b_rep["avg_enhanced_len"],
            "delta_pollution_occurrences": v_rep["pollution_term_occurrence_count"] - b_rep["pollution_term_occurrence_count"],
        }
        if b_m and v_m:
            for k in ("gIoU", "cIoU"):
                if k in b_m and k in v_m:
                    delta[f"delta_{k}"] = v_m[k] - b_m[k]

        _print_block("baseline", b_rep)
        _print_block(args.variant_label, v_rep)
        _print_block("delta_variant_minus_baseline", delta)

        print(
            "\n# 判读提示（仅启发式）\n"
            "- delta_failed_postcheck > 0：优先怀疑 boundary 负向列举等导致后验失败增多。\n"
            "- delta_unchanged > 0：变体更保守、改写减少。\n"
            "- delta_subtype_new_rows_total < 0 且 delta_gIoU/delta_cIoU 接近 0：语言层subtype注入减少但分割未变。\n"
            "- hard-negative 上 delta_gIoU / delta_cIoU 稳定为正：v0 公共边界表值得保留为论文 ablation。\n"
        )
        return

    print(
        "Specify either --jsonl PATH or both --baseline-jsonl and --variant-jsonl.",
        file=sys.stderr,
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
