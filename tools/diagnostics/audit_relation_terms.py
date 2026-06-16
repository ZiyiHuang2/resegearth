#!/usr/bin/env python3
"""CPU-only LaSeRS relation-term statistics for counterfactual eval design."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set


RELATION_PATTERNS: Dict[str, re.Pattern] = {
    "left": re.compile(r"\bleft\b", re.I),
    "right": re.compile(r"\bright\b", re.I),
    "top": re.compile(r"\btop\b", re.I),
    "bottom": re.compile(r"\bbottom\b", re.I),
    "upper": re.compile(r"\bupper\b", re.I),
    "lower": re.compile(r"\blower\b", re.I),
    "nearest": re.compile(r"\bnearest\b", re.I),
    "farthest": re.compile(r"\bfarthest\b|\bfurthest\b", re.I),
    "largest": re.compile(r"\blargest\b|\bbiggest\b", re.I),
    "smallest": re.compile(r"\bsmallest\b", re.I),
    "between": re.compile(r"\bbetween\b", re.I),
    "next_to": re.compile(r"\bnext to\b|\badjacent to\b", re.I),
    "adjacent": re.compile(r"\badjacent\b", re.I),
}


def load_annotations(ann_dir: Path) -> List[dict]:
    entries: List[dict] = []
    for path in sorted(ann_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            item["_source_file"] = path.name
        entries.extend(data)
    return entries


def seg_count(entry: dict) -> int:
    return entry.get("answer", "").count("[SEG]")


def matched_terms(text: str) -> Set[str]:
    hits = set()
    for name, pat in RELATION_PATTERNS.items():
        if pat.search(text):
            hits.add(name)
    return hits


def audit(ann_dir: Path) -> dict:
    entries = load_annotations(ann_dir)
    total = len(entries)

    term_stats = {}
    for term in RELATION_PATTERNS:
        matched = [e for e in entries if term in matched_terms(e.get("description", "") + " " + e.get("reasoning", ""))]
        n = len(matched)
        k_gt1 = sum(1 for e in matched if seg_count(e) > 1)
        multi_cate = sum(1 for e in matched if "multi_cate" in e.get("_source_file", ""))
        same_class_proxy = sum(
            1 for e in matched if "|" not in e.get("category", "") and seg_count(e) > 1
        )
        term_stats[term] = {
            "count": n,
            "fraction_of_all": n / total if total else 0,
            "k_gt_1_count": k_gt1,
            "k_gt_1_rate": k_gt1 / n if n else 0,
            "from_multi_cate_file": multi_cate,
            "multi_cate_rate": multi_cate / n if n else 0,
            "same_class_single_label_k_gt_1": same_class_proxy,
        }

    ranked = sorted(
        term_stats.items(),
        key=lambda kv: (kv[1]["k_gt_1_rate"] * kv[1]["count"], kv[1]["count"]),
        reverse=True,
    )
    recommendations = [
        {
            "term": t,
            "score": round(s["k_gt_1_rate"] * s["count"], 2),
            "count": s["count"],
            "k_gt_1_rate": round(s["k_gt_1_rate"], 3),
            "multi_cate_rate": round(s["multi_cate_rate"], 3),
        }
        for t, s in ranked[:8]
        if s["count"] >= 5
    ]

    by_file = Counter(e["_source_file"] for e in entries)
    k_dist = Counter(seg_count(e) for e in entries)

    return {
        "ann_dir": str(ann_dir),
        "total_samples": total,
        "files": dict(by_file),
        "seg_count_distribution": {str(k): v for k, v in sorted(k_dist.items())},
        "relation_term_stats": term_stats,
        "counterfactual_eval_recommendations": recommendations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ann-dir",
        default="/root/rivermind-data/huangziyi/data/LaSeRS/test/annotations",
        help="LaSeRS test annotations directory",
    )
    parser.add_argument("--out-json", required=True, help="Output JSON path")
    args = parser.parse_args()

    ann_dir = Path(args.ann_dir)
    if not ann_dir.is_dir():
        raise SystemExit(f"Annotation dir not found: {ann_dir}")

    report = audit(ann_dir)
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report["counterfactual_eval_recommendations"], indent=2))
    print(f"Wrote full report to {out}")


if __name__ == "__main__":
    main()
