#!/usr/bin/env python3
"""
Read-only analysis: compare RRSISD test metrics (baseline vs public_semantic_v2).

Does not train, call APIs, or modify models/datasets.

Mapping:
  rrsisd_test_metrics.json "details" rows align with RRSISDDataset order for split=test.
  Field "idx" is the dataloader index into refs(unc).p filtered by split==test (length 3481);
  one index (empty GT) is skipped in metrics, so "idx" has a single gap (83 in current runs).
  Field "id" equals ref["ref_id"] and seg_info["data_id"] (see dataset.py + llava_phi.eval_seg).

Outputs under --out-dir (default: outputs/source/public_semantic_v2_analysis).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


# User “focus” names — many do not exist in RRSISD (20-class RS subset). We still tag overlaps.
FOCUS_NAMES = {
    "water",
    "river",
    "lake",
    "bridge",
    "road",
    "vehicle",
    "airport",
    "parking lot",
    "building",
    "vegetation",
    "farmland",
    "grassland",
    "forest",
    "bare land",
    "ship",
    "harbor",
    "playground",
}

# Verdict heuristics: map RRSISD category names (instances.json) to hypotheses.
GROUP_A_SPATIAL = {
    "bridge",
    "airport",
    "vehicle",
    "harbor",
    "ship",
    "overpass",
    "Expressway-Service-area",
    "Expressway-toll-station",
}
GROUP_B_STUFFLIKE = {
    "golffield",
    "groundtrackfield",
    "stadium",
    "baseballfield",
    "basketballcourt",
    "tenniscourt",
    "dam",
}


def _safe_div(num: float, den: float) -> float:
    if den == 0 or math.isnan(den):
        return float("nan")
    return num / den


def _areas_from_metrics(row: Dict[str, Any]) -> Tuple[float, float]:
    """pred pixel count, gt pixel count from inter / precision / recall."""
    inter = float(row.get("inter") or 0)
    prec = float(row.get("precision") or 0)
    rec = float(row.get("recall") or 0)
    pred_area = _safe_div(inter, prec) if prec > 0 else 0.0
    gt_area = _safe_div(inter, rec) if rec > 0 else 0.0
    return pred_area, gt_area


def load_rrsisd_aux(rrsisd_data_root: Path) -> Tuple[List[dict], Dict[int, str], Dict[int, dict]]:
    refs_path = rrsisd_data_root / "rrsisd" / "refs(unc).p"
    inst_path = rrsisd_data_root / "rrsisd" / "instances.json"
    with open(refs_path, "rb") as f:
        refs = pickle.load(f)
    test_refs = [r for r in refs if r.get("split") == "test"]
    with open(inst_path, "r", encoding="utf-8") as f:
        inst = json.load(f)
    cat_id_to_name = {c["id"]: c["name"] for c in inst["categories"]}
    ann_by_id = {a["id"]: a for a in inst["annotations"]}
    return test_refs, cat_id_to_name, ann_by_id


def load_details(metrics_path: Path) -> List[dict]:
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data["details"])


def merge_runs(
    base_details: List[dict], v2_details: List[dict]
) -> List[Tuple[dict, dict]]:
    by_idx_b = {d["idx"]: d for d in base_details}
    by_idx_v = {d["idx"]: d for d in v2_details}
    if set(by_idx_b) != set(by_idx_v):
        raise ValueError(
            f"idx set mismatch: baseline {len(by_idx_b)} vs v2 {len(by_idx_v)}, "
            f"symmetric_diff={set(by_idx_b)^set(by_idx_v)}"
        )
    idxs = sorted(by_idx_b)
    return [(by_idx_b[i], by_idx_v[i]) for i in idxs]


def build_per_sample_rows(
    pairs: List[Tuple[dict, dict]],
    test_refs: List[dict],
    cat_id_to_name: Dict[int, str],
) -> List[dict]:
    rows: List[dict] = []
    for db, dv in pairs:
        idx = int(db["idx"])
        ref = test_refs[idx]
        ref_id = int(ref["ref_id"])
        if ref_id != int(db["id"]):
            raise ValueError(f"ref_id mismatch at idx={idx}: ref {ref_id} vs metrics {db['id']}")
        ann_id = int(ref["ann_id"])
        image_id = int(ref["image_id"])
        cat_id = int(ref.get("category_id", -1))
        category_name = cat_id_to_name.get(cat_id, "unknown")
        expression = str(db.get("description") or "")

        bi, bd = float(db["iou"]), float(db["dice"])
        vi, vd_ = float(dv["iou"]), float(dv["dice"])
        bp_area, bgt = _areas_from_metrics(db)
        vp_area, vgt = _areas_from_metrics(dv)

        rows.append(
            {
                "sample_index": idx,
                "ref_id": ref_id,
                "ann_id": ann_id,
                "image_id": image_id,
                "category_id": cat_id,
                "category_name": category_name,
                "expression": expression,
                "baseline_iou": bi,
                "public_v2_iou": vi,
                "delta_iou": vi - bi,
                "baseline_dice": bd,
                "public_v2_dice": vd_,
                "delta_dice": vd_ - bd,
                "baseline_pred_area": bp_area,
                "public_v2_pred_area": vp_area,
                "gt_area": bgt,
                "baseline_precision": float(db.get("precision") or 0),
                "public_v2_precision": float(dv.get("precision") or 0),
                "delta_precision": float(dv.get("precision") or 0) - float(db.get("precision") or 0),
                "baseline_recall": float(db.get("recall") or 0),
                "public_v2_recall": float(dv.get("recall") or 0),
                "delta_recall": float(dv.get("recall") or 0) - float(db.get("recall") or 0),
                "baseline_inter": int(db.get("inter") or 0),
                "baseline_union": int(db.get("union") or 0),
                "public_v2_inter": int(dv.get("inter") or 0),
                "public_v2_union": int(dv.get("union") or 0),
            }
        )
    return rows


def aggregate_category(rows: List[dict]) -> List[dict]:
    by_cat: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category_name"]].append(r)

    out: List[dict] = []
    for cat in sorted(by_cat.keys()):
        xs = by_cat[cat]
        n = len(xs)

        def mean(key: str) -> float:
            return sum(float(x[key]) for x in xs) / n if n else 0.0

        def pooled_iou(key_inter: str, key_union: str) -> float:
            si = sum(int(x[key_inter]) for x in xs)
            su = sum(int(x[key_union]) for x in xs)
            return si / su if su > 0 else float("nan")

        b_miou = mean("baseline_iou")
        v_miou = mean("public_v2_iou")
        b_mdice = mean("baseline_dice")
        v_mdice = mean("public_v2_dice")
        b_ciou = pooled_iou("baseline_inter", "baseline_union")
        v_ciou = pooled_iou("public_v2_inter", "public_v2_union")
        b_mpa = mean("baseline_pred_area")
        v_mpa = mean("public_v2_pred_area")
        mgta = mean("gt_area")

        out.append(
            {
                "category_name": cat,
                "sample_count": n,
                "baseline_gIoU_macro_mean_iou": b_miou,
                "public_v2_gIoU_macro_mean_iou": v_miou,
                "delta_gIoU_macro": v_miou - b_miou,
                "baseline_mDice": b_mdice,
                "public_v2_mDice": v_mdice,
                "delta_mDice": v_mdice - b_mdice,
                "baseline_cIoU_pooled": b_ciou,
                "public_v2_cIoU_pooled": v_ciou,
                "delta_cIoU_pooled": v_ciou - b_ciou,
                "baseline_mean_pred_area": b_mpa,
                "public_v2_mean_pred_area": v_mpa,
                "mean_gt_area": mgta,
                "delta_mean_pred_area": v_mpa - b_mpa,
                "user_focus_overlap": cat.lower() in {x.lower() for x in FOCUS_NAMES},
            }
        )

    out.sort(key=lambda z: z["delta_gIoU_macro"], reverse=True)
    return out


def verdict_block(rows: List[dict], cat_rows: List[dict]) -> Dict[str, Any]:
    """Heuristic A/B/C/D from per-sample and per-category stats."""
    n = len(rows)
    sum_delta = sum(r["delta_iou"] for r in rows)
    mean_delta_iou_global = sum_delta / n if n else 0.0

    def samples_in_group(names: set) -> List[dict]:
        return [r for r in rows if r["category_name"] in names]

    a_rows = samples_in_group(GROUP_A_SPATIAL)
    b_rows = samples_in_group(GROUP_B_STUFFLIKE)
    rest = [r for r in rows if r not in a_rows and r not in b_rows]

    def mean_group_delta(xs: List[dict]) -> float:
        return sum(x["delta_iou"] for x in xs) / len(xs) if xs else 0.0

    mean_a = mean_group_delta(a_rows)
    mean_b = mean_group_delta(b_rows)
    mean_r = mean_group_delta(rest)

    # concentration: top-50 samples share of total positive gain
    pos = sorted((r["delta_iou"] for r in rows if r["delta_iou"] > 0), reverse=True)
    pos_sum = sum(pos)
    top50_pos = sum(pos[:50])
    top50_share = top50_pos / pos_sum if pos_sum > 1e-9 else 0.0

    # category concentration: contribution = sum(delta_iou)
    cat_contrib = []
    for c in cat_rows:
        name = c["category_name"]
        sm = sum(r["delta_iou"] for r in rows if r["category_name"] == name)
        cat_contrib.append((name, sm, c["sample_count"]))
    cat_contrib.sort(key=lambda t: abs(t[1]), reverse=True)
    top3_abs = sum(abs(t[1]) for t in cat_contrib[:3])
    tot_abs = sum(abs(t[1]) for t in cat_contrib) or 1.0
    top3_share = top3_abs / tot_abs

    # precision / recall story (global means)
    mean_dp = sum(r["delta_precision"] for r in rows) / n
    mean_dr = sum(r["delta_recall"] for r in rows) / n
    mean_dpa = sum(r["public_v2_pred_area"] - r["baseline_pred_area"] for r in rows) / n

    # std of per-category macro delta
    dmacros = [c["delta_gIoU_macro"] for c in cat_rows]
    mean_cm = sum(dmacros) / len(dmacros) if dmacros else 0.0
    var = sum((x - mean_cm) ** 2 for x in dmacros) / len(dmacros) if dmacros else 0.0
    std_cm = math.sqrt(var)

    code = "C"
    rationale: List[str] = []

    if mean_a >= 0.002 and mean_a >= mean_b + 0.001 and mean_a > mean_r * 1.05:
        code = "A"
        rationale.append(
            "空间/交通结构类（bridge/airport/vehicle/harbor/ship/overpass/高速相关）"
            f" 平均 ΔIoU={mean_a:.5f}，高于 stuff 近似组({mean_b:.5f}) 与其余({mean_r:.5f})。"
        )
    elif mean_b >= 0.002 and mean_b >= mean_a + 0.001:
        code = "B"
        rationale.append(
            "运动场/场地近似组（golffield/groundtrackfield/stadium/球场类/dam 等）"
            f" 平均 ΔIoU={mean_b:.5f}，高于空间组({mean_a:.5f})。RRSISD 无植被/农田等细类，此为粗代理。"
        )
    elif std_cm < 0.004 and top3_share < 0.45:
        code = "C"
        rationale.append(
            f"各类别 macro-ΔIoU 标准差较小({std_cm:.5f})，且前三类别对 |ΣΔIoU| 占比 {top3_share:.2%}，"
            "提升较分散。"
        )
    elif top50_share > 0.35 or top3_share > 0.55:
        code = "D"
        rationale.append(
            f"样本或类别集中：top50 正增益样本占正增益总和 {top50_share:.2%}；"
            f"前三类别 |Δ| 占比 {top3_share:.2%}。"
        )
    else:
        code = "C"
        rationale.append("未强烈满足 A/B/D 触发条件，判为均匀小幅或混合。")

    return {
        "verdict_code": code,
        "rationale_lines": rationale,
        "stats": {
            "n_samples": n,
            "sum_delta_iou": sum_delta,
            "mean_delta_iou": mean_delta_iou_global,
            "group_A_mean_delta_iou": mean_a,
            "group_B_mean_delta_iou": mean_b,
            "other_mean_delta_iou": mean_r,
            "group_A_n": len(a_rows),
            "group_B_n": len(b_rows),
            "top50_positive_gain_share": top50_share,
            "top3_category_abs_delta_share": top3_share,
            "per_category_delta_iou_std": std_cm,
            "mean_delta_precision": mean_dp,
            "mean_delta_recall": mean_dr,
            "mean_delta_pred_area": mean_dpa,
        },
        "top_categories_by_sum_delta_iou": [
            {"category_name": a, "sum_delta_iou": float(b), "sample_count": int(c)}
            for a, b, c in sorted(cat_contrib, key=lambda t: t[1], reverse=True)[:8]
        ],
    }


def write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline-metrics",
        type=Path,
        default=Path(
            "/home/wangchengjun/huangziyi/reseg/output/source/rrsisd_baseline_raw_7w/rrsisd_test_metrics.json"
        ),
    )
    ap.add_argument(
        "--public-v2-metrics",
        type=Path,
        default=Path(
            "/home/wangchengjun/huangziyi/reseg/output/source/rrsisd_public_semantic_v2_7w/rrsisd_test_metrics.json"
        ),
    )
    ap.add_argument(
        "--rrsisd-data-root",
        type=Path,
        default=Path("/home/wangchengjun/huangziyi/data/RRSISD"),
        help="Parent of rrsisd/refs(unc).p and rrsisd/instances.json",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/source/public_semantic_v2_analysis"),
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = (repo_root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    test_refs, cat_id_to_name, _ann_by_id = load_rrsisd_aux(args.rrsisd_data_root)
    bd = load_details(args.baseline_metrics)
    vd = load_details(args.public_v2_metrics)
    pairs = merge_runs(bd, vd)
    rows = build_per_sample_rows(pairs, test_refs, cat_id_to_name)

    per_sample_path = out_dir / "per_sample_iou_diff.jsonl"
    write_jsonl(per_sample_path, rows)

    improved = sorted(rows, key=lambda r: r["delta_iou"], reverse=True)[:50]
    degraded = sorted(rows, key=lambda r: r["delta_iou"])[:50]
    write_jsonl(out_dir / "top50_improved.jsonl", improved)
    write_jsonl(out_dir / "top50_degraded.jsonl", degraded)

    cat_rows = aggregate_category(rows)
    verdict = verdict_block(rows, cat_rows)

    with open(out_dir / "per_category_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"categories": cat_rows, "verdict": verdict}, f, indent=2, ensure_ascii=False)

    csv_path = out_dir / "per_category_metrics.csv"
    if cat_rows:
        keys = list(cat_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for c in cat_rows:
                w.writerow(c)

    def cat_hist(rs: List[dict]) -> Dict[str, int]:
        h: Dict[str, int] = defaultdict(int)
        for r in rs:
            h[r["category_name"]] += 1
        return dict(sorted(h.items(), key=lambda kv: (-kv[1], kv[0])))

    summary = {
        "top50_improved_category_counts": cat_hist(improved),
        "top50_degraded_category_counts": cat_hist(degraded),
        "mapping_notes": {
            "refs_order": "refs(unc).p entries with split==test, same order as RRSISDDataset.__getitem__ index.",
            "metrics_idx": "rrsisd_test_metrics.json details[].idx equals that dataset index; one index skipped for empty GT.",
            "ref_id": "details[].id == ref_id == eval_seg output id == data_id in dataset annotations.",
            "pred_masks": "{image_stem}_{ref_id}_test_0.tif uint8 {0,255} per segearth_r2/eval/eval.py",
        },
        "counts": {"per_sample_rows": len(rows), "test_refs_total": len(test_refs)},
        "verdict": verdict,
    }
    with open(out_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
