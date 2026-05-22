#!/usr/bin/env python3
"""
Analysis-only: relate public_semantic_v2 per-sample metric deltas to semantic-library prior hits.

Matching is delegated to tools/rrsisd_explicitization_pipeline.retrieve_concept_semantics
(same path as training via segearth_r2.utils.concept_public_grounding_train), so policies
(word_boundary, continuous_phrase, longest_first_non_overlapping, child_suppresses_parent, …)
stay identical to public_semantic_v2 without reimplementing them here.

Read-only: no training, no API, no writes outside --output-dir.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import pickle
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_explicitization_pipeline():
    """Same bridge pattern as segearth_r2/utils/concept_public_grounding_train.py."""
    repo = _repo_root()
    pipe_path = repo / "tools" / "rrsisd_explicitization_pipeline.py"
    if not pipe_path.is_file():
        raise FileNotFoundError(f"Missing pipeline module: {pipe_path}")
    spec = importlib.util.spec_from_file_location(
        "rrsisd_explicitization_pipeline_prior_hit_analysis", pipe_path
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _norm_label(s: Any) -> str:
    return (str(s) if s is not None else "").strip().lower()


def _get_first(row: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def _float_or(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _int_or(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except (TypeError, ValueError):
        return default


def _canonicalize_per_sample_row(
    row: Dict[str, Any], field_log: Dict[str, str]
) -> Dict[str, Any]:
    """
    Map heterogeneous jsonl field names to canonical keys used downstream.
    Records chosen source keys into field_log (updated in-place, last write wins).
    """
    out: Dict[str, Any] = {}

    idx = _get_first(row, ("idx", "sample_index", "sample_idx"))
    if idx is not None:
        out["idx"] = _int_or(idx, 0)
        field_log["idx"] = next(k for k in ("idx", "sample_index", "sample_idx") if k in row)

    rid = _get_first(row, ("ref_id", "id"))
    if rid is not None:
        out["ref_id"] = _int_or(rid, 0)
        field_log["ref_id"] = next(k for k in ("ref_id", "id") if k in row)

    cat = _get_first(row, ("category_name", "category"))
    if cat is not None:
        out["category_name"] = str(cat)
        field_log["category_name"] = next(k for k in ("category_name", "category") if k in row)

    expr = _get_first(row, ("expression", "description", "raw_expression"))
    if expr is not None:
        out["expression"] = str(expr)
        field_log["expression"] = next(k for k in ("expression", "description", "raw_expression") if k in row)

    out["baseline_iou"] = _float_or(_get_first(row, ("baseline_iou", "base_iou")), float("nan"))
    field_log["baseline_iou"] = "baseline_iou" if "baseline_iou" in row else (
        "base_iou" if "base_iou" in row else field_log.get("baseline_iou", "missing")
    )

    out["v2_iou"] = _float_or(
        _get_first(row, ("public_v2_iou", "v2_iou", "public_semantic_v2_iou")), float("nan")
    )
    for k in ("public_v2_iou", "v2_iou", "public_semantic_v2_iou"):
        if k in row:
            field_log["v2_iou"] = k
            break

    if not math.isnan(out["baseline_iou"]) and not math.isnan(out["v2_iou"]):
        out["delta_iou"] = out["v2_iou"] - out["baseline_iou"]
    else:
        out["delta_iou"] = _float_or(_get_first(row, ("delta_iou",)), float("nan"))

    out["baseline_precision"] = _float_or(_get_first(row, ("baseline_precision",)), float("nan"))
    out["v2_precision"] = _float_or(
        _get_first(row, ("public_v2_precision", "v2_precision")), float("nan")
    )
    for k in ("public_v2_precision", "v2_precision"):
        if k in row:
            field_log["v2_precision"] = k
            break

    out["baseline_recall"] = _float_or(_get_first(row, ("baseline_recall",)), float("nan"))
    out["v2_recall"] = _float_or(_get_first(row, ("public_v2_recall", "v2_recall")), float("nan"))
    for k in ("public_v2_recall", "v2_recall"):
        if k in row:
            field_log["v2_recall"] = k
            break

    out["baseline_pred_area"] = _float_or(_get_first(row, ("baseline_pred_area",)), float("nan"))
    out["v2_pred_area"] = _float_or(
        _get_first(row, ("public_v2_pred_area", "v2_pred_area")), float("nan"
    ))
    for k in ("public_v2_pred_area", "v2_pred_area"):
        if k in row:
            field_log["v2_pred_area"] = k
            break

    out["gt_area"] = _float_or(_get_first(row, ("gt_area",)), float("nan"))

    if not math.isnan(out["baseline_precision"]) and not math.isnan(out["v2_precision"]):
        out["delta_precision"] = out["v2_precision"] - out["baseline_precision"]
    else:
        out["delta_precision"] = _float_or(_get_first(row, ("delta_precision",)), float("nan"))

    if not math.isnan(out["baseline_recall"]) and not math.isnan(out["v2_recall"]):
        out["delta_recall"] = out["v2_recall"] - out["baseline_recall"]
    else:
        out["delta_recall"] = _float_or(_get_first(row, ("delta_recall",)), float("nan"))

    if not math.isnan(out["baseline_pred_area"]) and not math.isnan(out["v2_pred_area"]):
        out["delta_pred_area"] = out["v2_pred_area"] - out["baseline_pred_area"]
    else:
        out["delta_pred_area"] = _float_or(_get_first(row, ("delta_pred_area",)), float("nan"))

    if "foreground_ratio" in row and row["foreground_ratio"] is not None:
        out["foreground_ratio"] = _float_or(row["foreground_ratio"], float("nan"))
        field_log["foreground_ratio"] = "foreground_ratio"
    else:
        out["foreground_ratio"] = float("nan")

    return out


def _build_library_category_labels(library: Dict[str, Any]) -> Set[str]:
    """Normalized labels for GT category ↔ library membership (keys + cfg['concept'])."""
    concepts = library.get("concepts", {})
    if not isinstance(concepts, dict):
        return set()
    labels: Set[str] = set()
    for key, cfg in concepts.items():
        labels.add(_norm_label(key))
        if isinstance(cfg, dict):
            labels.add(_norm_label(cfg.get("concept", key)))
    labels.discard("")
    return labels


def _category_in_library(category_name: str, labels: Set[str]) -> bool:
    return _norm_label(category_name) in labels


def _image_area_map(instances_json: Path) -> Dict[int, int]:
    with open(instances_json, "r", encoding="utf-8") as f:
        inst = json.load(f)
    out: Dict[int, int] = {}
    for im in inst.get("images", []) or []:
        if not isinstance(im, dict):
            continue
        iid = im.get("id")
        h = im.get("height")
        w = im.get("width")
        if iid is None or h is None or w is None:
            continue
        try:
            out[int(iid)] = int(h) * int(w)
        except (TypeError, ValueError):
            continue
    return out


def _enrich_from_refs(
    idx: int,
    test_refs: List[dict],
    cat_id_to_name: Dict[int, str],
) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    if idx < 0 or idx >= len(test_refs):
        return None, None, None
    ref = test_refs[idx]
    expr = None
    sents = ref.get("sentences") or []
    if sents and isinstance(sents[0], dict):
        for key in ("sent", "raw", "sentence", "text"):
            v = sents[0].get(key)
            if isinstance(v, str) and v.strip():
                expr = v.strip()
                break
    cid = ref.get("category_id")
    try:
        cid_i = int(cid) if cid is not None else None
    except (TypeError, ValueError):
        cid_i = None
    cname = cat_id_to_name.get(cid_i, None) if cid_i is not None else None
    return expr, cname, cid_i


def _enrich_from_details(
    idx: int, details_by_idx: Dict[int, dict]
) -> Optional[str]:
    d = details_by_idx.get(idx)
    if not d:
        return None
    return str(d.get("description") or "").strip() or None


def _foreground_ratio(gt_area: float, image_id: Optional[int], imap: Dict[int, int]) -> float:
    if math.isnan(gt_area) or gt_area <= 0 or image_id is None:
        return float("nan")
    area = imap.get(int(image_id))
    if not area or area <= 0:
        return float("nan")
    return float(gt_area) / float(area)


def _area_bucket(fr: float) -> str:
    if math.isnan(fr):
        return "unknown"
    if fr < 0.005:
        return "tiny"
    if fr < 0.02:
        return "small"
    if fr < 0.08:
        return "medium"
    return "large"


def _mean(xs: List[float]) -> float:
    ys = [x for x in xs if not math.isnan(x)]
    return sum(ys) / len(ys) if ys else float("nan")


def _aggregate_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "sample_count": 0,
            "baseline_mean_iou": None,
            "v2_mean_iou": None,
            "delta_mean_iou": None,
            "baseline_mean_precision": None,
            "v2_mean_precision": None,
            "delta_precision": None,
            "baseline_mean_recall": None,
            "v2_mean_recall": None,
            "delta_recall": None,
            "baseline_mean_pred_area": None,
            "v2_mean_pred_area": None,
            "delta_pred_area": None,
            "positive_sample_count": 0,
            "negative_sample_count": 0,
            "positive_ratio": None,
            "negative_ratio": None,
            "sum_positive_delta": None,
            "sum_negative_delta": None,
        }

    def col(key: str) -> List[float]:
        return [_float_or(r.get(key), float("nan")) for r in rows]

    di = col("delta_iou")
    pos_c = sum(1 for x in di if not math.isnan(x) and x > 0)
    neg_c = sum(1 for x in di if not math.isnan(x) and x < 0)
    pos_sum = sum(x for x in di if not math.isnan(x) and x > 0)
    neg_sum = sum(x for x in di if not math.isnan(x) and x < 0)

    bm_iou = _mean(col("baseline_iou"))
    vm_iou = _mean(col("v2_iou"))
    dm_iou = _mean(di)

    return {
        "sample_count": n,
        "baseline_mean_iou": bm_iou,
        "v2_mean_iou": vm_iou,
        "delta_mean_iou": dm_iou,
        "baseline_mean_precision": _mean(col("baseline_precision")),
        "v2_mean_precision": _mean(col("v2_precision")),
        "delta_precision": _mean(col("delta_precision")),
        "baseline_mean_recall": _mean(col("baseline_recall")),
        "v2_mean_recall": _mean(col("v2_recall")),
        "delta_recall": _mean(col("delta_recall")),
        "baseline_mean_pred_area": _mean(col("baseline_pred_area")),
        "v2_mean_pred_area": _mean(col("v2_pred_area")),
        "delta_pred_area": _mean(col("delta_pred_area")),
        "positive_sample_count": pos_c,
        "negative_sample_count": neg_c,
        "positive_ratio": pos_c / n if n else None,
        "negative_ratio": neg_c / n if n else None,
        "sum_positive_delta": pos_sum,
        "sum_negative_delta": neg_sum,
    }


def _fmt_float(x: Any, nd: int = 6) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, float) and math.isnan(x):
            return ""
    except TypeError:
        pass
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return ""


def _write_group_csv(path: Path, group_rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "group_name",
        "sample_count",
        "baseline_mean_iou",
        "v2_mean_iou",
        "delta_mean_iou",
        "baseline_mean_precision",
        "v2_mean_precision",
        "delta_precision",
        "baseline_mean_recall",
        "v2_mean_recall",
        "delta_recall",
        "baseline_mean_pred_area",
        "v2_mean_pred_area",
        "delta_pred_area",
        "positive_sample_count",
        "negative_sample_count",
        "positive_ratio",
        "negative_ratio",
        "sum_positive_delta",
        "sum_negative_delta",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for gr in group_rows:
            row = {k: gr.get(k) for k in cols}
            for k in (
                "baseline_mean_iou",
                "v2_mean_iou",
                "delta_mean_iou",
                "baseline_mean_precision",
                "v2_mean_precision",
                "delta_precision",
                "baseline_mean_recall",
                "v2_mean_recall",
                "delta_recall",
                "baseline_mean_pred_area",
                "v2_mean_pred_area",
                "delta_pred_area",
                "positive_ratio",
                "negative_ratio",
                "sum_positive_delta",
                "sum_negative_delta",
            ):
                row[k] = _fmt_float(row[k]) if row.get("sample_count", 0) else ""
            w.writerow(row)


def _top_cat_str(counter: Counter, topn: int = 5) -> str:
    parts = [f"{k}:{v}" for k, v in counter.most_common(topn)]
    return "|".join(parts)


def _choose_recommended_step(s: Dict[str, Any]) -> Tuple[str, str]:
    """
    Return (code, one_line_reason).
    Heuristic mapping to user A/B/C/D narrative (not mutually exclusive; first match wins).
    """
    hm = s.get("has_matched_prior_mean_delta_iou")
    nm = s.get("no_matched_prior_mean_delta_iou")
    ol_n = s.get("category_out_library_but_hit_prior_count", 0)
    ol_m = s.get("category_out_library_but_hit_prior_mean_delta_iou")
    mdp = s.get("global_mean_delta_pred_area")
    mpr = s.get("global_mean_delta_precision")
    mrc = s.get("global_mean_delta_recall")

    if hm is None or nm is None or (isinstance(hm, float) and math.isnan(hm)):
        return "B", "insufficient split stats; defaulting to format/control caution"

    gap = hm - nm
    if gap >= 0.004 and abs(nm) < 0.0015 and (ol_m is None or math.isnan(ol_m) or ol_m >= -0.003):
        return "A", "prior-hit samples gain more; no-hit flat; out-of-library hits not strongly harmful"

    if abs(gap) < 0.002 and abs(hm) > 0.002 and abs(nm) > 0.002:
        return "B", "hit and no-hit strata both move → likely prompt-format / global shift"

    if (isinstance(ol_m, float) and ol_m <= -0.004 and ol_n >= 5) or (
        s.get("overpass_hit_prior_count", 0) >= 15
        and isinstance(s.get("overpass_hit_prior_mean_delta_iou"), float)
        and s["overpass_hit_prior_mean_delta_iou"] < -0.008
    ):
        return "C", "out-of-library but prior-hit stratum harmful or overpass hits align with drops → match audit"

    if (
        isinstance(mdp, float)
        and mdp < -80
        and isinstance(mpr, float)
        and isinstance(mrc, float)
        and mpr > max(0.003, mrc * 1.4 if mrc != 0 else 0.003)
    ):
        return "D", "global pred-area shrink with precision-led delta → check overshrink on v2.1 probe"

    if gap > 0.002 and hm > 0:
        return "A", "prior-hit samples improve more than no-hit (moderate confidence)"

    return "C", "mixed signals; tighten matching / inspect suspicious list before v2.1"


def main() -> None:
    repo = _repo_root()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    ap = argparse.ArgumentParser(
        description="Analyze prior hit groups vs metric deltas (public_semantic_v2, read-only)."
    )
    ap.add_argument(
        "--per-sample-jsonl",
        type=Path,
        default=Path("outputs/source/public_semantic_v2_analysis/per_sample_iou_diff.jsonl"),
        help="Input per-sample diff jsonl (see analyze_public_semantic_v2_per_sample.py).",
    )
    ap.add_argument(
        "--semantic-library",
        type=Path,
        default=Path("configs/concept_public_semantic_library_v2.json"),
        help="concept_public_semantic_library_v2.json path.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/source/public_semantic_v2_prior_hit_analysis/"),
        help="Directory for all outputs (created if missing).",
    )
    ap.add_argument(
        "--refs-pkl",
        type=Path,
        default=None,
        help="Optional refs(unc).p for expression/category recovery.",
    )
    ap.add_argument(
        "--instances-json",
        type=Path,
        default=None,
        help="Optional instances.json (RRSISD) for image H×W to compute foreground_ratio.",
    )
    ap.add_argument(
        "--details-json",
        type=Path,
        default=None,
        help="Optional rrsisd_test_metrics.json (or any JSON with top-level 'details' list) for description fallback.",
    )
    ap.add_argument(
        "--rrsisd-data-root",
        type=Path,
        default=None,
        help="If set, defaults refs to <root>/rrsisd/refs(unc).p and instances to <root>/rrsisd/instances.json unless overridden.",
    )
    ap.add_argument(
        "--top-k-suspicious",
        type=int,
        default=0,
        help="If >0, keep only the k worst suspicious rows by delta_iou then ref_id.",
    )
    args = ap.parse_args()

    def resolve(p: Optional[Path]) -> Optional[Path]:
        if p is None:
            return None
        p = Path(p)
        return p if p.is_absolute() else (repo / p).resolve()

    per_sample_path = resolve(args.per_sample_jsonl)
    sem_path = resolve(args.semantic_library)
    out_dir = resolve(args.output_dir)
    assert per_sample_path is not None and sem_path is not None and out_dir is not None

    refs_pkl = resolve(args.refs_pkl)
    instances_json = resolve(args.instances_json)
    details_json = resolve(args.details_json)
    if args.rrsisd_data_root:
        root = resolve(args.rrsisd_data_root)
        assert root is not None
        if refs_pkl is None:
            refs_pkl = root / "rrsisd" / "refs(unc).p"
        if instances_json is None:
            instances_json = root / "rrsisd" / "instances.json"

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(sem_path, "r", encoding="utf-8") as f:
        library = json.load(f)
    mp = library.get("match_policy")
    if not isinstance(mp, dict):
        warnings.warn("semantic library missing match_policy object", UserWarning)

    pipe = _load_explicitization_pipeline()
    lib = pipe.load_concept_public_semantic_library(str(sem_path))
    lib_labels = _build_library_category_labels(lib)

    test_refs: Optional[List[dict]] = None
    cat_id_to_name: Dict[int, str] = {}
    if instances_json is not None and instances_json.is_file():
        with open(instances_json, "r", encoding="utf-8") as f:
            inst = json.load(f)
        cat_id_to_name = {c["id"]: c["name"] for c in inst.get("categories", []) if "id" in c}
    if refs_pkl is not None and refs_pkl.is_file():
        with open(refs_pkl, "rb") as f:
            refs = pickle.load(f)
        test_refs = [r for r in refs if r.get("split") == "test"]
    elif refs_pkl:
        warnings.warn(f"refs pkl not found: {refs_pkl}", UserWarning)

    details_by_idx: Dict[int, dict] = {}
    if details_json is not None and details_json.is_file():
        with open(details_json, "r", encoding="utf-8") as f:
            dj = json.load(f)
        for row in dj.get("details", []) or []:
            if isinstance(row, dict) and "idx" in row:
                details_by_idx[int(row["idx"])] = row

    imap: Dict[int, int] = {}
    if instances_json is not None and instances_json.is_file():
        imap = _image_area_map(instances_json)
    elif instances_json:
        warnings.warn(f"instances.json not found: {instances_json}", UserWarning)
    else:
        warnings.warn(
            "No --instances-json / --rrsisd-data-root instances: foreground_ratio / area_bucket default to unknown.",
            UserWarning,
        )

    field_log: Dict[str, str] = {}
    enriched_warnings: List[str] = []

    processed: List[Dict[str, Any]] = []
    with open(per_sample_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON at {per_sample_path}:{line_no}: {e}") from e
            if not isinstance(raw, dict):
                continue
            fl: Dict[str, str] = {}
            c = _canonicalize_per_sample_row(raw, fl)
            for fk, fv in fl.items():
                field_log.setdefault(fk, fv)

            idx = c.get("idx")
            if idx is None:
                enriched_warnings.append(f"line {line_no}: missing idx-like field, skipped")
                continue

            expr = c.get("expression")
            cname = c.get("category_name")
            image_id = _get_first(raw, ("image_id",))
            try:
                image_id_i = int(image_id) if image_id is not None else None
            except (TypeError, ValueError):
                image_id_i = None

            if (not expr or not str(expr).strip()) and test_refs is not None:
                e2, cn2, _ = _enrich_from_refs(int(idx), test_refs, cat_id_to_name)
                if e2:
                    expr = e2
                    c["expression"] = expr
                if (not cname or not str(cname).strip()) and cn2:
                    cname = cn2
                    c["category_name"] = cname

            if (not expr or not str(expr).strip()) and details_by_idx:
                e3 = _enrich_from_details(int(idx), details_by_idx)
                if e3:
                    expr = e3
                    c["expression"] = expr

            if not expr or not str(expr).strip():
                enriched_warnings.append(f"idx={idx}: missing expression after enrichment")

            expr_s = (expr or "").strip()
            ctx = pipe.retrieve_concept_semantics(expr_s, lib, private_library_for_audit=None)
            matched_rows = [x for x in (ctx.get("matched_concepts") or []) if isinstance(x, dict)]
            matched_concepts = []
            for m in matched_rows:
                lab = m.get("concept")
                if isinstance(lab, str) and lab.strip():
                    matched_concepts.append(lab.strip())

            in_lib = _category_in_library(str(cname or ""), lib_labels)
            has_prior = len(matched_concepts) > 0
            cat_hit = in_lib and has_prior
            cat_miss = in_lib and not has_prior
            out_hit = (not in_lib) and has_prior

            gt_area = c.get("gt_area", float("nan"))
            fr = c.get("foreground_ratio", float("nan"))
            if math.isnan(fr):
                fr = _foreground_ratio(_float_or(gt_area, float("nan")), image_id_i, imap)
            bucket = _area_bucket(fr)

            rec_out = {
                "idx": int(idx),
                "ref_id": int(c.get("ref_id", 0)),
                "category_name": str(cname or ""),
                "expression": expr_s,
                "matched_concepts": matched_concepts,
                "matched_concept_count": len(matched_concepts),
                "has_matched_prior": has_prior,
                "is_category_in_library": in_lib,
                "category_in_library_and_hit_prior": cat_hit,
                "category_in_library_but_no_hit_prior": cat_miss,
                "category_out_library_but_hit_prior": out_hit,
                "baseline_iou": c.get("baseline_iou"),
                "v2_iou": c.get("v2_iou"),
                "delta_iou": c.get("delta_iou"),
                "baseline_precision": c.get("baseline_precision"),
                "v2_precision": c.get("v2_precision"),
                "delta_precision": c.get("delta_precision"),
                "baseline_recall": c.get("baseline_recall"),
                "v2_recall": c.get("v2_recall"),
                "delta_recall": c.get("delta_recall"),
                "baseline_pred_area": c.get("baseline_pred_area"),
                "v2_pred_area": c.get("v2_pred_area"),
                "delta_pred_area": c.get("delta_pred_area"),
                "gt_area": gt_area if not math.isnan(_float_or(gt_area, float("nan"))) else None,
                "foreground_ratio": fr if not math.isnan(fr) else None,
                "area_bucket": bucket,
            }
            processed.append(rec_out)

    per_sample_out = out_dir / "per_sample_prior_hit_diff.jsonl"
    with open(per_sample_out, "w", encoding="utf-8") as f:
        for r in processed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- prior_hit_group_metrics ---
    def pick(pred: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
        return [r for r in processed if pred(r)]

    group_defs: List[Tuple[str, Callable[[Dict[str, Any]], bool]]] = [
        ("all", lambda r: True),
        ("has_matched_prior=true", lambda r: r["has_matched_prior"]),
        ("has_matched_prior=false", lambda r: not r["has_matched_prior"]),
        ("is_category_in_library=true", lambda r: r["is_category_in_library"]),
        ("is_category_in_library=false", lambda r: not r["is_category_in_library"]),
        (
            "category_in_library_and_hit_prior",
            lambda r: r["category_in_library_and_hit_prior"],
        ),
        (
            "category_in_library_but_no_hit_prior",
            lambda r: r["category_in_library_but_no_hit_prior"],
        ),
        (
            "category_out_library_but_hit_prior",
            lambda r: r["category_out_library_but_hit_prior"],
        ),
        ("area_bucket=tiny", lambda r: r["area_bucket"] == "tiny"),
        ("area_bucket=small", lambda r: r["area_bucket"] == "small"),
        ("area_bucket=medium", lambda r: r["area_bucket"] == "medium"),
        ("area_bucket=large", lambda r: r["area_bucket"] == "large"),
    ]

    group_csv_rows: List[Dict[str, Any]] = []
    for gname, pred in group_defs:
        sub = pick(pred)
        m = _aggregate_metrics(sub)
        m = {"group_name": gname, **m}
        group_csv_rows.append(m)
    _write_group_csv(out_dir / "prior_hit_group_metrics.csv", group_csv_rows)

    # --- concept_hit_metrics (repeat per concept hit) ---
    concept_samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    concept_cat_counter: Dict[str, Counter] = defaultdict(Counter)
    for r in processed:
        for cn in r["matched_concepts"]:
            concept_samples[cn].append(r)
            concept_cat_counter[cn][r["category_name"]] += 1

    concept_rows: List[Dict[str, Any]] = []
    for concept, samples in sorted(concept_samples.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        di = [float(x["delta_iou"]) for x in samples if not math.isnan(float(x.get("delta_iou") or float("nan")))]
        dpr = [float(x["delta_precision"]) for x in samples if not math.isnan(float(x.get("delta_precision") or float("nan")))]
        dre = [float(x["delta_recall"]) for x in samples if not math.isnan(float(x.get("delta_recall") or float("nan")))]
        dpa = [float(x["delta_pred_area"]) for x in samples if not math.isnan(float(x.get("delta_pred_area") or float("nan")))]
        pos = sum(1 for x in di if x > 0)
        neg = sum(1 for x in di if x < 0)
        n = len(samples)
        concept_rows.append(
            {
                "concept": concept,
                "hit_sample_count": n,
                "mean_delta_iou": _mean(di),
                "mean_delta_precision": _mean(dpr),
                "mean_delta_recall": _mean(dre),
                "mean_delta_pred_area": _mean(dpa),
                "positive_sample_count": pos,
                "negative_sample_count": neg,
                "positive_ratio": pos / n if n else None,
                "top_categories_by_count": _top_cat_str(concept_cat_counter[concept], 8),
            }
        )

    with open(out_dir / "concept_hit_metrics.csv", "w", newline="", encoding="utf-8") as f:
        cols = list(concept_rows[0].keys()) if concept_rows else [
            "concept",
            "hit_sample_count",
            "mean_delta_iou",
            "mean_delta_precision",
            "mean_delta_recall",
            "mean_delta_pred_area",
            "positive_sample_count",
            "negative_sample_count",
            "positive_ratio",
            "top_categories_by_count",
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in concept_rows:
            out = dict(row)
            for k in ("mean_delta_iou", "mean_delta_precision", "mean_delta_recall", "mean_delta_pred_area", "positive_ratio"):
                out[k] = _fmt_float(out.get(k)) if out.get("hit_sample_count", 0) else ""
            w.writerow(out)

    # --- category_prior_cross_metrics ---
    cross_map: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in processed:
        key = (r["category_name"], ",".join(sorted(set(r["matched_concepts"]))) if r["matched_concepts"] else "")
        cross_map[key].append(r)

    cross_rows: List[Dict[str, Any]] = []
    for (cname, mjoin), samples in sorted(cross_map.items(), key=lambda kv: (-len(kv[1]), kv[0][0], kv[0][1])):
        di = [float(s["delta_iou"]) for s in samples if not math.isnan(float(s.get("delta_iou") or float("nan")))]
        dpr = [float(s["delta_precision"]) for s in samples if not math.isnan(float(s.get("delta_precision") or float("nan")))]
        dre = [float(s["delta_recall"]) for s in samples if not math.isnan(float(s.get("delta_recall") or float("nan")))]
        dpa = [float(s["delta_pred_area"]) for s in samples if not math.isnan(float(s.get("delta_pred_area") or float("nan")))]
        pos = sum(1 for x in di if x > 0)
        neg = sum(1 for x in di if x < 0)
        n = len(samples)
        cross_rows.append(
            {
                "category_name": cname,
                "matched_concepts_joined": mjoin,
                "sample_count": n,
                "mean_delta_iou": _mean(di),
                "mean_delta_precision": _mean(dpr),
                "mean_delta_recall": _mean(dre),
                "mean_delta_pred_area": _mean(dpa),
                "positive_sample_count": pos,
                "negative_sample_count": neg,
            }
        )

    with open(out_dir / "category_prior_cross_metrics.csv", "w", newline="", encoding="utf-8") as f:
        cols = [
            "category_name",
            "matched_concepts_joined",
            "sample_count",
            "mean_delta_iou",
            "mean_delta_precision",
            "mean_delta_recall",
            "mean_delta_pred_area",
            "positive_sample_count",
            "negative_sample_count",
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in cross_rows:
            o = dict(row)
            for k in ("mean_delta_iou", "mean_delta_precision", "mean_delta_recall", "mean_delta_pred_area"):
                o[k] = _fmt_float(o[k]) if o["sample_count"] else ""
            w.writerow(o)

    # --- suspicious ---
    suspicious: List[Dict[str, Any]] = []

    def add_sus(r: Dict[str, Any], reasons: List[str]) -> None:
        suspicious.append(
            {
                "idx": r["idx"],
                "ref_id": r["ref_id"],
                "category_name": r["category_name"],
                "expression": r["expression"],
                "matched_concepts": r["matched_concepts"],
                "baseline_iou": r["baseline_iou"],
                "v2_iou": r["v2_iou"],
                "delta_iou": r["delta_iou"],
                "delta_precision": r["delta_precision"],
                "delta_recall": r["delta_recall"],
                "delta_pred_area": r["delta_pred_area"],
                "suspicious_reasons": reasons,
            }
        )

    for r in processed:
        reasons: List[str] = []
        in_lib = r["is_category_in_library"]
        matched = r["matched_concepts"]
        di = _float_or(r.get("delta_iou"), float("nan"))
        dpa = _float_or(r.get("delta_pred_area"), float("nan"))
        drc = _float_or(r.get("delta_recall"), float("nan"))

        if not in_lib and matched:
            reasons.append("category_out_library_but_matched_prior")
        cnm = _norm_label(r["category_name"])
        if cnm in {"overpass", "chimney"} and matched:
            reasons.append("overpass_or_chimney_with_prior_hit")
        if (not math.isnan(di) and di < -0.1) and matched:
            reasons.append("large_iou_drop_with_prior")
        if (not math.isnan(di) and di < -0.1) and (not math.isnan(dpa) and dpa < 0):
            reasons.append("large_iou_drop_with_pred_area_shrink")
        if r["has_matched_prior"] and (not math.isnan(drc) and drc < -0.1):
            reasons.append("prior_hit_with_recall_drop")

        if reasons:
            add_sus(r, reasons)

    suspicious.sort(
        key=lambda x: (_float_or(x.get("delta_iou"), 0.0), int(x.get("ref_id") or 0))
    )
    if args.top_k_suspicious and args.top_k_suspicious > 0:
        suspicious = suspicious[: args.top_k_suspicious]

    with open(out_dir / "suspicious_mismatch_samples.jsonl", "w", encoding="utf-8") as f:
        for r in suspicious:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- summary ---
    all_rows = processed
    n_total = len(all_rows)
    di_all = [float(r["delta_iou"]) for r in all_rows if not math.isnan(float(r.get("delta_iou") or float("nan")))]
    hit_rows = [r for r in all_rows if r["has_matched_prior"]]
    noh_rows = [r for r in all_rows if not r["has_matched_prior"]]
    ol_rows = [r for r in all_rows if r["category_out_library_but_hit_prior"]]

    def mean_di(rs: List[Dict[str, Any]]) -> float:
        xs = [float(r["delta_iou"]) for r in rs if not math.isnan(float(r.get("delta_iou") or float("nan")))]
        return _mean(xs)

    op_hit = [r for r in all_rows if _norm_label(r["category_name"]) == "overpass" and r["matched_concepts"]]
    ch_hit = [r for r in all_rows if _norm_label(r["category_name"]) == "chimney" and r["matched_concepts"]]

    global_dpr = _mean(
        [float(r["delta_precision"]) for r in all_rows if not math.isnan(float(r.get("delta_precision") or float("nan")))]
    )
    global_drc = _mean(
        [float(r["delta_recall"]) for r in all_rows if not math.isnan(float(r.get("delta_recall") or float("nan")))]
    )
    global_dpa = _mean(
        [float(r["delta_pred_area"]) for r in all_rows if not math.isnan(float(r.get("delta_pred_area") or float("nan")))]
    )

    pos_concepts = sorted(
        [(r["concept"], r["mean_delta_iou"], r["hit_sample_count"]) for r in concept_rows],
        key=lambda t: (t[1] if not math.isnan(t[1]) else -1e9, t[2]),
        reverse=True,
    )
    neg_concepts = sorted(
        [(r["concept"], r["mean_delta_iou"], r["hit_sample_count"]) for r in concept_rows],
        key=lambda t: (t[1] if not math.isnan(t[1]) else 1e9, -t[2]),
    )

    bucket_summary = {name: sum(1 for r in all_rows if r["area_bucket"] == name) for name in ("tiny", "small", "medium", "large", "unknown")}

    summary_core = {
        "total_samples": n_total,
        "all_mean_delta_iou": _mean(di_all),
        "has_matched_prior_sample_count": len(hit_rows),
        "no_matched_prior_sample_count": len(noh_rows),
        "has_matched_prior_mean_delta_iou": mean_di(hit_rows),
        "no_matched_prior_mean_delta_iou": mean_di(noh_rows),
        "category_out_library_but_hit_prior_count": len(ol_rows),
        "category_out_library_but_hit_prior_mean_delta_iou": mean_di(ol_rows),
        "overpass_hit_prior_count": len(op_hit),
        "overpass_hit_prior_mean_delta_iou": mean_di(op_hit),
        "chimney_hit_prior_count": len(ch_hit),
        "chimney_hit_prior_mean_delta_iou": mean_di(ch_hit),
        "improved_precision_vs_recall_summary": {
            "global_mean_delta_precision": global_dpr,
            "global_mean_delta_recall": global_drc,
            "interpretation": "precision_delta > recall_delta suggests cleaner masks / less spill if paired with pred-area shrink",
        },
        "shrinkage_summary": {
            "global_mean_delta_pred_area": global_dpa,
            "note": "negative means average predicted foreground pixels decreased vs baseline",
        },
        "area_bucket_summary": bucket_summary,
        "top_positive_concepts": [{"concept": a, "mean_delta_iou": b, "hit_sample_count": c} for a, b, c in pos_concepts[:8]],
        "top_negative_concepts": [{"concept": a, "mean_delta_iou": b, "hit_sample_count": c} for a, b, c in neg_concepts[:8]],
        "concept_level_counting_note": (
            "Each sample with k matched concepts contributes k times to per-concept hit_sample_count / means."
        ),
    }

    step_stats = {
        **summary_core,
        "global_mean_delta_pred_area": global_dpa,
        "global_mean_delta_precision": global_dpr,
        "global_mean_delta_recall": global_drc,
    }
    code, reason = _choose_recommended_step(step_stats)
    summary = {
        "field_aliases_used": field_log,
        "library_match_policy": mp if isinstance(mp, dict) else None,
        "retrieval_backend": "rrsisd_explicitization_pipeline.retrieve_concept_semantics",
        "enriched_warnings": enriched_warnings[:50],
        **summary_core,
        "recommended_next_step": code,
        "recommended_next_step_reason": reason,
    }

    def _json_sanitize(o: Any) -> Any:
        if isinstance(o, dict):
            return {k: _json_sanitize(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_json_sanitize(v) for v in o]
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        return o

    with open(out_dir / "prior_hit_analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(summary), f, indent=2, ensure_ascii=False)

    # stdout
    print(f"output_dir={out_dir}")
    print(f"total_samples={n_total}")
    print(
        f"mean_delta_iou: has_matched_prior={summary_core['has_matched_prior_mean_delta_iou']:.6f} "
        f"no_matched_prior={summary_core['no_matched_prior_mean_delta_iou']:.6f}"
    )
    print(
        f"category_out_library_but_hit_prior: n={summary_core['category_out_library_but_hit_prior_count']} "
        f"mean_delta_iou={summary_core['category_out_library_but_hit_prior_mean_delta_iou']:.6f}"
    )
    print(
        f"overpass_hit_prior: n={summary_core['overpass_hit_prior_count']} "
        f"mean_delta_iou={summary_core['overpass_hit_prior_mean_delta_iou']:.6f}"
    )
    ch_m = summary_core["chimney_hit_prior_mean_delta_iou"]
    ch_s = f"{ch_m:.6f}" if isinstance(ch_m, (int, float)) and not math.isnan(float(ch_m)) else "n/a"
    print(
        f"chimney_hit_prior: n={summary_core['chimney_hit_prior_count']} mean_delta_iou={ch_s}"
    )
    print(f"recommended_next_step={code} ({reason})")


if __name__ == "__main__":
    main()
