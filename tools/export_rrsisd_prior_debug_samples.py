import argparse
import json
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image, ImageDraw


def load_rrsisd(rrsisd_root):
    root = Path(rrsisd_root)
    refs_path = root / "rrsisd" / "refs(unc).p"
    instances_path = root / "rrsisd" / "instances.json"
    image_dir = root / "images" / "rrsisd" / "JPEGImages"

    with open(refs_path, "rb") as f:
        refs = pickle.load(f)
    with open(instances_path, "r", encoding="utf-8") as f:
        instances = json.load(f)

    anns = {a["id"]: a for a in instances.get("annotations", [])}
    imgs = {i["id"]: i for i in instances.get("images", [])}
    cats = {c["id"]: c.get("name", "unknown") for c in instances.get("categories", [])}
    return refs, anns, imgs, cats, image_dir, refs_path, instances_path


def decode_polygon_to_mask(segmentation, h, w):
    mask_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_img)
    for poly in segmentation:
        if not isinstance(poly, list) or len(poly) < 6:
            continue
        pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
        draw.polygon(pts, outline=1, fill=1)
    return np.array(mask_img, dtype=np.uint8)


def _ensure_hw(mask, h, w):
    if mask.shape != (h, w):
        mask = np.array(Image.fromarray((mask > 0).astype(np.uint8)).resize((w, h), resample=Image.NEAREST), dtype=np.uint8)
    return mask


def decode_segmentation(ann, h, w):
    seg = ann.get("segmentation")
    if seg is None:
        return None, "missing_segmentation"

    try:
        from pycocotools import mask as mask_utils  # type: ignore
    except Exception:
        mask_utils = None

    if isinstance(seg, list):
        if len(seg) == 0:
            return None, "missing_segmentation"
        if isinstance(seg[0], dict):
            if mask_utils is None:
                return None, "missing_segmentation"
            m_all = np.zeros((h, w), dtype=np.uint8)
            for r in seg:
                m = mask_utils.decode(r)
                if m.ndim == 3:
                    m = m[..., 0]
                m = _ensure_hw((m > 0).astype(np.uint8), h, w)
                m_all = np.logical_or(m_all, m > 0)
            return m_all.astype(np.uint8), None
        m = decode_polygon_to_mask(seg, h, w)
        return _ensure_hw(m, h, w), None

    if isinstance(seg, dict):
        if mask_utils is None:
            return None, "missing_segmentation"
        m = mask_utils.decode(seg)
        if m.ndim == 3:
            m = m[..., 0]
        m = _ensure_hw((m > 0).astype(np.uint8), h, w)
        return m, None

    return None, "missing_segmentation"


def pick_expression(sent, sent_idx):
    if isinstance(sent, dict):
        txt = sent.get("sent") or sent.get("raw") or ""
        sid = sent.get("sent_id", sent_idx)
    else:
        txt = str(sent)
        sid = sent_idx
    return txt.strip(), sid


def difficulty_from_area(foreground_ratio, mask_pixel_area):
    if foreground_ratio is None or mask_pixel_area is None:
        return "unknown"
    if foreground_ratio >= 0.05 or mask_pixel_area >= 50000:
        return "large_object"
    if foreground_ratio >= 0.01 or mask_pixel_area >= 10000:
        return "medium_object"
    if foreground_ratio < 0.01 and mask_pixel_area < 10000:
        return "small_object"
    return "unknown"


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rrsisd-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--small-object-count", type=int, default=20)
    ap.add_argument("--medium-object-count", type=int, default=15)
    ap.add_argument("--large-object-count", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    refs, anns, imgs, cats, image_dir, refs_path, instances_path = load_rrsisd(args.rrsisd_root)

    invalid = []
    exported = []
    duplicate_sample_id = 0
    stats_counter = Counter()

    for ref in refs:
        if ref.get("split") != args.split:
            continue

        ref_id = ref.get("ref_id")
        ann_id = ref.get("ann_id")
        ref_image_id = ref.get("image_id")

        ann = anns.get(ann_id)
        if ann is None:
            stats_counter["ann_id_not_found_count"] += 1
            invalid.append({"reason": "ann_id_not_found", "ref_id": ref_id, "ann_id": ann_id, "image_id": ref_image_id, "expression": "", "details": "annotation missing"})
            continue

        ann_image_id = ann.get("image_id")
        if ann_image_id != ref_image_id:
            stats_counter["image_id_mismatch_count"] += 1
            invalid.append({"reason": "image_id_mismatch", "ref_id": ref_id, "ann_id": ann_id, "image_id": ref_image_id, "expression": "", "details": f"ann.image_id={ann_image_id}"})
            continue

        img_info = imgs.get(ref_image_id)
        if img_info is None:
            stats_counter["missing_image_count"] += 1
            invalid.append({"reason": "missing_image", "ref_id": ref_id, "ann_id": ann_id, "image_id": ref_image_id, "expression": "", "details": "image info missing"})
            continue

        file_name = img_info.get("file_name")
        img_path = image_dir / file_name
        if not img_path.exists():
            stats_counter["missing_image_count"] += 1
            invalid.append({"reason": "missing_image", "ref_id": ref_id, "ann_id": ann_id, "image_id": ref_image_id, "expression": "", "details": str(img_path)})
            continue

        w = int(img_info.get("width", 0))
        h = int(img_info.get("height", 0))
        if w <= 0 or h <= 0:
            with Image.open(img_path) as im:
                w, h = im.size

        mask, err = decode_segmentation(ann, h, w)
        if err is not None or mask is None:
            stats_counter["missing_segmentation_count"] += 1
            invalid.append({"reason": "missing_segmentation", "ref_id": ref_id, "ann_id": ann_id, "image_id": ref_image_id, "expression": "", "details": "cannot decode segmentation"})
            continue

        if int(mask.sum()) == 0:
            stats_counter["empty_mask_count"] += 1
            invalid.append({"reason": "empty_mask", "ref_id": ref_id, "ann_id": ann_id, "image_id": ref_image_id, "expression": "", "details": "decoded mask all background"})
            continue

        cat_id = ann.get("category_id", ann.get("categories_id")); cat_name = cats.get(cat_id, "unknown")
        bbox = ann.get("bbox", [0, 0, 0, 0])
        if len(bbox) != 4:
            bbox = [0, 0, 0, 0]

        mask_area = int(mask.sum())
        fg_ratio = float(mask_area / float(w * h)) if w > 0 and h > 0 else None
        bbox_area = float(max(0.0, bbox[2]) * max(0.0, bbox[3]))
        bbox_ratio = float(bbox_area / float(w * h)) if w > 0 and h > 0 else None
        diff = difficulty_from_area(fg_ratio, mask_area)

        group_id = f"{ref_image_id}_{ann_id}"
        mask_path = masks_dir / f"{group_id}.png"
        if not mask_path.exists():
            Image.fromarray((mask > 0).astype(np.uint8) * 255).save(mask_path)

        sents = ref.get("sentences", [])
        if not sents:
            stats_counter["missing_expression_count"] += 1
            invalid.append({"reason": "missing_expression", "ref_id": ref_id, "ann_id": ann_id, "image_id": ref_image_id, "expression": "", "details": "no sentences"})
            continue

        for sidx, sent in enumerate(sents):
            expr, sent_id = pick_expression(sent, sidx)
            if not expr:
                stats_counter["missing_expression_count"] += 1
                invalid.append({"reason": "missing_expression", "ref_id": ref_id, "ann_id": ann_id, "image_id": ref_image_id, "expression": "", "details": f"sent_idx={sidx}"})
                continue

            sample_id = f"{ref_image_id}_{ann_id}_{ref_id}_{sent_id}"
            if any(x.get("sample_id") == sample_id for x in exported):
                duplicate_sample_id += 1
                invalid.append({"reason": "duplicate_sample_id", "ref_id": ref_id, "ann_id": ann_id, "image_id": ref_image_id, "expression": expr, "details": sample_id})
                continue

            exported.append({
                "sample_id": sample_id,
                "group_id": group_id,
                "image_path": str(img_path.resolve()),
                "expression": expr,
                "category": cat_name if cat_name else "unknown",
                "mask_path": str(mask_path.resolve()),
                "ann_id": str(ann_id),
                "image_id": str(ref_image_id),
                "ref_id": str(ref_id),
                "sent_id": str(sent_id),
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                "area": float(ann.get("area", mask_area)),
                "mask_pixel_area": int(mask_area),
                "image_width": int(w),
                "image_height": int(h),
                "foreground_ratio": float(fg_ratio) if fg_ratio is not None else None,
                "bbox_ratio": float(bbox_ratio) if bbox_ratio is not None else None,
                "difficulty": diff,
            })

    by_diff = defaultdict(list)
    for r in exported:
        by_diff[r["difficulty"]].append(r)
    for k in by_diff:
        random.shuffle(by_diff[k])

    picked = []
    picked += by_diff.get("small_object", [])[: args.small_object_count]
    picked += by_diff.get("medium_object", [])[: args.medium_object_count]
    picked += by_diff.get("large_object", [])[: args.large_object_count]

    if len(picked) < args.max_samples:
        rest = [r for r in exported if r not in picked]
        random.shuffle(rest)
        picked += rest[: max(0, args.max_samples - len(picked))]

    if len(picked) > args.max_samples:
        picked = picked[: args.max_samples]

    valid_final = []
    seen_ids = set()
    for r in picked:
        sid = r["sample_id"]
        if sid in seen_ids:
            duplicate_sample_id += 1
            invalid.append({"reason": "duplicate_sample_id", "ref_id": r["ref_id"], "ann_id": r["ann_id"], "image_id": r["image_id"], "expression": r["expression"], "details": sid})
            continue
        seen_ids.add(sid)

        ip = Path(r["image_path"])
        mp = Path(r["mask_path"])
        if not ip.exists():
            stats_counter["missing_image_count"] += 1
            invalid.append({"reason": "missing_image", "ref_id": r["ref_id"], "ann_id": r["ann_id"], "image_id": r["image_id"], "expression": r["expression"], "details": r["image_path"]})
            continue
        if not mp.exists():
            stats_counter["missing_mask_count"] += 1
            invalid.append({"reason": "missing_mask", "ref_id": r["ref_id"], "ann_id": r["ann_id"], "image_id": r["image_id"], "expression": r["expression"], "details": r["mask_path"]})
            continue
        if not r.get("expression"):
            stats_counter["missing_expression_count"] += 1
            invalid.append({"reason": "missing_expression", "ref_id": r["ref_id"], "ann_id": r["ann_id"], "image_id": r["image_id"], "expression": "", "details": sid})
            continue
        valid_final.append(r)

    samples_jsonl = out_dir / "samples.jsonl"
    invalid_jsonl = out_dir / "invalid_samples.jsonl"
    summary_json = out_dir / "summary.json"

    write_jsonl(samples_jsonl, valid_final)
    write_jsonl(invalid_jsonl, invalid)

    diffs = Counter([r.get("difficulty", "unknown") for r in valid_final])
    cats_count = Counter([r.get("category", "unknown") for r in valid_final])

    fg_vals = sorted([r["foreground_ratio"] for r in valid_final if r.get("foreground_ratio") is not None])
    area_vals = sorted([r["mask_pixel_area"] for r in valid_final if r.get("mask_pixel_area") is not None])

    def med(vals):
        return float(median(vals)) if vals else 0.0

    group_counts = Counter([r["group_id"] for r in valid_final])
    multi_expr_group_count = sum(1 for _, c in group_counts.items() if c > 1)

    summary = {
        "total_exported": len(valid_final),
        "small_object_count": int(diffs.get("small_object", 0)),
        "medium_object_count": int(diffs.get("medium_object", 0)),
        "large_object_count": int(diffs.get("large_object", 0)),
        "unknown_count": int(diffs.get("unknown", 0)),
        "missing_mask_count": int(stats_counter.get("missing_mask_count", 0)),
        "missing_expression_count": int(stats_counter.get("missing_expression_count", 0)),
        "missing_image_count": int(stats_counter.get("missing_image_count", 0)),
        "invalid_sample_count": len(invalid),
        "ann_id_not_found_count": int(stats_counter.get("ann_id_not_found_count", 0)),
        "image_id_mismatch_count": int(stats_counter.get("image_id_mismatch_count", 0)),
        "missing_segmentation_count": int(stats_counter.get("missing_segmentation_count", 0)),
        "empty_mask_count": int(stats_counter.get("empty_mask_count", 0)),
        "duplicate_sample_id_count": int(duplicate_sample_id),
        "multi_expression_group_count": int(multi_expr_group_count),
        "unique_group_count": int(len(group_counts)),
        "rrsisd_root": str(Path(args.rrsisd_root).resolve()),
        "output_dir": str(out_dir.resolve()),
        "split": args.split,
        "category_counts": dict(cats_count),
        "difficulty_counts": dict(diffs),
        "area_stats": {
            "foreground_ratio_min": float(fg_vals[0]) if fg_vals else 0.0,
            "foreground_ratio_median": med(fg_vals),
            "foreground_ratio_max": float(fg_vals[-1]) if fg_vals else 0.0,
            "mask_pixel_area_min": int(area_vals[0]) if area_vals else 0,
            "mask_pixel_area_median": med(area_vals),
            "mask_pixel_area_max": int(area_vals[-1]) if area_vals else 0,
        },
        "sources": {
            "refs_path": str(refs_path.resolve()),
            "instances_path": str(instances_path.resolve()),
            "image_dir": str(image_dir.resolve()),
        },
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if diffs.get("small_object", 0) < args.small_object_count:
        print(f"[WARNING] small_object target not met: wanted {args.small_object_count}, got {diffs.get('small_object', 0)}")
    if diffs.get("medium_object", 0) < args.medium_object_count:
        print(f"[WARNING] medium_object target not met: wanted {args.medium_object_count}, got {diffs.get('medium_object', 0)}")
    if diffs.get("large_object", 0) < args.large_object_count:
        print(f"[WARNING] large_object target not met: wanted {args.large_object_count}, got {diffs.get('large_object', 0)}")

    print(f"Exported samples: {len(valid_final)}")
    print(f"samples.jsonl: {samples_jsonl}")
    print(f"masks dir: {masks_dir}")
    print(f"summary.json: {summary_json}")
    print(f"invalid_samples.jsonl: {invalid_jsonl}")


if __name__ == "__main__":
    main()



