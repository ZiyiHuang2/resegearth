import argparse
import csv
import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
_mod_path = Path(project_root) / "tools" / "debug_remoteclip_prior.py"
_spec = importlib.util.spec_from_file_location("debug_remoteclip_prior_mod", _mod_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Failed to load module from {_mod_path}")
_debug_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_debug_mod)
run_one = _debug_mod.run_one


def top3_spatial_variance(topk_patches):
    pts = [p["center_in_prior_space"] for p in topk_patches[:3]]
    if len(pts) < 2:
        return 0.0
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs) / len(xs)
    vy = sum((y - my) ** 2 for y in ys) / len(ys)
    return float(vx + vy)


def summarize_rows(rows):
    n = len(rows)
    if n == 0:
        return {}

    def mean(k):
        vals = [r[k] for r in rows if r.get(k) is not None]
        return float(sum(vals) / len(vals)) if vals else 0.0

    def median(k):
        vals = sorted([r[k] for r in rows if r.get(k) is not None])
        if not vals:
            return 0.0
        m = len(vals) // 2
        return float(vals[m]) if len(vals) % 2 == 1 else float((vals[m - 1] + vals[m]) / 2)

    return {
        "total_samples": n,
        "patch_grid_size": rows[0].get("patch_grid_size", []),
        "top1_hit_rate": mean("top1_hit"),
        "top3_hit_rate": mean("top3_hit"),
        "top5_hit_rate": mean("top5_hit"),
        "mean_gt_center_patch_rank": mean("gt_center_patch_rank"),
        "median_gt_center_patch_rank": median("gt_center_patch_rank"),
        "mean_top1_bbox_iou": mean("top1_bbox_iou"),
        "mean_top3_union_bbox_iou": mean("top3_union_bbox_iou"),
        "mean_top5_union_bbox_iou": mean("top5_union_bbox_iou"),
        "mean_global_conf": mean("global_conf"),
        "mean_normalized_entropy": mean("normalized_entropy"),
        "mean_top3_spatial_variance": mean("top3_spatial_variance"),
        "median_top3_spatial_variance": median("top3_spatial_variance"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--remoteclip-weight", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--clip-input-sizes", type=int, nargs="+", default=[224, 336])
    parser.add_argument("--save-npy", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    samples = []
    with open(args.jsonl, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    rows = []
    for s in samples:
        sample_id = str(s.get("sample_id", Path(s["image_path"]).stem))
        expr = s["expression"]
        cat = s.get("category", "")
        diff = s.get("difficulty", "")
        mask_path = s.get("mask_path")

        for res in args.clip_input_sizes:
            sample_dir = out_root / sample_id / str(res)
            result = run_one(
                image_path=s["image_path"],
                expression=expr,
                remoteclip_weight=args.remoteclip_weight,
                output_dir=str(sample_dir),
                model_name=args.model_name,
                device=args.device,
                topk=args.topk,
                temperature=args.temperature,
                clip_input_size=int(res),
                save_npy=args.save_npy,
                category=cat,
                sample_id=sample_id,
                mask_path=mask_path,
            )

            topk = result["topk"]
            align = result.get("alignment")
            stats = result["stats"]
            row = {
                "sample_id": sample_id,
                "resolution": int(res),
                "category": cat,
                "difficulty": diff,
                "patch_grid_size": stats["patch_grid_size"],
                "global_conf": stats["global_conf"],
                "normalized_entropy": stats["normalized_entropy"],
                "top3_spatial_variance": top3_spatial_variance(topk),
            }
            if align:
                row.update({
                    "top1_hit": int(bool(align["top1_patch_hits_gt"])),
                    "top3_hit": int(bool(align["top3_patch_hits_gt"])),
                    "top5_hit": int(bool(align["top5_patch_hits_gt"])),
                    "gt_center_patch_rank": int(align["gt_center_patch_rank"]),
                    "top1_bbox_iou": float(align["top1_patch_bbox_iou_with_gt_bbox"]),
                    "top3_union_bbox_iou": float(align["top3_union_bbox_iou_with_gt_bbox"]),
                    "top5_union_bbox_iou": float(align["top5_union_bbox_iou_with_gt_bbox"]),
                })
            else:
                row.update({
                    "top1_hit": None,
                    "top3_hit": None,
                    "top5_hit": None,
                    "gt_center_patch_rank": None,
                    "top1_bbox_iou": None,
                    "top3_union_bbox_iou": None,
                    "top5_union_bbox_iou": None,
                })
            rows.append(row)

    by_res = defaultdict(list)
    by_res_cat = defaultdict(lambda: defaultdict(list))
    by_res_diff = defaultdict(lambda: defaultdict(list))
    for r in rows:
        rs = str(r["resolution"])
        by_res[rs].append(r)
        if r.get("category"):
            by_res_cat[rs][r["category"]].append(r)
        if r.get("difficulty"):
            by_res_diff[rs][r["difficulty"]].append(r)

    summary = {"total_samples": len(samples), "by_resolution": {}}
    for rs, vals in by_res.items():
        s = summarize_rows(vals)
        s["resolution"] = int(rs)
        s["by_category"] = {k: summarize_rows(v) for k, v in by_res_cat[rs].items()}
        s["by_difficulty"] = {k: summarize_rows(v) for k, v in by_res_diff[rs].items()}
        summary["by_resolution"][rs] = s

    with open(out_root / "summary.json", "w", encoding="utf-8-sig") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    csv_fields = ["sample_id", "resolution", "category", "difficulty", "patch_grid_size", "global_conf", "normalized_entropy", "top3_spatial_variance", "top1_hit", "top3_hit", "top5_hit", "gt_center_patch_rank", "top1_bbox_iou", "top3_union_bbox_iou", "top5_union_bbox_iou"]
    with open(out_root / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for r in rows:
            rr = dict(r)
            rr["patch_grid_size"] = str(rr["patch_grid_size"])
            w.writerow(rr)

    for rs in sorted(summary["by_resolution"].keys(), key=lambda x: int(x)):
        s = summary["by_resolution"][rs]
        print(f"resolution={rs} patch_grid_size={s.get('patch_grid_size')} top1_hit_rate={s.get('top1_hit_rate')} top3_hit_rate={s.get('top3_hit_rate')} top5_hit_rate={s.get('top5_hit_rate')} mean_gt_center_patch_rank={s.get('mean_gt_center_patch_rank')} mean_top3_spatial_variance={s.get('mean_top3_spatial_variance')} mean_global_conf={s.get('mean_global_conf')} mean_normalized_entropy={s.get('mean_normalized_entropy')}")

    if "224" in summary["by_resolution"] and "336" in summary["by_resolution"]:
        h224 = summary["by_resolution"]["224"].get("top3_hit_rate", 0.0)
        h336 = summary["by_resolution"]["336"].get("top3_hit_rate", 0.0)
        if h336 > h224 + 0.05:
            print("[INFO] Higher CLIP input resolution improves prior-target alignment. Consider using ViT-B/32 with larger input size before switching to ViT-L/14.")
        elif h224 < 0.2 and h336 < 0.2:
            print("[WARNING] RemoteCLIP ViT-B/32 prior remains weak even with higher input resolution. Consider ViT-L/14 or alternative prior strategy.")

    mean_vars = [v.get("mean_top3_spatial_variance", 0.0) for v in summary["by_resolution"].values() if isinstance(v, dict)]
    if mean_vars and (sum(mean_vars)/len(mean_vars) > 5000):
        print("[WARNING] Prior is highly scattered and uncorrelated with targets.")


if __name__ == "__main__":
    main()

