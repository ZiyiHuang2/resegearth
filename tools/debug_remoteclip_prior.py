import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

_mod_path = Path(project_root) / "segearth_r2" / "model" / "prior" / "remoteclip_prior.py"
_spec = importlib.util.spec_from_file_location("remoteclip_prior_mod", _mod_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Failed to load module from {_mod_path}")
_remoteclip_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_remoteclip_mod)
RemoteCLIPPriorBranch = _remoteclip_mod.RemoteCLIPPriorBranch
topk_patch_info = _remoteclip_mod.topk_patch_info


def bbox_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    ub = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = ua + ub - inter + 1e-8
    return float(inter / union)


def union_bbox(boxes):
    if not boxes:
        return [0, 0, 0, 0]
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return [x1, y1, x2, y2]


def save_raw_grid_heatmap(grid, out):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    im = ax.imshow(grid, cmap="viridis", interpolation="nearest")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", color="white", fontsize=6)
    ax.set_title("Raw patch-text similarity grid")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_raw_overlay_mosaic(image, grid, out):
    img = image.convert("RGB")
    arr = np.asarray(img).copy()
    h, w = arr.shape[:2]
    gh, gw = grid.shape
    gmin, gmax = float(grid.min()), float(grid.max())
    norm = (grid - gmin) / (gmax - gmin + 1e-12)
    th, tw = h / gh, w / gw
    for r in range(gh):
        for c in range(gw):
            y1, y2 = int(round(r * th)), int(round((r + 1) * th))
            x1, x2 = int(round(c * tw)), int(round((c + 1) * tw))
            alpha = float(norm[r, c]) * 0.6
            arr[y1:y2, x1:x2] = (1 - alpha) * arr[y1:y2, x1:x2] + alpha * np.array([255, 0, 0], dtype=np.float32)
    out_img = Image.fromarray(arr.astype(np.uint8))
    draw = ImageDraw.Draw(out_img)
    for r in range(gh + 1):
        y = int(round(r * th))
        draw.line([(0, y), (w, y)], fill=(255, 255, 255), width=1)
    for c in range(gw + 1):
        x = int(round(c * tw))
        draw.line([(x, 0), (x, h)], fill=(255, 255, 255), width=1)
    out_img.save(out)


def save_interpolated(image, grid, heatmap_out, overlay_out):
    arr = np.asarray(image.convert("RGB"))
    h, w = arr.shape[:2]
    interp = np.array(Image.fromarray(grid.astype(np.float32), mode="F").resize((w, h), resample=Image.BILINEAR))
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.imshow(interp, cmap="viridis", interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(heatmap_out, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.imshow(arr)
    ax.imshow(interp, cmap="jet", alpha=0.45, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(overlay_out, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def compute_alignment(mask_path, prior_hw, original_hw, topk_patches, sim_vec, prob_vec, patch_grid_size):
    gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")
    gt_bin = (gt > 0).astype(np.uint8)
    oh, ow = original_hw
    ph, pw = prior_hw
    gt_resized = cv2.resize(gt_bin, (pw, ph), interpolation=cv2.INTER_NEAREST)

    ys, xs = np.where(gt_bin > 0)
    if len(xs) == 0:
        gt_bbox_orig = [0, 0, 0, 0]
    else:
        gt_bbox_orig = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

    ys2, xs2 = np.where(gt_resized > 0)
    if len(xs2) == 0:
        gt_bbox_prior = [0, 0, 0, 0]
    else:
        gt_bbox_prior = [int(xs2.min()), int(ys2.min()), int(xs2.max()) + 1, int(ys2.max()) + 1]

    gt_area_orig = int(gt_bin.sum())
    gt_area_prior = int(gt_resized.sum())

    h, w = patch_grid_size
    if gt_area_prior > 0:
        cy = float(np.mean(ys2))
        cx = float(np.mean(xs2))
    else:
        cy, cx = 0.0, 0.0
    row = min(h - 1, max(0, int((cy / max(1, ph)) * h)))
    col = min(w - 1, max(0, int((cx / max(1, pw)) * w)))
    center_idx = row * w + col

    order = np.argsort(-sim_vec)
    rank = int(np.where(order == center_idx)[0][0]) + 1

    top1_box = topk_patches[0]["bbox_in_prior_space"] if topk_patches else [0, 0, 0, 0]
    top3_union = union_bbox([p["bbox_in_prior_space"] for p in topk_patches[:3]])
    top5_union = union_bbox([p["bbox_in_prior_space"] for p in topk_patches[:5]])

    def patch_hits_gt(patch):
        x1, y1, x2, y2 = patch["bbox_in_prior_space"]
        crop = gt_resized[y1:y2, x1:x2]
        return bool(crop.size > 0 and np.any(crop > 0))

    return {
        "prior_input_size": [ph, pw],
        "original_image_size": [oh, ow],
        "gt_bbox_in_original_space": gt_bbox_orig,
        "gt_bbox_in_prior_space": gt_bbox_prior,
        "gt_area_in_original_space": gt_area_orig,
        "gt_area_in_prior_space": gt_area_prior,
        "gt_center_patch": {"row": row, "col": col, "flat_index": int(center_idx)},
        "gt_center_patch_similarity": float(sim_vec[center_idx]),
        "gt_center_patch_probability": float(prob_vec[center_idx]),
        "gt_center_patch_rank": rank,
        "top1_patch_hits_gt": patch_hits_gt(topk_patches[0]) if len(topk_patches) >= 1 else False,
        "top3_patch_hits_gt": any(patch_hits_gt(p) for p in topk_patches[:3]),
        "top5_patch_hits_gt": any(patch_hits_gt(p) for p in topk_patches[:5]),
        "top1_patch_bbox_iou_with_gt_bbox": bbox_iou(top1_box, gt_bbox_prior),
        "top3_union_bbox_iou_with_gt_bbox": bbox_iou(top3_union, gt_bbox_prior),
        "top5_union_bbox_iou_with_gt_bbox": bbox_iou(top5_union, gt_bbox_prior),
        "top1_patch_bbox_in_prior_space": top1_box,
        "top3_union_bbox_in_prior_space": top3_union,
        "top5_union_bbox_in_prior_space": top5_union,
        "iou_coordinate_space": "prior_space",
    }, gt_bin, gt_resized


def run_one(image_path, expression, remoteclip_weight, output_dir, model_name, device, topk, temperature, clip_input_size, save_npy, category="", sample_id="", mask_path=None):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    ow, oh = image.size
    image.save(out_dir / "original_image.png")

    branch = RemoteCLIPPriorBranch(
        model_name=model_name,
        weight_path=remoteclip_weight,
        device=device,
        unfreeze_last_layer=False,
        temperature=temperature,
    )
    out = branch.forward_image_expression(image_path, expression, clip_input_size=clip_input_size)

    sim = out.patch_text_sim[0].detach().cpu().numpy()
    prior = out.patch_prior[0].detach().cpu().numpy()
    prob = out.patch_prob[0].detach().cpu().numpy()
    gh, gw = out.patch_grid_size
    sim_grid = sim.reshape(gh, gw)
    prior_hw = (clip_input_size, clip_input_size)

    save_raw_grid_heatmap(sim_grid, out_dir / "raw_grid_heatmap.png")
    save_raw_overlay_mosaic(image, sim_grid, out_dir / "raw_grid_overlay_mosaic.png")
    save_interpolated(image, sim_grid, out_dir / "interpolated_heatmap.png", out_dir / "interpolated_overlay.png")

    topk_data = topk_patch_info(out.patch_text_sim[0], out.patch_prob[0], out.patch_grid_size, (oh, ow), prior_hw, topk=topk)
    with open(out_dir / "topk_patches.json", "w", encoding="utf-8") as f:
        json.dump({"topk": min(topk, gh * gw), "resolution": clip_input_size, "patch_grid_size": [gh, gw], "patches": topk_data}, f, ensure_ascii=False, indent=2)

    stats = {
        "image_path": image_path,
        "expression": expression,
        "remoteclip_weight": remoteclip_weight,
        "model_name": model_name,
        "input_image_size": [clip_input_size, clip_input_size],
        "original_image_size": [oh, ow],
        "patch_grid_size": [gh, gw],
        "num_patches": int(gh * gw),
        "min_sim": float(sim.min()),
        "max_sim": float(sim.max()),
        "mean_sim": float(sim.mean()),
        "std_sim": float(sim.std()),
        "peak_prob": float(out.peak_prob[0, 0].item()),
        "entropy": float(out.entropy[0, 0].item()),
        "normalized_entropy": float(out.normalized_entropy[0, 0].item()),
        "global_conf": float(out.global_conf[0, 0].item()),
        "temperature": float(temperature),
        "expression_truncated": bool(out.expression_truncated),
    }
    with open(out_dir / "prior_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    if save_npy:
        np.save(out_dir / "patch_text_sim.npy", sim)
        np.save(out_dir / "patch_prior.npy", prior)
        np.save(out_dir / "patch_prob.npy", prob)

    alignment = None
    if mask_path:
        alignment, gt_orig, gt_prior = compute_alignment(mask_path, prior_hw, (oh, ow), topk_data, sim, prob, (gh, gw))
        alignment.update({"sample_id": sample_id, "expression": expression, "category": category, "resolution": clip_input_size, "patch_grid_size": [gh, gw], "num_patches": int(gh * gw)})
        with open(out_dir / "prior_gt_alignment.json", "w", encoding="utf-8") as f:
            json.dump(alignment, f, ensure_ascii=False, indent=2)

        Image.fromarray((gt_orig * 255).astype(np.uint8)).save(out_dir / "gt_mask.png")
        ov = np.asarray(image.convert("RGB")).copy()
        bx = alignment["gt_bbox_in_original_space"]
        cv2.rectangle(ov, (bx[0], bx[1]), (max(bx[2]-1,bx[0]), max(bx[3]-1,bx[1])), (0,255,0), 2)
        Image.fromarray(ov).save(out_dir / "gt_bbox_overlay.png")

        ov2 = ov.copy()
        colors = [(255,0,0),(255,165,0),(255,255,0),(255,0,255),(0,255,255)]
        for i,p in enumerate(topk_data[:5]):
            b = p["bbox_in_original_space"]
            cv2.rectangle(ov2, (b[0], b[1]), (max(b[2]-1,b[0]), max(b[3]-1,b[1])), colors[i%len(colors)], 2)
        Image.fromarray(ov2).save(out_dir / "topk_vs_gt_overlay.png")

    top1 = topk_data[0] if topk_data else None
    print("patch_grid_size:", [gh, gw])
    print("num_patches:", gh * gw)
    if top1 is not None:
        print("top1_patch:", {"row": top1["row"], "col": top1["col"], "bbox": top1["bbox_in_original_space"]})
    print("global_conf:", stats["global_conf"])
    print("normalized_entropy:", stats["normalized_entropy"])
    if gh < 10 or gw < 10:
        print("WARNING: RemoteCLIP ViT-B/32 prior is very coarse. For small objects, 7x7 patch-level prior may be insufficient. Please inspect raw_grid_overlay_mosaic.png instead of relying on interpolated_overlay.png.")

    return {"stats": stats, "topk": topk_data, "alignment": alignment}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--expression", required=True)
    parser.add_argument("--remoteclip-weight", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-npy", action="store_true")
    parser.add_argument("--clip-input-size", type=int, default=224)
    parser.add_argument("--mask-path", default=None)
    parser.add_argument("--category", default="")
    parser.add_argument("--sample-id", default="")
    args = parser.parse_args()

    run_one(
        image_path=args.image,
        expression=args.expression,
        remoteclip_weight=args.remoteclip_weight,
        output_dir=args.output_dir,
        model_name=args.model_name,
        device=args.device,
        topk=args.topk,
        temperature=args.temperature,
        clip_input_size=args.clip_input_size,
        save_npy=args.save_npy,
        category=args.category,
        sample_id=args.sample_id,
        mask_path=args.mask_path,
    )


if __name__ == "__main__":
    main()
