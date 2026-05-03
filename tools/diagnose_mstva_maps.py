import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import SiglipImageProcessor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from segearth_r2.datasets.dataset import DataCollatorForCOCODatasetV2, LaSeRSDataset, RRSISDDataset  # noqa: E402
from segearth_r2.utils import conversation as conversation_lib  # noqa: E402
from segearth_r2.utils.builder import load_pretrained_model  # noqa: E402


@dataclass
class DataArguments:
    vision_tower: str = ""
    vision_tower_mask: str = ""
    base_data_path: str = ""
    model_path: str = ""
    mask_config: str = ""
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    model_map_name: str = "segearth_r2"
    version: str = "llava_phi"
    dataset_name: str = "rrsisd"
    split: str = "test"
    eval_batch_size: int = 1
    dataloader_num_workers: int = 4
    seg_task: str = "instance"
    local_rank: int = 0
    is_multimodal: bool = True


def parse_args():
    parser = argparse.ArgumentParser("Diagnose MSTVA maps")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--vision_tower", type=str, required=True)
    parser.add_argument("--vision_tower_mask", type=str, required=True)
    parser.add_argument("--mask_config", type=str, required=True)
    parser.add_argument("--base_data_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="rrsisd", choices=["rrsisd", "lasers"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def build_eval_dataset(data_args: DataArguments, tokenizer):
    if data_args.dataset_name.lower() == "rrsisd":
        dataset = RRSISDDataset(
            base_data_path=data_args.base_data_path,
            tokenizer=tokenizer,
            data_args=data_args,
            split=data_args.split.lower(),
        )
    else:
        dataset = LaSeRSDataset(
            base_data_path=data_args.base_data_path,
            tokenizer=tokenizer,
            data_args=data_args,
            split=data_args.split,
        )
    return dataset


def tensor_to_image_for_vis(image_tensor: torch.Tensor) -> np.ndarray:
    img = image_tensor.detach().float().cpu()
    if img.dim() == 3:
        img = img.permute(1, 2, 0)
    img = img.numpy()
    img_min = float(np.min(img))
    img_max = float(np.max(img))
    if img_max - img_min < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - img_min) / (img_max - img_min)).astype(np.float32)


def normalize_01(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


def extract_gt_mask(seg_item: Dict[str, Any], device: torch.device) -> Optional[torch.Tensor]:
    if "mask" in seg_item and seg_item["mask"] is not None:
        gt = seg_item["mask"]
        if torch.is_tensor(gt):
            gt = gt.to(device=device, dtype=torch.float32)
            if gt.dim() == 3:
                gt = gt[0]
            return gt
    if "instances" in seg_item and seg_item["instances"] is not None:
        inst = seg_item["instances"]
        if isinstance(inst, list) and len(inst) > 0:
            inst = inst[0]
        if hasattr(inst, "gt_masks"):
            gt_masks = inst.gt_masks
            if hasattr(gt_masks, "tensor"):
                gt_masks = gt_masks.tensor
            if torch.is_tensor(gt_masks) and gt_masks.numel() > 0:
                gt_masks = gt_masks.to(device=device, dtype=torch.float32)
                if gt_masks.dim() == 3:
                    return gt_masks.any(dim=0).float()
                if gt_masks.dim() == 2:
                    return gt_masks
    return None


def build_topk_mask(x: torch.Tensor, ratio: float = 0.1) -> torch.Tensor:
    flat = x.flatten()
    k = max(1, int(math.ceil(flat.numel() * ratio)))
    _, idx = torch.topk(flat, k=k, largest=True)
    out = torch.zeros_like(flat, dtype=torch.bool)
    out[idx] = True
    return out.view_as(x)


def iou_binary(a: torch.Tensor, b: torch.Tensor) -> float:
    inter = (a.bool() & b.bool()).sum().item()
    union = (a.bool() | b.bool()).sum().item()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def pick_sample_id(seg_item: Dict[str, Any], fallback: str) -> str:
    for k in ["data_id", "id", "mask_id", "image_id", "image_name", "ref_id"]:
        if k in seg_item and seg_item[k] is not None:
            return str(seg_item[k])
    return fallback


def save_vis(save_path: str, image: torch.Tensor, gt: torch.Tensor, r3: torch.Tensor, r4: torch.Tensor, r5: torch.Tensor):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for visualization.") from exc

    image_np = tensor_to_image_for_vis(image)
    gt_np = gt.detach().float().cpu().numpy() if gt is not None else np.zeros(image_np.shape[:2], dtype=np.float32)
    r3_up = F.interpolate(r3.unsqueeze(0).unsqueeze(0), size=image_np.shape[:2], mode="bilinear", align_corners=False)[0, 0]
    r4_up = F.interpolate(r4.unsqueeze(0).unsqueeze(0), size=image_np.shape[:2], mode="bilinear", align_corners=False)[0, 0]
    r5_up = F.interpolate(r5.unsqueeze(0).unsqueeze(0), size=image_np.shape[:2], mode="bilinear", align_corners=False)[0, 0]
    r3_up = normalize_01(r3_up).cpu().numpy()
    r4_up = normalize_01(r4_up).cpu().numpy()
    r5_up = normalize_01(r5_up).cpu().numpy()

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    ax = axes.ravel()
    ax[0].imshow(image_np); ax[0].set_title("Image"); ax[0].axis("off")
    ax[1].imshow(gt_np, cmap="gray"); ax[1].set_title("GT"); ax[1].axis("off")
    ax[2].imshow(normalize_01(r3).cpu().numpy(), cmap="viridis"); ax[2].set_title("R3"); ax[2].axis("off")
    ax[3].imshow(normalize_01(r4).cpu().numpy(), cmap="viridis"); ax[3].set_title("R4"); ax[3].axis("off")
    ax[4].imshow(normalize_01(r5).cpu().numpy(), cmap="viridis"); ax[4].set_title("R5"); ax[4].axis("off")
    ax[5].imshow(image_np); ax[5].imshow(r3_up, cmap="jet", alpha=0.45); ax[5].set_title("R3 overlay"); ax[5].axis("off")
    ax[6].imshow(image_np); ax[6].imshow(r4_up, cmap="jet", alpha=0.45); ax[6].set_title("R4 overlay"); ax[6].axis("off")
    ax[7].imshow(image_np); ax[7].imshow(r5_up, cmap="jet", alpha=0.45); ax[7].set_title("R5 overlay"); ax[7].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close(fig)


def nanmean(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key) is not None and not math.isnan(float(r[key]))]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    run_dtype = resolve_dtype(args.dtype)
    if args.device == "cpu" and run_dtype in (torch.float16, torch.bfloat16):
        run_dtype = torch.float32

    data_args = DataArguments(
        vision_tower=args.vision_tower,
        vision_tower_mask=args.vision_tower_mask,
        base_data_path=args.base_data_path,
        model_path=args.model_path,
        mask_config=args.mask_config,
        dataset_name=args.dataset_name,
        split=args.split,
    )
    tokenizer, model, _, _ = load_pretrained_model(
        model_path=args.model_path,
        model_args=data_args,
        mask_config=args.mask_config,
        device=args.device,
    )
    model.to(device=args.device, dtype=run_dtype)
    model.eval()
    conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version]

    collator = DataCollatorForCOCODatasetV2(
        tokenizer=tokenizer,
        clip_image_processor=SiglipImageProcessor.from_pretrained(args.vision_tower),
    )
    dataset = build_eval_dataset(data_args, tokenizer)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    csv_fields = [
        "sample_id",
        "R3_inside_mean", "R3_outside_mean", "R3_inside_outside_ratio", "R3_top10_iou", "R3_peak_inside",
        "R4_inside_mean", "R4_outside_mean", "R4_inside_outside_ratio", "R4_top10_iou", "R4_peak_inside",
        "R5_inside_mean", "R5_outside_mean", "R5_inside_outside_ratio", "R5_top10_iou", "R5_peak_inside",
        "effective_gap_R3", "effective_gap_R4", "effective_gap_R5",
        "R3_alpha", "R4_alpha", "R5_alpha",
        "alpha3", "alpha4", "alpha5",
    ]
    rows: List[Dict[str, Any]] = []
    processed = 0
    text_valid_token_counts: List[int] = []

    with torch.no_grad():
        for step, inputs in tqdm(enumerate(loader), total=len(loader), desc="Diagnose MSTVA"):
            if processed >= args.num_samples:
                break
            inputs = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            if "token_refer_id" in inputs and isinstance(inputs["token_refer_id"], list):
                inputs["token_refer_id"] = [x.to(args.device) for x in inputs["token_refer_id"]]

            images = inputs["images"].to(device=args.device, dtype=run_dtype)
            text_cond = model.build_text_condition(inputs.get("token_refer_id", None), batch_size=images.shape[0], device=images.device)
            text_tokens, text_mask = model.build_text_tokens(inputs.get("token_refer_id", None), batch_size=images.shape[0], device=images.device)
            if text_mask is not None:
                valid_counts = text_mask.sum(dim=-1).detach().cpu().tolist()
                text_valid_token_counts.extend([int(x) for x in valid_counts])
                if min(valid_counts) <= 0:
                    print("[WARNING] text_mask contains sample with zero valid tokens.")
            _, extra = model.get_vision_tower_feature(
                images,
                text_cond=text_cond,
                text_tokens=text_tokens,
                text_mask=text_mask,
                return_mstva_maps=True,
            )
            mstva = extra.get("mstva", None) if isinstance(extra, dict) else None
            if not isinstance(mstva, dict):
                print(f"[WARNING] step={step}: mstva maps not returned, skip.")
                continue

            r3_batch = mstva.get("R3", None)
            r4_batch = mstva.get("R4", None)
            r5_batch = mstva.get("R5", None)
            alpha3 = float(mstva.get("alpha3", torch.zeros(1)).detach().float().view(-1)[0].item())
            alpha4 = float(mstva.get("alpha4", torch.zeros(1)).detach().float().view(-1)[0].item())
            alpha5 = float(mstva.get("alpha5", torch.zeros(1)).detach().float().view(-1)[0].item())
            if r3_batch is None or r4_batch is None or r5_batch is None:
                continue

            for b in range(images.shape[0]):
                if processed >= args.num_samples:
                    break
                seg_item = inputs["seg_info"][b]
                sample_id = pick_sample_id(seg_item, fallback=f"sample_{processed:04d}")
                gt = extract_gt_mask(seg_item, device=images.device)
                if gt is None:
                    print(f"[WARNING] sample={sample_id}: GT unavailable, skip.")
                    continue

                row = {
                    "sample_id": sample_id,
                    "R3_alpha": alpha3,
                    "R4_alpha": alpha4,
                    "R5_alpha": alpha5,
                    "alpha3": alpha3,
                    "alpha4": alpha4,
                    "alpha5": alpha5,
                }
                all_maps = {"R3": r3_batch[b, 0], "R4": r4_batch[b, 0], "R5": r5_batch[b, 0]}
                for scale_name, r_map in all_maps.items():
                    gt_s = F.interpolate(gt.unsqueeze(0).unsqueeze(0), size=r_map.shape[-2:], mode="nearest")[0, 0] > 0.5
                    inside = gt_s
                    outside = ~gt_s
                    if inside.sum().item() == 0 or outside.sum().item() == 0:
                        print(f"[WARNING] sample={sample_id}, {scale_name}: empty inside/outside, skip this scale.")
                        continue
                    inside_mean = r_map[inside].mean()
                    outside_mean = r_map[outside].mean()
                    ratio = inside_mean / (outside_mean + 1e-6)
                    top10 = build_topk_mask(r_map, ratio=0.1)
                    top10_iou = iou_binary(top10, gt_s)
                    peak_idx = torch.argmax(r_map)
                    peak_h = int(peak_idx.item() // r_map.shape[1])
                    peak_w = int(peak_idx.item() % r_map.shape[1])
                    peak_inside = int(gt_s[peak_h, peak_w].item())
                    row[f"{scale_name}_inside_mean"] = float(inside_mean.detach().float().cpu().item())
                    row[f"{scale_name}_outside_mean"] = float(outside_mean.detach().float().cpu().item())
                    row[f"{scale_name}_inside_outside_ratio"] = float(ratio.detach().float().cpu().item())
                    row[f"{scale_name}_top10_iou"] = float(top10_iou)
                    row[f"{scale_name}_peak_inside"] = peak_inside
                    eff_gap = {"R3": alpha3, "R4": alpha4, "R5": alpha5}[scale_name] * (
                        row[f"{scale_name}_inside_mean"] - row[f"{scale_name}_outside_mean"]
                    )
                    row[f"effective_gap_{scale_name}"] = float(eff_gap)

                rows.append(row)
                save_vis(
                    os.path.join(vis_dir, f"sample_{processed:04d}_{sample_id}.png"),
                    images[b],
                    gt,
                    r3_batch[b, 0],
                    r4_batch[b, 0],
                    r5_batch[b, 0],
                )
                processed += 1

    csv_path = os.path.join(args.output_dir, "mstva_diagnosis.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    summary = {
        "R3_inside_outside_ratio": nanmean(rows, "R3_inside_outside_ratio"),
        "R4_inside_outside_ratio": nanmean(rows, "R4_inside_outside_ratio"),
        "R5_inside_outside_ratio": nanmean(rows, "R5_inside_outside_ratio"),
        "R3_top10_iou": nanmean(rows, "R3_top10_iou"),
        "R4_top10_iou": nanmean(rows, "R4_top10_iou"),
        "R5_top10_iou": nanmean(rows, "R5_top10_iou"),
        "R3_peak_inside_rate": nanmean(rows, "R3_peak_inside"),
        "R4_peak_inside_rate": nanmean(rows, "R4_peak_inside"),
        "R5_peak_inside_rate": nanmean(rows, "R5_peak_inside"),
        "effective_gap_R3": nanmean(rows, "effective_gap_R3"),
        "effective_gap_R4": nanmean(rows, "effective_gap_R4"),
        "effective_gap_R5": nanmean(rows, "effective_gap_R5"),
        "R3_alpha": nanmean(rows, "R3_alpha"),
        "R4_alpha": nanmean(rows, "R4_alpha"),
        "R5_alpha": nanmean(rows, "R5_alpha"),
        "alpha3": nanmean(rows, "alpha3"),
        "alpha4": nanmean(rows, "alpha4"),
        "alpha5": nanmean(rows, "alpha5"),
        "text_valid_tokens_mean": float(np.mean(text_valid_token_counts)) if text_valid_token_counts else float("nan"),
        "text_valid_tokens_min": int(np.min(text_valid_token_counts)) if text_valid_token_counts else -1,
        "text_valid_tokens_max": int(np.max(text_valid_token_counts)) if text_valid_token_counts else -1,
        "num_samples": processed,
    }
    if text_valid_token_counts and min(text_valid_token_counts) <= 0:
        print("[WARNING] text_valid_tokens_min <= 0, check text mask construction.")
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"CSV saved: {csv_path}")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
