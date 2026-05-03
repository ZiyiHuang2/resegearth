import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import SiglipImageProcessor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from segearth_r2.datasets.dataset import (  # noqa: E402
    DataCollatorForCOCODatasetV2,
    LaSeRSDataset,
    RRSISDDataset,
)
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


CSV_FIELDS = [
    "sample_id",
    "expression",
    "alpha",
    "alpha_abs",
    "alpha_sign",
    "gate_inside_mean",
    "gate_outside_mean",
    "inside_outside_ratio",
    "effective_inside_mean",
    "effective_outside_mean",
    "effective_inside_outside_gap",
    "gate_gt_iou_top10",
    "effective_gate_gt_iou_top10",
    "gate_peak_inside",
    "effective_peak_inside",
]


def parse_args():
    parser = argparse.ArgumentParser("Diagnose Mid-Stage Gate")
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
    dataset_name = data_args.dataset_name.lower()
    if dataset_name == "rrsisd":
        split = data_args.split.lower()
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Unsupported RRSISD split: {split}")
        dataset = RRSISDDataset(
            base_data_path=data_args.base_data_path,
            tokenizer=tokenizer,
            data_args=data_args,
            split=split,
        )
        split_name = f"{split}.json"
        return split_name, dataset
    if dataset_name == "lasers":
        dataset = LaSeRSDataset(
            base_data_path=data_args.base_data_path,
            tokenizer=tokenizer,
            data_args=data_args,
            split=data_args.split,
        )
        return data_args.split, dataset
    raise ValueError(f"Unsupported dataset_name: {data_args.dataset_name}")


def tensor_to_image_for_vis(image_tensor: torch.Tensor) -> np.ndarray:
    img = image_tensor.detach().float().cpu()
    if img.dim() == 4:
        img = img[0]
    if img.dim() == 3:
        img = img.permute(1, 2, 0)
    img = img.numpy()
    img_min = float(np.min(img))
    img_max = float(np.max(img))
    if img_max - img_min < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - img_min) / (img_max - img_min)).astype(np.float32)


def normalize_01(x: torch.Tensor) -> torch.Tensor:
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + 1e-6)


def to_float(v: Any) -> float:
    if isinstance(v, torch.Tensor):
        return float(v.detach().float().cpu().item())
    return float(v)


def pick_expression(seg_item: Dict[str, Any]) -> str:
    for k in ["expression", "text", "sent", "sentence", "refer_expression", "query"]:
        if k in seg_item and seg_item[k] is not None:
            return str(seg_item[k])
    return ""


def pick_sample_id(seg_item: Dict[str, Any], fallback: str) -> str:
    for k in ["data_id", "id", "mask_id", "image_id", "image_name", "ref_id"]:
        if k in seg_item and seg_item[k] is not None:
            return str(seg_item[k])
    return fallback


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
                    gt = gt_masks.any(dim=0).float()
                elif gt_masks.dim() == 2:
                    gt = gt_masks
                else:
                    return None
                return gt
    return None


def resize_mask_to_gate(gt_mask: torch.Tensor, gate_hw: Tuple[int, int]) -> torch.Tensor:
    gt = gt_mask.unsqueeze(0).unsqueeze(0)
    gt_resized = F.interpolate(gt, size=gate_hw, mode="nearest")
    return (gt_resized[0, 0] > 0.5)


def iou_binary(a: torch.Tensor, b: torch.Tensor) -> float:
    a_bool = a.bool()
    b_bool = b.bool()
    inter = (a_bool & b_bool).sum().item()
    union = (a_bool | b_bool).sum().item()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def build_topk_mask(x: torch.Tensor, ratio: float, largest: bool = True) -> torch.Tensor:
    flat = x.flatten()
    n = flat.numel()
    k = max(1, int(math.ceil(n * ratio)))
    if largest:
        _, idx = torch.topk(flat, k=k, largest=True)
    else:
        _, idx = torch.topk(flat, k=k, largest=False)
    out = torch.zeros_like(flat, dtype=torch.bool)
    out[idx] = True
    return out.view_as(x)


def save_visualization(
    save_path: str,
    image_chw: torch.Tensor,
    gt_mask: Optional[torch.Tensor],
    gate_spatial: torch.Tensor,
    effective_gate: torch.Tensor,
):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualization. Please install it (e.g. pip install matplotlib)."
        ) from exc

    image_np = tensor_to_image_for_vis(image_chw)
    gate_norm = normalize_01(gate_spatial).detach().cpu().numpy()
    eff_norm = normalize_01(effective_gate).detach().cpu().numpy()

    if gt_mask is not None:
        gt_vis = gt_mask.detach().float().cpu().numpy()
    else:
        gt_vis = np.zeros_like(gate_norm, dtype=np.float32)

    gate_up = F.interpolate(
        gate_spatial.unsqueeze(0).unsqueeze(0),
        size=image_np.shape[:2],
        mode="bilinear",
        align_corners=False,
    )[0, 0].detach().cpu().numpy()
    gate_up = (gate_up - gate_up.min()) / (gate_up.max() - gate_up.min() + 1e-6)

    eff_up = F.interpolate(
        effective_gate.unsqueeze(0).unsqueeze(0),
        size=image_np.shape[:2],
        mode="bilinear",
        align_corners=False,
    )[0, 0].detach().cpu().numpy()
    eff_up = (eff_up - eff_up.min()) / (eff_up.max() - eff_up.min() + 1e-6)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    ax = axes.ravel()

    ax[0].imshow(image_np)
    ax[0].set_title("Image")
    ax[0].axis("off")

    ax[1].imshow(gt_vis, cmap="gray")
    ax[1].set_title("GT Mask (gate scale if available)")
    ax[1].axis("off")

    im2 = ax[2].imshow(gate_norm, cmap="viridis")
    ax[2].set_title("Gate Spatial")
    ax[2].axis("off")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

    im3 = ax[3].imshow(eff_norm, cmap="coolwarm")
    ax[3].set_title("Effective Gate (alpha*gate)")
    ax[3].axis("off")
    fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)

    ax[4].imshow(image_np)
    ax[4].imshow(gate_up, cmap="jet", alpha=0.45)
    ax[4].set_title("Gate Overlay")
    ax[4].axis("off")

    ax[5].imshow(image_np)
    ax[5].imshow(eff_up, cmap="jet", alpha=0.45)
    ax[5].set_title("Effective Gate Overlay")
    ax[5].axis("off")

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
    device = torch.device(args.device)
    if device.type == "cpu" and run_dtype in (torch.float16, torch.bfloat16):
        print("[INFO] CPU 不支持低精度诊断，自动切换到 float32。")
        run_dtype = torch.float32

    model_path = os.path.expanduser(args.model_path)
    if not os.path.exists(os.path.join(model_path, "config.json")):
        print("[WARNING] model_path 下未检测到 config.json，当前脚本优先支持 merged model。")
        print("[WARNING] 若为 Trainer checkpoint，请先合并导出后再运行，或自行确保其可被 from_pretrained 加载。")

    data_args = DataArguments(
        vision_tower=args.vision_tower,
        vision_tower_mask=args.vision_tower_mask,
        base_data_path=args.base_data_path,
        model_path=args.model_path,
        mask_config=args.mask_config,
        dataset_name=args.dataset_name,
        split=args.split,
        dataloader_num_workers=args.num_workers,
    )

    tokenizer, model, _, _ = load_pretrained_model(
        model_path=model_path,
        model_args=data_args,
        mask_config=args.mask_config,
        device=args.device,
    )
    model.to(device=device, dtype=run_dtype)
    model.eval()
    conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version]

    clip_image_processor = SiglipImageProcessor.from_pretrained(args.vision_tower)
    collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer, clip_image_processor=clip_image_processor)
    split_name, dataset = build_eval_dataset(data_args, tokenizer)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=collator,
    )

    vision_tower_mask = model.get_model().get_vision_tower_mask()
    if not hasattr(vision_tower_mask, "mid_stage_text_recalibration"):
        raise RuntimeError("未找到 mid_stage_text_recalibration 模块。")
    if not hasattr(vision_tower_mask.mid_stage_text_recalibration, "alpha"):
        raise RuntimeError("未找到 mid_stage_text_recalibration.alpha。")

    alpha_tensor = vision_tower_mask.mid_stage_text_recalibration.alpha.detach().float().cpu().view(-1)[0]
    alpha_value = float(alpha_tensor.item())
    alpha_abs = abs(alpha_value)
    alpha_sign = "pos" if alpha_value > 0 else ("neg" if alpha_value < 0 else "zero")
    print(f"[Alpha] value={alpha_value:.8f}, abs={alpha_abs:.8f}, sign={alpha_sign}")

    rows: List[Dict[str, Any]] = []
    text_cond_norms: List[float] = []
    text_cond_has_nan = False
    text_cond_has_inf = False
    processed = 0

    with torch.no_grad():
        for step, inputs in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Diagnose {split_name}"):
            if processed >= args.num_samples:
                break

            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            if "token_refer_id" in inputs and isinstance(inputs["token_refer_id"], list):
                inputs["token_refer_id"] = [ids.to(device) for ids in inputs["token_refer_id"]]

            images = inputs["images"].to(device=device, dtype=run_dtype)
            text_cond = model.build_text_condition(
                token_refer_id=inputs.get("token_refer_id", None),
                batch_size=images.shape[0],
                device=images.device,
            )
            if text_cond is not None:
                text_cond_norm = text_cond.detach().float().norm(dim=-1)
                text_cond_norms.extend(text_cond_norm.cpu().tolist())
                text_cond_has_nan = text_cond_has_nan or bool(torch.isnan(text_cond).any().item())
                text_cond_has_inf = text_cond_has_inf or bool(torch.isinf(text_cond).any().item())

            if text_cond is not None and text_cond.dtype != images.dtype:
                text_cond = text_cond.to(dtype=images.dtype, device=images.device)

            outputs = model.get_vision_tower_feature(
                images,
                text_cond=text_cond,
                return_midstage_gate=True,
            )
            features_dict, gate_info = outputs
            _ = features_dict  # keep for interface validation

            mid_stage_gate = gate_info.get("mid_stage_gate", None) if isinstance(gate_info, dict) else None
            gate_alpha = gate_info.get("mid_stage_alpha", None) if isinstance(gate_info, dict) else None
            if gate_alpha is not None:
                gate_alpha = gate_alpha.to(device=images.device, dtype=torch.float32).view(-1)[0]
            else:
                gate_alpha = torch.tensor(alpha_value, device=images.device, dtype=torch.float32)

            if mid_stage_gate is None:
                print(f"[WARNING] step={step} mid_stage_gate is None（模块未启用或 text_cond 缺失）。")
                sample_id = pick_sample_id(inputs["seg_info"][0], fallback=f"sample_{processed:04d}")
                expression = pick_expression(inputs["seg_info"][0])
                row = {k: np.nan for k in CSV_FIELDS}
                row["sample_id"] = sample_id
                row["expression"] = expression
                row["alpha"] = alpha_value
                row["alpha_abs"] = alpha_abs
                row["alpha_sign"] = alpha_sign
                rows.append(row)
                processed += 1
                continue

            if mid_stage_gate.dtype != images.dtype:
                mid_stage_gate = mid_stage_gate.to(dtype=images.dtype, device=images.device)

            gate_spatial = mid_stage_gate.mean(dim=1)  # [B,H,W]
            for b in range(gate_spatial.shape[0]):
                if processed >= args.num_samples:
                    break
                seg_item = inputs["seg_info"][b]
                sample_id = pick_sample_id(seg_item, fallback=f"sample_{processed:04d}")
                expression = pick_expression(seg_item)

                gate_map = gate_spatial[b].detach()
                eff_map = gate_map * gate_alpha.to(dtype=gate_map.dtype)

                gt_mask = extract_gt_mask(seg_item, device=gate_map.device)
                row = {
                    "sample_id": sample_id,
                    "expression": expression,
                    "alpha": alpha_value,
                    "alpha_abs": alpha_abs,
                    "alpha_sign": alpha_sign,
                }

                gt_for_vis = None
                if gt_mask is None:
                    print(f"[WARNING] sample={sample_id} 无 GT mask，仅保存 gate 可视化，跳过对齐统计。")
                    for k in CSV_FIELDS:
                        if k not in row:
                            row[k] = np.nan
                else:
                    gt_resized = resize_mask_to_gate(gt_mask, gate_map.shape[-2:])
                    gt_for_vis = gt_resized.float()
                    inside = gt_resized
                    outside = ~gt_resized

                    if inside.sum().item() == 0 or outside.sum().item() == 0:
                        print(f"[WARNING] sample={sample_id} resized GT 前景或背景为空，统计可能不稳定。")

                    gate_inside_mean = gate_map[inside].mean() if inside.any() else torch.tensor(float("nan"), device=gate_map.device)
                    gate_outside_mean = gate_map[outside].mean() if outside.any() else torch.tensor(float("nan"), device=gate_map.device)
                    inside_outside_ratio = gate_inside_mean / (gate_outside_mean + 1e-6)

                    eff_inside_mean = eff_map[inside].mean() if inside.any() else torch.tensor(float("nan"), device=gate_map.device)
                    eff_outside_mean = eff_map[outside].mean() if outside.any() else torch.tensor(float("nan"), device=gate_map.device)
                    eff_gap = eff_inside_mean - eff_outside_mean

                    gate_top10 = build_topk_mask(gate_map, ratio=0.1, largest=True)
                    gate_iou = iou_binary(gate_top10, gt_resized)

                    if alpha_value >= 0:
                        eff_focus = build_topk_mask(eff_map, ratio=0.1, largest=True)
                        eff_peak_idx = torch.argmax(eff_map)
                    else:
                        eff_focus = build_topk_mask(eff_map, ratio=0.1, largest=False)
                        eff_peak_idx = torch.argmin(eff_map)
                    eff_iou = iou_binary(eff_focus, gt_resized)

                    gate_peak_idx = torch.argmax(gate_map)
                    gate_peak_h = int(gate_peak_idx.item() // gate_map.shape[1])
                    gate_peak_w = int(gate_peak_idx.item() % gate_map.shape[1])
                    gate_peak_inside = bool(gt_resized[gate_peak_h, gate_peak_w].item())

                    eff_peak_h = int(eff_peak_idx.item() // eff_map.shape[1])
                    eff_peak_w = int(eff_peak_idx.item() % eff_map.shape[1])
                    eff_peak_inside = bool(gt_resized[eff_peak_h, eff_peak_w].item())

                    row.update(
                        {
                            "gate_inside_mean": to_float(gate_inside_mean),
                            "gate_outside_mean": to_float(gate_outside_mean),
                            "inside_outside_ratio": to_float(inside_outside_ratio),
                            "effective_inside_mean": to_float(eff_inside_mean),
                            "effective_outside_mean": to_float(eff_outside_mean),
                            "effective_inside_outside_gap": to_float(eff_gap),
                            "gate_gt_iou_top10": float(gate_iou),
                            "effective_gate_gt_iou_top10": float(eff_iou),
                            "gate_peak_inside": int(gate_peak_inside),
                            "effective_peak_inside": int(eff_peak_inside),
                        }
                    )

                rows.append(row)
                vis_path = os.path.join(vis_dir, f"sample_{processed:04d}_{sample_id}.png")
                save_visualization(
                    save_path=vis_path,
                    image_chw=images[b],
                    gt_mask=gt_for_vis,
                    gate_spatial=gate_map,
                    effective_gate=eff_map,
                )
                processed += 1

    csv_path = os.path.join(args.output_dir, "gate_diagnosis.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {
        "alpha_value": alpha_value,
        "alpha_abs": alpha_abs,
        "alpha_sign": alpha_sign,
        "num_samples": processed,
        "mean_gate_inside_mean": nanmean(rows, "gate_inside_mean"),
        "mean_gate_outside_mean": nanmean(rows, "gate_outside_mean"),
        "mean_inside_outside_ratio": nanmean(rows, "inside_outside_ratio"),
        "mean_effective_inside_outside_gap": nanmean(rows, "effective_inside_outside_gap"),
        "mean_gate_gt_iou_top10": nanmean(rows, "gate_gt_iou_top10"),
        "mean_effective_gate_gt_iou_top10": nanmean(rows, "effective_gate_gt_iou_top10"),
        "gate_peak_inside_rate": nanmean(rows, "gate_peak_inside"),
        "effective_peak_inside_rate": nanmean(rows, "effective_peak_inside"),
        "text_cond_norm_mean": float(np.mean(text_cond_norms)) if text_cond_norms else float("nan"),
        "text_cond_norm_std": float(np.std(text_cond_norms)) if text_cond_norms else float("nan"),
        "text_cond_has_nan": bool(text_cond_has_nan),
        "text_cond_has_inf": bool(text_cond_has_inf),
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("===== Mid-stage Gate Diagnosis Summary =====")
    print(f"alpha value: {summary['alpha_value']}")
    print(f"alpha abs: {summary['alpha_abs']}")
    print(f"alpha sign: {summary['alpha_sign']}")
    print(f"mean gate_inside_mean: {summary['mean_gate_inside_mean']}")
    print(f"mean gate_outside_mean: {summary['mean_gate_outside_mean']}")
    print(f"mean inside_outside_ratio: {summary['mean_inside_outside_ratio']}")
    print(f"mean effective_inside_outside_gap: {summary['mean_effective_inside_outside_gap']}")
    print(f"mean gate_gt_iou_top10: {summary['mean_gate_gt_iou_top10']}")
    print(f"mean effective_gate_gt_iou_top10: {summary['mean_effective_gate_gt_iou_top10']}")
    print(f"gate_peak_inside rate: {summary['gate_peak_inside_rate']}")
    print(f"effective_peak_inside rate: {summary['effective_peak_inside_rate']}")
    print(f"text_cond norm mean: {summary['text_cond_norm_mean']}")
    print(f"text_cond norm std: {summary['text_cond_norm_std']}")
    print(f"text_cond has NaN: {summary['text_cond_has_nan']}")
    print(f"text_cond has Inf: {summary['text_cond_has_inf']}")
    print(f"CSV saved: {csv_path}")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
