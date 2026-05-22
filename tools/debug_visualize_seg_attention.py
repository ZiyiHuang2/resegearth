#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEBUG-ONLY: visualize [SEG] -> image-token attention A_S on RRSIS-D val.

This script is not used in training or evaluation pipelines. It does not modify
SegEarthR2.eval_seg, model.forward return types, losses, Mask2Former, or Q/K/V.

A_S indexing follows the same slicing as training forward (llava_phi.py):
  attentions = [layer.sum(dim=1) for layer in outputs.attentions]  # sum heads, not mean
  attention = attention_map[SEG_mask][:, image_features_mask]

No seg_info / GT / AttentionLoss participates in A_S extraction; GT is only for
visualization, pred_iou, and foreground_ratio.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import SiglipImageProcessor

# Project root = parent of tools/
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from segearth_r2.datasets.dataset import DataCollatorForCOCODatasetV2, RRSISDDataset
from segearth_r2.model.language_model.llava_phi import SegEarthR2
from segearth_r2.utils import conversation as conversation_lib
from segearth_r2.utils.builder import load_pretrained_model
from detectron2.structures import ImageList


def _as_grid_or_raise(n_img: int, n_seg: int, img_mask_count: int) -> int:
    root = int(n_img**0.5)
    if root * root != n_img:
        raise RuntimeError(
            f"N_img={n_img} is not a perfect square; cannot reshape to sqrt x sqrt grid. "
            f"num_seg_tokens={n_seg}, image_features_mask True count={img_mask_count}."
        )
    return root


def _validate_masks_and_slice(
    full_attention_map_b: torch.Tensor,
    seg_mask_1d: torch.Tensor,
    img_mask_1d: torch.Tensor,
    batch_idx: int,
    input_ids_shape: Tuple[int, ...],
) -> torch.Tensor:
    """Return A_S of shape [num_seg, N_img]. full_attention_map_b: [S, S]."""
    if full_attention_map_b.dim() != 2:
        raise RuntimeError(f"Expected attention map [S,S], got shape {tuple(full_attention_map_b.shape)}")
    S = full_attention_map_b.shape[0]
    if full_attention_map_b.shape[1] != S:
        raise RuntimeError(f"Expected square attention, got {tuple(full_attention_map_b.shape)}")
    if seg_mask_1d.numel() != S or img_mask_1d.numel() != S:
        raise RuntimeError(
            "SEG_mask or image_features_mask length does not match attention sequence length S.\n"
            f"  attention map shape: {tuple(full_attention_map_b.shape)}\n"
            f"  SEG_mask shape: {tuple(seg_mask_1d.shape)}\n"
            f"  image_features_mask shape: {tuple(img_mask_1d.shape)}\n"
            f"  input_ids shape (batched, pre-multimodal in dataloader): {input_ids_shape}\n"
            f"  batch index: {batch_idx}"
        )
    seg_bool = seg_mask_1d.bool()
    img_bool = img_mask_1d.bool()
    sub = full_attention_map_b[seg_bool][:, img_bool]
    return sub


def extract_as_triplet(
    outputs_attentions: Optional[Tuple[torch.Tensor, ...]],
    seg_token_embedding_indices: torch.Tensor,
    image_features_indices: torch.Tensor,
    batch_idx: int,
    input_ids_shape: Tuple[int, ...],
) -> Dict[str, Any]:
    """
    Head aggregation: sum(dim=1) on each layer (same as llava_phi SegEarthR2.forward).
    Returns tensors on CPU float32, shape [num_seg, N_img].
    """
    if outputs_attentions is None:
        raise RuntimeError(
            "outputs.attentions is None. Cannot build A_S. This is often caused by "
            "Flash Attention or an attention implementation that skips materializing "
            "full attention weights. Try eager attention / disable flash-attn for debugging."
        )
    # Same as forward line 659: sum over heads (not mean)
    head_summed = [layer.float().sum(dim=1) for layer in outputs_attentions]
    seg_mask = seg_token_embedding_indices[batch_idx].detach().cpu()
    img_mask = image_features_indices[batch_idx].detach().cpu()
    S = head_summed[0].shape[-1]
    per_layer: List[torch.Tensor] = []
    for li, full_b in enumerate(head_summed):
        amap = full_b[batch_idx].detach().float().cpu()
        try:
            sub = _validate_masks_and_slice(amap, seg_mask, img_mask, batch_idx, input_ids_shape)
        except RuntimeError as e:
            raise RuntimeError(f"Layer index {li}: {e}") from e
        per_layer.append(sub)
    n_seg, n_img = per_layer[0].shape
    n_img_mask = int(img_mask.bool().sum().item())
    if n_img != n_img_mask:
        raise RuntimeError(f"Internal mismatch: sliced width {n_img} vs mask count {n_img_mask}")
    _as_grid_or_raise(n_img, n_seg, n_img_mask)
    last_layer = per_layer[-1]
    mean_layer = torch.stack(per_layer, dim=0).mean(dim=0)
    loss_style = torch.stack(per_layer, dim=0).sum(dim=0)
    return {
        "last_layer_A_S": last_layer,
        "mean_layer_A_S": mean_layer,
        "loss_style_A_S": loss_style,
        "num_seg_tokens": n_seg,
        "num_image_tokens": n_img,
        "grid_side": int(n_img**0.5),
        "num_layers": len(per_layer),
    }


def _stats(t: torch.Tensor) -> Dict[str, float]:
    t = t.float().reshape(-1)
    return {
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
    }


def _iou_pred_gt(pred_u8: np.ndarray, gt_u8: np.ndarray) -> float:
    p = pred_u8.astype(bool)
    g = gt_u8.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / union)


def _read_instruction_and_category(dataset: RRSISDDataset, idx: int) -> Tuple[str, str]:
    ref = dataset.reason_file[idx]
    if len(ref["sentences"]) > 0 and "sent" in ref["sentences"][0]:
        instruction = ref["sentences"][0]["sent"].strip()
    elif len(ref["sentences"]) > 0 and "raw" in ref["sentences"][0]:
        instruction = ref["sentences"][0]["raw"].strip()
    else:
        ann = dataset.ann_dict[ref["ann_id"]]
        cat_id = ann.get("categories_id", None)
        cat_name = dataset.category_dict.get(cat_id, "target")
        instruction = f"segment the {cat_name} in this remote sensing image"
    ann = dataset.ann_dict[ref["ann_id"]]
    cat_id = ann.get("categories_id", None)
    category = dataset.category_dict.get(cat_id, "unknown")
    return instruction, category


def _foreground_ratio_from_sample(sample: Dict[str, Any]) -> float:
    """Single-instance RRSIS-D: first annotation mask, original resolution."""
    ann0 = sample["annotations"][0]
    m = ann0["mask"]
    if isinstance(m, torch.Tensor):
        m = m.numpy()
    if m.ndim == 3:
        m = m[0]
    h, w = int(ann0["height"]), int(ann0["width"])
    area = float((m > 0).sum())
    return area / float(h * w + 1e-8)


@torch.no_grad()
def debug_eval_seg_with_attentions(
    model: SegEarthR2,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    images: torch.Tensor,
    images_clip: torch.Tensor,
    seg_info: List[Dict[str, Any]],
    token_refer_id: List[torch.Tensor],
    SEG_token_embedding_indices: torch.Tensor,
    mask_num: List[int],
    input_ids_shape_pre_mm: Tuple[int, ...],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Mirrors SegEarthR2.eval_seg but forces output_attentions=True and returns A_S metadata.
    Does not change eval_seg on the class.
    """
    device = next(model.parameters()).device
    output_attentions = True
    output_hidden_states = False
    # Force dict output so .attentions is always available (independent of config).
    return_dict = True

    image_features = model.get_vision_tower_feature(images.float())

    (
        input_ids_mm,
        attention_mask_mm,
        past_key_values,
        inputs_embeds,
        labels_mm,
        seg_idx_mm,
        image_idx_mm,
    ) = model.prepare_inputs_labels_for_multimodal(
        input_ids,
        attention_mask,
        None,
        labels,
        images_clip.float(),
        token_refer_id=token_refer_id,
        SEG_token_embedding_indices=SEG_token_embedding_indices,
    )

    outputs = model.model(
        input_ids=input_ids_mm,
        attention_mask=attention_mask_mm,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=None,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs.last_hidden_state
    SEG_embedding = model.SEG_token_projector(model.get_SEG_embedding(hidden_states, seg_idx_mm))

    mask_features, _t, multi_scale_features = model.pixel_decoder.forward_features(image_features)

    images_list = [image.repeat((num, 1, 1, 1)) for image, num in zip(images, mask_num)]
    images_list = [s[0] for image_repeat in images_list for s in torch.split(image_repeat, 1, dim=0)]
    mask_num_t = torch.tensor(mask_num, device=mask_features.device)
    mask_features = torch.repeat_interleave(mask_features, repeats=mask_num_t, dim=0)
    multi_scale_features = [
        torch.repeat_interleave(feat, repeats=mask_num_t, dim=0) for feat in multi_scale_features
    ]

    mask_outputs = model.predictor(multi_scale_features, mask_features, None, None, SEG_embedding)
    mask_pred_results = mask_outputs["pred_masks"]
    images_il = ImageList.from_tensors(images_list, model.size_divisibility)
    mask_pred_results = F.interpolate(
        mask_pred_results,
        size=(images_il.tensor.shape[-2], images_il.tensor.shape[-1]),
        mode="bilinear",
        align_corners=False,
    )

    processed: List[Dict[str, Any]] = []
    for _seg_info, mask_pred_result in zip(seg_info, mask_pred_results):
        processed.append(
            {
                "pred": ((mask_pred_result.detach().float().cpu().numpy() > 0) * 255).astype(np.uint8),
                "image_name": _seg_info["image_id"],
                "id": _seg_info["data_id"],
                "mask_id": _seg_info["mask_id"],
            }
        )

    # Batch size must be 1 for this debug tool (mask / meta alignment).
    as_pack = extract_as_triplet(
        outputs.attentions,
        seg_idx_mm,
        image_idx_mm,
        batch_idx=0,
        input_ids_shape=input_ids_shape_pre_mm,
    )
    return processed, as_pack


def _upsample_map(map_2d: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    h, w = out_hw
    return cv2.resize(map_2d, (w, h), interpolation=cv2.INTER_LINEAR)


def _heatmap_u8(map_2d: np.ndarray) -> np.ndarray:
    m = map_2d.astype(np.float32)
    m = m - m.min()
    denom = float(m.max() + 1e-8)
    m = m / denom
    u8 = (m * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_JET)[..., ::-1]  # RGB


def _overlay_rgb(image_rgb: np.ndarray, heat_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    im = image_rgb.astype(np.float32)
    hm = heat_rgb.astype(np.float32)
    out = (1 - alpha) * im + alpha * hm
    return np.clip(out, 0, 255).astype(np.uint8)


def _maybe_limit_vis(img: np.ndarray, max_vis: Optional[int]) -> np.ndarray:
    if max_vis is None:
        return img
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_vis:
        return img
    scale = max_vis / float(m)
    nh, nw = int(h * scale + 0.5), int(w * scale + 0.5)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


@dataclass
class CliArgs:
    checkpoint: str
    dataset_root: str
    split: str
    output_dir: str
    num_samples: int
    sample_keyword: Optional[str]
    device: str
    save_overlay: bool
    save_raw_npy: bool
    max_vis_size: Optional[int]
    prefer_small_foreground: bool
    vision_tower: str
    mask_config: str
    version: str


def parse_args() -> CliArgs:
    p = argparse.ArgumentParser(description="Debug A_S visualization (RRSIS-D val).")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--output-dir", type=str, default="outputs/debug_as_quality/rrsisd_val_30/")
    p.add_argument("--num-samples", type=int, default=30)
    p.add_argument("--sample-keyword", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save-overlay", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-raw-npy", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-vis-size", type=int, default=None)
    p.add_argument("--prefer-small-foreground", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--vision-tower",
        type=str,
        default="pretrained_model/CLIP/siglip-so400m-patch14-384",
    )
    p.add_argument(
        "--mask-config",
        type=str,
        default="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml",
    )
    p.add_argument("--version", type=str, default="llava_phi")
    a = p.parse_args()
    return CliArgs(
        checkpoint=a.checkpoint,
        dataset_root=a.dataset_root,
        split=a.split,
        output_dir=a.output_dir,
        num_samples=a.num_samples,
        sample_keyword=a.sample_keyword,
        device=a.device,
        save_overlay=a.save_overlay,
        save_raw_npy=a.save_raw_npy,
        max_vis_size=a.max_vis_size,
        prefer_small_foreground=a.prefer_small_foreground,
        vision_tower=a.vision_tower,
        mask_config=a.mask_config,
        version=a.version,
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "summary.jsonl")

    @dataclass
    class _ModelArgs:
        vision_tower: str = ""
        seg_task: str = "instance"

    model_args = _ModelArgs(vision_tower=args.vision_tower)
    mask_cfg_path = args.mask_config
    if not os.path.isabs(mask_cfg_path):
        mask_cfg_path = os.path.join(_PROJECT_ROOT, mask_cfg_path)

    tokenizer, model, _image_processor, _ctx = load_pretrained_model(
        os.path.expanduser(args.checkpoint),
        model_args=model_args,
        mask_config=mask_cfg_path,
        device="cpu",
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device=device, dtype=torch.float32)
    model.eval()

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.version]
    clip_image_processor = SiglipImageProcessor.from_pretrained(
        os.path.join(_PROJECT_ROOT, args.vision_tower)
        if not os.path.isabs(args.vision_tower)
        else args.vision_tower
    )
    collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer, clip_image_processor=clip_image_processor)

    @dataclass
    class _DataArgs:
        image_aspect_ratio: str = "square"
        image_grid_pinpoints: Optional[str] = None

    data_args = _DataArgs()
    dataset = RRSISDDataset(
        base_data_path=args.dataset_root,
        tokenizer=tokenizer,
        data_args=data_args,
        split=args.split,
    )

    records: List[Tuple[float, int]] = []
    for idx in tqdm(range(len(dataset)), desc="Scanning dataset", disable=False):
        instruction, _cat = _read_instruction_and_category(dataset, idx)
        if args.sample_keyword and args.sample_keyword not in instruction:
            continue
        sample = dataset[idx]
        ratio = _foreground_ratio_from_sample(sample)
        records.append((ratio, idx))

    if not records:
        raise RuntimeError("No samples after filtering. Check dataset-root, split, or --sample-keyword.")

    ratio_by_idx = {ix: r for r, ix in records}
    if args.prefer_small_foreground:
        records.sort(key=lambda x: x[0])
    else:
        records.sort(key=lambda x: x[1])

    chosen = [ix for _, ix in records[: args.num_samples]]
    if len(chosen) < args.num_samples:
        print(f"[warn] Only {len(chosen)} samples match filters; num-samples={args.num_samples}")

    # Clear summary if re-run
    if os.path.isfile(summary_path):
        os.remove(summary_path)

    for out_i, ds_idx in enumerate(tqdm(chosen, desc="Inference")):
        sample = dataset[ds_idx]
        instruction, category = _read_instruction_and_category(dataset, ds_idx)
        batch = collator([sample])
        if batch["input_ids"].shape[0] != 1:
            raise RuntimeError("This debug script only supports batch size 1 (collator returned B>1).")
        input_ids_shape = tuple(batch["input_ids"].shape)

        def _to_dev(t):
            return t.to(device) if torch.is_tensor(t) else t

        inputs = {k: _to_dev(v) if torch.is_tensor(v) else v for k, v in batch.items()}
        inputs["token_refer_id"] = [ids.to(device) for ids in inputs["token_refer_id"]]

        processed, as_pack = debug_eval_seg_with_attentions(
            model,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            images=inputs["images"].float(),
            images_clip=inputs["images_clip"].float(),
            seg_info=inputs["seg_info"],
            token_refer_id=inputs["token_refer_id"],
            SEG_token_embedding_indices=inputs["SEG_token_embedding_indices"],
            mask_num=inputs["mask_num"],
            input_ids_shape_pre_mm=input_ids_shape,
        )

        # Original-resolution RGB (shared by all SEG tokens in this batch item)
        seg0 = inputs["seg_info"][0]
        orig_path = seg0.get("image_path") or sample["annotations"][0]["image_path"]
        bgr = cv2.imread(orig_path)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {orig_path}")
        orig_hw = (bgr.shape[0], bgr.shape[1])
        image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        num_seg = as_pack["num_seg_tokens"]
        variants = {
            "last_layer_as": as_pack["last_layer_A_S"],
            "mean_layer_as": as_pack["mean_layer_A_S"],
            "loss_style_as": as_pack["loss_style_A_S"],
        }

        if len(processed) != len(inputs["seg_info"]) or len(processed) != num_seg:
            raise RuntimeError(
                f"seg_info / processed / num_seg mismatch: len(processed)={len(processed)}, "
                f"len(seg_info)={len(inputs['seg_info'])}, num_seg_tokens={num_seg}"
            )

        sample_tag = f"sample_{out_i + 1:04d}"
        out_dir = args.output_dir
        fg_ratio = _foreground_ratio_from_sample(sample)
        fg_ratio_sort = ratio_by_idx.get(ds_idx, fg_ratio)

        def _name_prefix(seg_idx: int, n_seg: int) -> str:
            if n_seg > 1:
                return f"{sample_tag}_seg{seg_idx}"
            return sample_tag

        def _save_as_variant(
            prefix: str, file_tag: str, tensor_cpu: torch.Tensor, seg_idx: int
        ) -> Dict[str, Any]:
            row = tensor_cpu[seg_idx].numpy().astype(np.float32)
            side = int(row.shape[-1] ** 0.5)
            grid = row.reshape(side, side)
            up = _upsample_map(grid, orig_hw)
            hm_rgb = _heatmap_u8(up)
            vis_img = _maybe_limit_vis(image_rgb, args.max_vis_size)
            vis_hm = _maybe_limit_vis(hm_rgb, args.max_vis_size)
            heat_path = os.path.join(out_dir, f"{prefix}_{file_tag}_heatmap.png")
            cv2.imwrite(heat_path, cv2.cvtColor(vis_hm, cv2.COLOR_RGB2BGR))
            if args.save_overlay:
                ov = _overlay_rgb(vis_img, vis_hm)
                cv2.imwrite(
                    os.path.join(out_dir, f"{prefix}_{file_tag}_overlay.png"),
                    cv2.cvtColor(ov, cv2.COLOR_RGB2BGR),
                )
            if args.save_raw_npy:
                np.save(os.path.join(out_dir, f"{prefix}_{file_tag}.npy"), row)
            st = _stats(tensor_cpu[seg_idx])
            return {"grid": grid, "up": up, "stats": st, "row_vec": row}

        cv2.imwrite(
            os.path.join(out_dir, f"{sample_tag}_image.png"),
            cv2.cvtColor(_maybe_limit_vis(image_rgb, args.max_vis_size), cv2.COLOR_RGB2BGR),
        )

        summary_lines: List[Dict[str, Any]] = []

        for seg_idx in range(num_seg):
            seg_i = inputs["seg_info"][seg_idx]
            gt = seg_i["mask"]
            if isinstance(gt, torch.Tensor):
                gt = gt.cpu().numpy()
            if gt.ndim == 3:
                gt = gt[0]
            gt_u8 = (gt > 0).astype(np.uint8) * 255

            pred_u8 = processed[seg_idx]["pred"]
            if pred_u8.ndim > 2:
                pred_u8 = np.squeeze(pred_u8)
            pred_resized = cv2.resize(
                pred_u8.astype(np.float32), (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_LINEAR
            )
            pred_bin = (pred_resized > 127).astype(np.uint8) * 255
            pred_iou = _iou_pred_gt(pred_bin, gt_u8)

            prefix = _name_prefix(seg_idx, num_seg)
            if num_seg == 1:
                gt_name = f"{sample_tag}_gt_mask.png"
                pr_name = f"{sample_tag}_pred_mask.png"
            else:
                gt_name = f"{prefix}_gt_mask.png"
                pr_name = f"{prefix}_pred_mask.png"
            cv2.imwrite(
                os.path.join(out_dir, gt_name),
                cv2.cvtColor(
                    _maybe_limit_vis(cv2.cvtColor(gt_u8, cv2.COLOR_GRAY2RGB), args.max_vis_size),
                    cv2.COLOR_RGB2BGR,
                ),
            )
            cv2.imwrite(
                os.path.join(out_dir, pr_name),
                cv2.cvtColor(
                    _maybe_limit_vis(cv2.cvtColor(pred_bin, cv2.COLOR_GRAY2RGB), args.max_vis_size),
                    cv2.COLOR_RGB2BGR,
                ),
            )

            last_d = _save_as_variant(prefix, "last_layer_as", variants["last_layer_as"], seg_idx)
            mean_d = _save_as_variant(prefix, "mean_layer_as", variants["mean_layer_as"], seg_idx)
            loss_d = _save_as_variant(prefix, "loss_style_as", variants["loss_style_as"], seg_idx)

            meta = {
                "debug_script": "tools/debug_visualize_seg_attention.py",
                "dataset_index": ds_idx,
                "output_index": out_i,
                "head_aggregation": "sum_over_heads_dim1_not_mean",
                "mean_layer_aggregation": "mean_over_transformer_layers_of_per_layer_A_S_slices",
                "loss_style_aggregation": "sum_over_transformer_layers_of_per_layer_A_S_slices "
                "(mirrors per-layer accumulation order in training forward without loss)",
                "num_seg_tokens": num_seg,
                "num_image_tokens": as_pack["num_image_tokens"],
                "as_grid_size": [as_pack["grid_side"], as_pack["grid_side"]],
                "num_layers": as_pack["num_layers"],
                "seg_idx": seg_idx,
                "image_path": orig_path,
                "expression": instruction,
                "category": category,
                "foreground_ratio": fg_ratio,
                "foreground_ratio_used_for_sorting": fg_ratio_sort,
                "prefer_small_foreground": args.prefer_small_foreground,
                "pred_iou": pred_iou,
                "mask_id": int(seg_i.get("mask_id", seg_idx)),
                "last_layer_as_stats": last_d["stats"],
                "mean_layer_as_stats": mean_d["stats"],
                "loss_style_as_stats": loss_d["stats"],
            }
            meta_name = f"{prefix}_meta.json" if num_seg > 1 else f"{sample_tag}_meta.json"
            with open(os.path.join(out_dir, meta_name), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            sample_id = prefix
            line = {
                "sample_id": sample_id,
                "image_name": str(processed[0]["image_name"]),
                "expression": instruction,
                "category": category,
                "num_seg_tokens": num_seg,
                "num_image_tokens": as_pack["num_image_tokens"],
                "as_grid_size": [as_pack["grid_side"], as_pack["grid_side"]],
                "last_layer_as_min": last_d["stats"]["min"],
                "last_layer_as_max": last_d["stats"]["max"],
                "last_layer_as_mean": last_d["stats"]["mean"],
                "last_layer_as_std": last_d["stats"]["std"],
                "mean_layer_as_min": mean_d["stats"]["min"],
                "mean_layer_as_max": mean_d["stats"]["max"],
                "mean_layer_as_mean": mean_d["stats"]["mean"],
                "mean_layer_as_std": mean_d["stats"]["std"],
                "loss_style_as_min": loss_d["stats"]["min"],
                "loss_style_as_max": loss_d["stats"]["max"],
                "loss_style_as_mean": loss_d["stats"]["mean"],
                "loss_style_as_std": loss_d["stats"]["std"],
                "pred_iou": pred_iou,
                "foreground_ratio": fg_ratio,
            }
            if num_seg > 1:
                line["seg_idx"] = seg_idx
            summary_lines.append(line)

        with open(summary_path, "a", encoding="utf-8") as sf:
            for line in summary_lines:
                sf.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"Done. Outputs under: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
