#!/usr/bin/env python3
"""
Layer/Head-wise [SEG]->image attention mining vs GT (diagnostic only).
Does not modify model code, training, or decoder. Does not read overlay PNGs.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import SiglipImageProcessor

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from detectron2.structures import ImageList
from segearth_r2.datasets.dataset import DataCollatorForCOCODatasetV2, RRSISDDataset
from segearth_r2.model.language_model.llava_phi import SegEarthR2
from segearth_r2.utils import conversation as conversation_lib
from segearth_r2.utils.builder import load_pretrained_model

# ---- GT + metrics (aligned with as_gt_alignment_diagnose.py) ----


def _load_gt_binary_from_array(gt: Any, h: int, w: int) -> np.ndarray:
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    if gt.ndim == 3:
        gt = gt[0]
    return (gt > 0).astype(np.float32)


def _centroid_2d(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(mask > 0.5)
    if ys.size == 0:
        return float("nan"), float("nan")
    return float(ys.mean()), float(xs.mean())


def _map_centroid_to_grid(cy: float, cx: float, h: int, w: int) -> Tuple[int, int]:
    gy = (cy + 0.5) / h * 27.0 - 0.5
    gx = (cx + 0.5) / w * 27.0 - 0.5
    r = int(round(np.clip(gy, 0, 26)))
    c = int(round(np.clip(gx, 0, 26)))
    return r, c


def _downsample_gt_to_27(gt_hw: np.ndarray) -> Tuple[np.ndarray, bool]:
    h, w = gt_hw.shape
    small = cv2.resize(gt_hw, (27, 27), interpolation=cv2.INTER_AREA)
    gt27 = (small >= 0.5).astype(np.float32)
    disappeared = bool(gt27.sum() < 1e-6)
    if disappeared:
        cy, cx = _centroid_2d(gt_hw)
        if math.isnan(cy):
            gt27 = np.zeros((27, 27), dtype=np.float32)
        else:
            r, c = _map_centroid_to_grid(cy, cx, h, w)
            gt27 = np.zeros((27, 27), dtype=np.float32)
            gt27[r, c] = 1.0
    return gt27, disappeared


def _dilate27(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0.5).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    d = cv2.dilate(m, kernel, iterations=1)
    return (d > 0).astype(np.float32)


def _normalize_as(as_flat: np.ndarray) -> np.ndarray:
    x = as_flat.astype(np.float64).reshape(-1)
    x = x - x.min()
    s = x.sum()
    if s <= 0 or not np.isfinite(s):
        raise ValueError("A_S constant after min-subtract; cannot normalize.")
    return (x / s).reshape(27, 27)


def _gt_centroid_grid(gt27: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(gt27 > 0.5)
    if ys.size == 0:
        return 13.0, 13.0
    return float(ys.mean()), float(xs.mean())


def _cell_center(r: int, c: int) -> Tuple[float, float]:
    return r + 0.5, c + 0.5


def _dist_grid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _topk_indices(p: np.ndarray, k: int) -> List[Tuple[int, int]]:
    flat = p.reshape(-1)
    idx = np.argsort(-flat)[:k]
    return [divmod(int(ii), 27) for ii in idx]


def _border_cell(r: int, c: int) -> bool:
    return r == 0 or r == 26 or c == 0 or c == 26


def _entropy(p: np.ndarray) -> float:
    p = p.reshape(-1).astype(np.float64)
    p = np.clip(p, 1e-30, 1.0)
    return float(-(p * np.log(p)).sum())


def compute_alignment_metrics(
    p: np.ndarray,
    gt27: np.ndarray,
    gt_dil: np.ndarray,
    gt_centroid_rc: Tuple[float, float],
) -> Dict[str, Any]:
    flat = p.reshape(-1)
    r1, c1 = divmod(int(np.argmax(flat)), 27)
    top5 = _topk_indices(p, 5)
    top10 = _topk_indices(p, 10)
    return {
        "top1_hit_gt": bool(gt27[r1, c1] > 0.5),
        "top5_hit_gt": any(gt27[r, c] > 0.5 for r, c in top5),
        "top10_hit_gt": any(gt27[r, c] > 0.5 for r, c in top10),
        "mass_in_gt": float((p * gt27).sum()),
        "mass_in_dilated_gt": float((p * gt_dil).sum()),
        "top1_distance_to_gt_centroid": _dist_grid(_cell_center(r1, c1), gt_centroid_rc),
        "top5_min_distance_to_gt_centroid": min(
            _dist_grid(_cell_center(r, c), gt_centroid_rc) for r, c in top5
        ),
        "border_top5_count": sum(1 for r, c in top5 if _border_cell(r, c)),
        "entropy": _entropy(p),
    }


def _iou_pred_gt(pred_u8: np.ndarray, gt_u8: np.ndarray) -> float:
    p = pred_u8.astype(bool)
    g = gt_u8.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / union)


def _as_grid_or_raise(n_img: int) -> int:
    root = int(n_img**0.5)
    if root * root != n_img:
        raise RuntimeError(f"N_img={n_img} is not a perfect square.")
    return root


def _slice_as_map(
    amap_2d: torch.Tensor,
    seg_mask: torch.Tensor,
    img_mask: torch.Tensor,
    batch_idx: int,
    input_ids_shape: Tuple[int, ...],
) -> torch.Tensor:
    """amap_2d [S,S] -> [num_seg, N_img]"""
    if amap_2d.dim() != 2 or amap_2d.shape[0] != amap_2d.shape[1]:
        raise RuntimeError(f"Bad attention 2d shape {tuple(amap_2d.shape)}")
    S = amap_2d.shape[0]
    sm = seg_mask.bool()
    im = img_mask.bool()
    if sm.numel() != S or im.numel() != S:
        raise RuntimeError(
            f"Mask length != S={S}. seg {tuple(sm.shape)} img {tuple(im.shape)} "
            f"input_ids {input_ids_shape} batch {batch_idx}"
        )
    sub = amap_2d[sm][:, im]
    return sub


@torch.no_grad()
def forward_with_attentions(
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
) -> Tuple[Any, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """Returns outputs (with .attentions), seg_idx_mm, image_idx_mm, processed."""
    output_attentions = True
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
        output_hidden_states=False,
        return_dict=return_dict,
    )
    if outputs.attentions is None:
        raise RuntimeError("outputs.attentions is None (FlashAttention / sdp kernel?)")

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
    return outputs, seg_idx_mm, image_idx_mm, processed


def _mean_layer_as_tensor(
    outputs_attentions: Tuple[torch.Tensor, ...],
    seg_idx_mm: torch.Tensor,
    image_idx_mm: torch.Tensor,
    batch_idx: int,
    input_ids_shape: Tuple[int, ...],
) -> torch.Tensor:
    """Head-sum per layer, then mean over layers -> [num_seg, N_img] on CPU float32."""
    seg_mask = seg_idx_mm[batch_idx].detach().cpu()
    img_mask = image_idx_mm[batch_idx].detach().cpu()
    per_layer: List[torch.Tensor] = []
    for li, layer in enumerate(outputs_attentions):
        hs = layer.float().sum(dim=1)[batch_idx].cpu()
        sub = _slice_as_map(hs, seg_mask, img_mask, batch_idx, input_ids_shape)
        per_layer.append(sub)
    return torch.stack(per_layer, dim=0).mean(dim=0)


def _read_instruction(dataset: RRSISDDataset, idx: int) -> str:
    ref = dataset.reason_file[idx]
    if len(ref["sentences"]) > 0 and "sent" in ref["sentences"][0]:
        return ref["sentences"][0]["sent"].strip()
    if len(ref["sentences"]) > 0 and "raw" in ref["sentences"][0]:
        return ref["sentences"][0]["raw"].strip()
    ann = dataset.ann_dict[ref["ann_id"]]
    cat_id = ann.get("categories_id", None)
    cat_name = dataset.category_dict.get(cat_id, "target")
    return f"segment the {cat_name} in this remote sensing image"


def _foreground_ratio_from_sample(sample: Dict[str, Any]) -> float:
    ann0 = sample["annotations"][0]
    m = ann0["mask"]
    if isinstance(m, torch.Tensor):
        m = m.numpy()
    if m.ndim == 3:
        m = m[0]
    h, w = int(ann0["height"]), int(ann0["width"])
    return float((m > 0).sum()) / float(h * w + 1e-8)


def main() -> None:
    ap = argparse.ArgumentParser(description="Layer/head-wise A_S vs GT mining.")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--dataset-root", type=str, required=True)
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--metrics-dir", type=str, default="outputs/debug_as_quality/test_10_small")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--sample-keyword", type=str, default=None)
    ap.add_argument("--no-prefer-small-foreground", action="store_true")
    ap.add_argument("--vision-tower", type=str, required=True)
    ap.add_argument(
        "--mask-config",
        type=str,
        default="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml",
    )
    ap.add_argument("--version", type=str, default="llava_phi")
    args = ap.parse_args()

    metrics_dir = os.path.abspath(
        args.metrics_dir if os.path.isabs(args.metrics_dir) else os.path.join(_PROJECT_ROOT, args.metrics_dir)
    )
    os.makedirs(metrics_dir, exist_ok=True)
    out_detail = os.path.join(metrics_dir, "layer_head_alignment_metrics.csv")
    out_summary = os.path.join(metrics_dir, "layer_head_alignment_summary.csv")
    out_compare = os.path.join(metrics_dir, "layer_head_vs_mean_layer_compare.csv")

    @dataclass
    class _ModelArgs:
        vision_tower: str = ""
        seg_task: str = "instance"

    mask_cfg = args.mask_config
    if not os.path.isabs(mask_cfg):
        mask_cfg = os.path.join(_PROJECT_ROOT, mask_cfg)

    tokenizer, model, _, _ = load_pretrained_model(
        os.path.expanduser(args.checkpoint),
        model_args=_ModelArgs(vision_tower=args.vision_tower),
        mask_config=mask_cfg,
        device="cpu",
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device=device, dtype=torch.float32)
    model.eval()

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.version]
    vt = args.vision_tower
    if not os.path.isabs(vt):
        vt = os.path.join(_PROJECT_ROOT, vt)
    clip_image_processor = SiglipImageProcessor.from_pretrained(vt)
    collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer, clip_image_processor=clip_image_processor)

    @dataclass
    class _DataArgs:
        image_aspect_ratio: str = "square"
        image_grid_pinpoints: Optional[str] = None

    dataset = RRSISDDataset(
        base_data_path=args.dataset_root,
        tokenizer=tokenizer,
        data_args=_DataArgs(),
        split=args.split,
    )

    records: List[Tuple[float, int]] = []
    for idx in tqdm(range(len(dataset)), desc="Scan dataset"):
        instr = _read_instruction(dataset, idx)
        if args.sample_keyword and args.sample_keyword not in instr:
            continue
        sample = dataset[idx]
        records.append((_foreground_ratio_from_sample(sample), idx))
    if not records:
        raise RuntimeError("No samples after filter.")
    if not args.no_prefer_small_foreground:
        records.sort(key=lambda x: x[0])
    else:
        records.sort(key=lambda x: x[1])
    chosen = [ix for _, ix in records[: args.num_samples]]

    detail_fieldnames = [
        "sample_id",
        "seg_idx",
        "layer_id",
        "head_id",
        "pred_iou",
        "foreground_ratio",
        "top1_hit_gt",
        "top5_hit_gt",
        "top10_hit_gt",
        "mass_in_gt",
        "mass_in_dilated_gt",
        "top1_distance_to_gt_centroid",
        "top5_min_distance_to_gt_centroid",
        "border_top5_count",
        "entropy",
        "gt_disappeared_after_downsample",
    ]

    detail_rows: List[Dict[str, Any]] = []
    mean_layer_sample_metrics: List[Dict[str, Any]] = []

    with open(out_detail, "w", encoding="utf-8", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=detail_fieldnames)
        writer.writeheader()

        for out_i, ds_idx in enumerate(tqdm(chosen, desc="Inference")):
            sample = dataset[ds_idx]
            batch = collator([sample])
            if batch["input_ids"].shape[0] != 1:
                raise RuntimeError("Batch size must be 1.")
            input_ids_shape = tuple(batch["input_ids"].shape)
            sid = f"sample_{out_i + 1:04d}"

            def _to_dev(t):
                return t.to(device) if torch.is_tensor(t) else t

            inputs = {k: _to_dev(v) if torch.is_tensor(v) else v for k, v in batch.items()}
            inputs["token_refer_id"] = [ids.to(device) for ids in inputs["token_refer_id"]]

            outputs, seg_idx_mm, image_idx_mm, processed = forward_with_attentions(
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
            )

            seg_mask = seg_idx_mm[0].detach().cpu()
            img_mask = image_idx_mm[0].detach().cpu()
            n_img = int(img_mask.bool().sum().item())
            _as_grid_or_raise(n_img)

            num_layers = len(outputs.attentions)
            num_heads = int(outputs.attentions[0].shape[1])

            # per-seg GT + pred_iou + fg
            seg_infos = inputs["seg_info"]
            num_seg = len(seg_infos)
            if len(processed) != num_seg:
                raise RuntimeError("processed vs seg_info length mismatch")

            seg_gt_pack: List[Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[float, float], bool]] = []
            orig_path = seg_infos[0].get("image_path") or sample["annotations"][0]["image_path"]
            bgr = cv2.imread(orig_path)
            if bgr is None:
                raise FileNotFoundError(orig_path)
            oh, ow = bgr.shape[0], bgr.shape[1]

            pred_ious: List[float] = []
            for si, seg_i in enumerate(seg_infos):
                gt_hw = _load_gt_binary_from_array(seg_i["mask"], seg_i["height"], seg_i["width"])
                gt27, dis = _downsample_gt_to_27(gt_hw)
                gt_dil = _dilate27(gt27)
                gc = _gt_centroid_grid(gt27)
                seg_gt_pack.append((gt27, gt_dil, gt_hw, gc, dis))

                pred_u8 = processed[si]["pred"]
                if pred_u8.ndim > 2:
                    pred_u8 = np.squeeze(pred_u8)
                pred_r = cv2.resize(pred_u8.astype(np.float32), (ow, oh), interpolation=cv2.INTER_LINEAR)
                pred_bin = (pred_r > 127).astype(np.uint8) * 255
                gt_u8 = (gt_hw > 0.5).astype(np.uint8) * 255
                pred_ious.append(_iou_pred_gt(pred_bin, gt_u8))

            fg_ratio = _foreground_ratio_from_sample(sample)

            # mean_layer_as for this sample (per seg)
            mean_layer_mat = _mean_layer_as_tensor(
                outputs.attentions, seg_idx_mm, image_idx_mm, 0, input_ids_shape
            )
            for si in range(num_seg):
                vec = mean_layer_mat[si].numpy().astype(np.float64)
                p = _normalize_as(vec)
                gt27, gt_dil, _ghw, gc, dis = seg_gt_pack[si]
                m = compute_alignment_metrics(p, gt27, gt_dil, gc)
                mean_layer_sample_metrics.append(
                    {
                        "sample_id": sid,
                        "seg_idx": si,
                        "pred_iou": pred_ious[si],
                        "foreground_ratio": fg_ratio,
                        **m,
                        "gt_disappeared_after_downsample": dis,
                    }
                )

            # layer / head loop on CPU
            att_cpu = [layer[0].float().cpu() for layer in outputs.attentions]  # [H,S,S] each

            for li in range(num_layers):
                for hi in range(num_heads):
                    amap = att_cpu[li][hi]
                    sub = _slice_as_map(amap, seg_mask, img_mask, 0, input_ids_shape)
                    for si in range(num_seg):
                        vec = sub[si].numpy().astype(np.float64)
                        try:
                            p = _normalize_as(vec)
                        except ValueError as e:
                            raise RuntimeError(
                                f"{sid} layer={li} head={hi} seg={si}: normalize failed: {e}"
                            ) from e
                        gt27, gt_dil, _ghw, gc, dis = seg_gt_pack[si]
                        m = compute_alignment_metrics(p, gt27, gt_dil, gc)
                        row = {
                            "sample_id": sid,
                            "seg_idx": si,
                            "layer_id": li,
                            "head_id": hi,
                            "pred_iou": pred_ious[si],
                            "foreground_ratio": fg_ratio,
                            "top1_hit_gt": m["top1_hit_gt"],
                            "top5_hit_gt": m["top5_hit_gt"],
                            "top10_hit_gt": m["top10_hit_gt"],
                            "mass_in_gt": m["mass_in_gt"],
                            "mass_in_dilated_gt": m["mass_in_dilated_gt"],
                            "top1_distance_to_gt_centroid": m["top1_distance_to_gt_centroid"],
                            "top5_min_distance_to_gt_centroid": m["top5_min_distance_to_gt_centroid"],
                            "border_top5_count": m["border_top5_count"],
                            "entropy": m["entropy"],
                            "gt_disappeared_after_downsample": dis,
                        }
                        detail_rows.append(row)
                        writer.writerow(row)

    # aggregate by (layer, head)
    by_lh: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in detail_rows:
        by_lh[(int(r["layer_id"]), int(r["head_id"]))].append(r)

    summary_fieldnames = [
        "layer_id",
        "head_id",
        "n",
        "top1_hit_gt_rate",
        "top5_hit_gt_rate",
        "top10_hit_gt_rate",
        "mean_mass_in_gt",
        "mean_mass_in_dilated_gt",
        "mean_top1_distance_to_gt_centroid",
        "mean_top5_min_distance_to_gt_centroid",
        "mean_border_top5_count",
        "mean_entropy",
    ]
    summary_rows: List[Dict[str, Any]] = []
    for (li, hi), rs in sorted(by_lh.items()):
        n = len(rs)
        summary_rows.append(
            {
                "layer_id": li,
                "head_id": hi,
                "n": n,
                "top1_hit_gt_rate": sum(1 for x in rs if x["top1_hit_gt"]) / n,
                "top5_hit_gt_rate": sum(1 for x in rs if x["top5_hit_gt"]) / n,
                "top10_hit_gt_rate": sum(1 for x in rs if x["top10_hit_gt"]) / n,
                "mean_mass_in_gt": float(np.mean([x["mass_in_gt"] for x in rs])),
                "mean_mass_in_dilated_gt": float(np.mean([x["mass_in_dilated_gt"] for x in rs])),
                "mean_top1_distance_to_gt_centroid": float(
                    np.mean([x["top1_distance_to_gt_centroid"] for x in rs])
                ),
                "mean_top5_min_distance_to_gt_centroid": float(
                    np.mean([x["top5_min_distance_to_gt_centroid"] for x in rs])
                ),
                "mean_border_top5_count": float(np.mean([x["border_top5_count"] for x in rs])),
                "mean_entropy": float(np.mean([x["entropy"] for x in rs])),
            }
        )

    with open(out_summary, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fieldnames)
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    # rank top-10 heads
    def sort_key(r: Dict[str, Any]):
        return (
            -r["top5_hit_gt_rate"],
            -r["mean_mass_in_dilated_gt"],
            r["mean_border_top5_count"],
            r["mean_top1_distance_to_gt_centroid"],
        )

    ranked = sorted(summary_rows, key=sort_key)
    print("=== Top 10 (layer_id, head_id) by: top5_hit_gt_rate, mean_mass_in_dilated_gt, "
          "-mean_border_top5_count, -mean_top1_distance ===")
    for i, r in enumerate(ranked[:10], 1):
        print(
            f"  {i:2d}  L{r['layer_id']:2d} H{r['head_id']:2d}  "
            f"top5_hit={r['top5_hit_gt_rate']:.4f}  "
            f"mass_dil={r['mean_mass_in_dilated_gt']:.6f}  "
            f"border5={r['mean_border_top5_count']:.3f}  "
            f"d_top1={r['mean_top1_distance_to_gt_centroid']:.4f}"
        )

    # mean_layer_as pooled (same samples/seg as detail)
    ml_keys = [
        "top1_hit_gt",
        "top5_hit_gt",
        "top10_hit_gt",
        "mass_in_gt",
        "mass_in_dilated_gt",
        "border_top5_count",
    ]
    n_ml = len(mean_layer_sample_metrics)
    ml_agg = {
        "strategy": "mean_layer_as (head-sum per layer, mean over layers)",
        "top1_hit_gt_rate": sum(1 for r in mean_layer_sample_metrics if r["top1_hit_gt"]) / n_ml,
        "top5_hit_gt_rate": sum(1 for r in mean_layer_sample_metrics if r["top5_hit_gt"]) / n_ml,
        "top10_hit_gt_rate": sum(1 for r in mean_layer_sample_metrics if r["top10_hit_gt"]) / n_ml,
        "mean_mass_in_gt": float(np.mean([r["mass_in_gt"] for r in mean_layer_sample_metrics])),
        "mean_mass_in_dilated_gt": float(
            np.mean([r["mass_in_dilated_gt"] for r in mean_layer_sample_metrics])
        ),
        "mean_border_top5_count": float(
            np.mean([r["border_top5_count"] for r in mean_layer_sample_metrics])
        ),
    }

    best = ranked[0]
    best_agg = {
        "strategy": f"best_head L{best['layer_id']} H{best['head_id']}",
        "top1_hit_gt_rate": best["top1_hit_gt_rate"],
        "top5_hit_gt_rate": best["top5_hit_gt_rate"],
        "top10_hit_gt_rate": best["top10_hit_gt_rate"],
        "mean_mass_in_gt": best["mean_mass_in_gt"],
        "mean_mass_in_dilated_gt": best["mean_mass_in_dilated_gt"],
        "mean_border_top5_count": best["mean_border_top5_count"],
    }

    cmp_fields = ["strategy", "top1_hit_gt_rate", "top5_hit_gt_rate", "top10_hit_gt_rate",
                  "mean_mass_in_gt", "mean_mass_in_dilated_gt", "mean_border_top5_count"]
    with open(out_compare, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cmp_fields)
        w.writeheader()
        w.writerow({k: ml_agg[k] for k in cmp_fields})
        w.writerow({k: best_agg[k] for k in cmp_fields})

    print()
    print(f"Wrote {out_detail}")
    print(f"Wrote {out_summary}")
    print(f"Wrote {out_compare}")
    print()
    print("=== mean_layer_as vs best (layer,head) summary (numeric) ===")
    for label, d in [("mean_layer_as", ml_agg), ("best_lh", best_agg)]:
        print(
            label,
            f"top1_rate={d['top1_hit_gt_rate']:.4f}",
            f"top5_rate={d['top5_hit_gt_rate']:.4f}",
            f"top10_rate={d['top10_hit_gt_rate']:.4f}",
            f"mean_mass_gt={d['mean_mass_in_gt']:.6f}",
            f"mean_mass_dil={d['mean_mass_in_dilated_gt']:.6f}",
            f"mean_border5={d['mean_border_top5_count']:.4f}",
        )

    print()
    print("=== Risk notes (threshold-free logic; numeric comparison only) ===")
    if (
        best_agg["top5_hit_gt_rate"] > ml_agg["top5_hit_gt_rate"] + 0.05
        or best_agg["mean_mass_in_dilated_gt"] > ml_agg["mean_mass_in_dilated_gt"] + 0.01
    ):
        print(
            "存在 (layer,head) 在 top5_hit_gt_rate 或 mean_mass_in_dilated_gt 上明显优于 mean_layer_as："
            "head-sum / layer-mean 可能淹没部分空间信息；后续可考虑 selective head prior（仅诊断结论）。"
        )
    else:
        print(
            "排名首位的 (layer,head) 在 top5_hit_gt_rate / mean_mass_in_dilated_gt 上未显著优于 mean_layer_as："
            "当前 MLLM attention 作为空间 prior 的收益有限（仅诊断结论）；可暂停 P_bias 方向。"
        )
    if best_agg["mean_border_top5_count"] >= 2.0:
        print(
            f"best (layer,head) 的 mean_border_top5_count={best_agg['mean_border_top5_count']:.4f} 仍较高 (>=2.0)："
            "边界热点可能主要来自 attention 本身偏置，而非仅由聚合引起（仅诊断结论）。"
        )


if __name__ == "__main__":
    main()
