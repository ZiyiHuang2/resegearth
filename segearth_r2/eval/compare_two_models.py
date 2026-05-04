"""
Compare segmentation outputs of two checkpoints (model A vs model B) on the same split.

用法示例:
  python -m segearth_r2.eval.compare_two_models \\
    --model-a-path /path/to/baseline_merged \\
    --model-b-path /path/to/structured_merged \\
    --base-data-path /data/RRSISD \\
    --dataset-name rrsisd \\
    --split test \\
    --output-dir ./compare_out/baseline_vs_struct \\
    --mask-config segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml \\
    --vision-tower pretrained_model/CLIP/siglip-so400m-patch14-384

默认单进程、batch_size=1；可将 --model-b-device cuda:1 以减轻显存压力。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import SiglipImageProcessor

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from segearth_r2.datasets.dataset import DataCollatorForCOCODatasetV2
from segearth_r2.eval.eval import build_eval_datasets
from segearth_r2.utils import conversation as conversation_lib
from segearth_r2.utils.builder import load_pretrained_model


def _binary_iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = int(np.logical_and(p, g).sum())
    union = int(np.logical_or(p, g).sum())
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / (union + eps))


def _iou_pair(pred_a: np.ndarray, pred_b: np.ndarray, eps: float = 1e-7) -> float:
    return _binary_iou(pred_a, pred_b, eps=eps)


def _resolve_subset_key(seg: Dict[str, Any]) -> str:
    for k in ("subset", "subset_name", "eval_subset"):
        v = seg.get(k)
        if v is None:
            continue
        t = str(v).strip().upper()
        if t.startswith("B"):
            return "B"
        if t.startswith("R"):
            return "R"
        return str(v).strip() or "UNK"
    return "UNK"


def _extract_gt_hw(seg: Dict[str, Any]) -> Optional[np.ndarray]:
    if "mask" in seg and seg["mask"] is not None:
        m = seg["mask"]
        if torch.is_tensor(m):
            m = m.detach().cpu().numpy()
        if m.ndim == 3:
            m = m[0]
        return (m > 0).astype(np.uint8)
    inst = seg.get("instances", None)
    if inst is not None and hasattr(inst, "gt_masks"):
        try:
            t = inst.gt_masks.tensor
            if torch.is_tensor(t):
                t = t.detach().cpu().numpy()
            if t.ndim == 3 and t.shape[0] > 0:
                return (t[0] > 0).astype(np.uint8)
        except Exception:
            return None
    return None


def _pick_best_query_logits(
    pred_logits_qhw: np.ndarray, gt_hw: np.ndarray, logit_thresh: float
) -> Tuple[np.ndarray, float]:
    """pred_logits_qhw: [Q,H,W], gt_hw: [H,W] uint8/bool."""
    gh, gw = int(gt_hw.shape[0]), int(gt_hw.shape[1])
    qh, qw = int(pred_logits_qhw.shape[1]), int(pred_logits_qhw.shape[2])
    if qh != gh or qw != gw:
        t = torch.from_numpy(pred_logits_qhw.astype(np.float32)).unsqueeze(0)
        pred_logits_qhw = (
            F.interpolate(t, size=(gh, gw), mode="bilinear", align_corners=False).squeeze(0).numpy()
        )
    q = int(pred_logits_qhw.shape[0])
    gt_bin = (gt_hw > 0).astype(bool)
    best_logits = pred_logits_qhw[0]
    best_iou = -1.0
    for qi in range(q):
        lg = pred_logits_qhw[qi]
        pb = lg > logit_thresh
        iou = _binary_iou(pb, gt_bin)
        if iou > best_iou:
            best_iou = iou
            best_logits = lg
    return best_logits.astype(np.float32), float(best_iou)


@dataclass
class CompareDataArgs:
    local_rank: int = 0
    vision_tower: str = "pretrained_model/CLIP/siglip-so400m-patch14-384"
    vision_tower_mask: str = "pretrained_model/mask2former/model_final_54b88a.pkl"
    lazy_preprocess: bool = False
    base_data_path: Optional[str] = field(default=None)
    model_path: str = ""
    mask_config: str = field(
        default="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
    )
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    model_map_name: str = "segearth_r2"
    version: str = "llava_phi"
    output_dir: str = "compare_out"
    eval_batch_size: int = 1
    dataloader_num_workers: int = 4
    max_eval_samples: int = 0
    dataset_name: str = "rrsisd"
    split: str = "test"
    zip_results: bool = False


def parse_args():
    p = argparse.ArgumentParser(description="Compare two SegEarth checkpoints on the same split.")
    p.add_argument("--model-a-path", type=str, required=True, help="Checkpoint A (e.g. baseline merged HF folder).")
    p.add_argument("--model-b-path", type=str, required=True, help="Checkpoint B (e.g. structured merged).")
    p.add_argument("--base-data-path", type=str, required=True)
    p.add_argument("--dataset-name", type=str, default="rrsisd", choices=["rrsisd", "lasers"])
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--mask-config", type=str, default=CompareDataArgs.mask_config)
    p.add_argument("--vision-tower", type=str, default=CompareDataArgs.vision_tower)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--dataloader-num-workers", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=0, help="0 = full split.")
    p.add_argument("--small-area-ratio-threshold", type=float, default=0.01)
    p.add_argument("--logit-thresh", type=float, default=0.0, help="Threshold on mask logits for binary metrics.")
    p.add_argument("--model-a-device", type=str, default="cuda:0")
    p.add_argument("--model-b-device", type=str, default=None, help="Defaults to same as model-a-device.")
    p.add_argument("--version", type=str, default="llava_phi")
    return p.parse_args()


def _move_model(model: torch.nn.Module, device: torch.device, dtype: Optional[torch.dtype] = None):
    # One SegEarth+M2F per GPU: fp32 is stable (bf16/fp16 can hit CUBLAS / illegal access in M2F paths).
    # Same-GPU sequential uses fp32 inside _process_batch_pair instead of this helper.
    if dtype is None:
        dtype = torch.float32
    model.to(dtype=dtype, device=device)
    model.eval()


def _model_param_dtype(model: torch.nn.Module) -> torch.dtype:
    return next(model.parameters()).dtype


def _process_batch_pair(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    inputs: Dict[str, Any],
    device_a: torch.device,
    device_b: torch.device,
    logit_thresh: float,
    small_th: float,
    sequential_single_gpu: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def _run_pair(out_a_local: List[Any], out_b_local: List[Any]) -> None:
        for seg, oa, ob in zip(inputs["seg_info"], out_a_local, out_b_local):
            gt = _extract_gt_hw(seg)
            if gt is None:
                continue
            la = oa.get("pred_logits_all")
            lb = ob.get("pred_logits_all")
            if la is None or lb is None:
                continue
            logits_a, _ = _pick_best_query_logits(la, gt, logit_thresh)
            logits_b, _ = _pick_best_query_logits(lb, gt, logit_thresh)
            if logits_a.shape != logits_b.shape:
                raise ValueError(f"logits shape mismatch A {logits_a.shape} vs B {logits_b.shape}")
            if logits_a.shape != gt.shape:
                raise ValueError(f"logits vs gt shape {logits_a.shape} vs {gt.shape}")

            mean_abs = float(np.mean(np.abs(logits_a - logits_b)))
            bin_a = (logits_a > logit_thresh).astype(np.uint8)
            bin_b = (logits_b > logit_thresh).astype(np.uint8)
            disagree = float(np.mean(bin_a != bin_b))
            iou_ab = _iou_pair(bin_a, bin_b)
            area_b = max(int(bin_b.sum()), 1)
            area_change = float(abs(int(bin_a.sum()) - int(bin_b.sum())) / area_b)
            iou_a_gt = _binary_iou(bin_a, gt)
            iou_b_gt = _binary_iou(bin_b, gt)
            delta_iou = float(iou_b_gt - iou_a_gt)
            h, w = gt.shape
            is_small = bool(float(gt.sum()) / float(h * w + 1e-9) < small_th)
            sample_key = f"{oa['image_name']}_{oa['id']}_{oa['mask_id']}"
            rows.append(
                {
                    "sample_key": sample_key,
                    "image_id": str(oa["image_name"]),
                    "data_id": str(oa["id"]),
                    "mask_id": str(oa["mask_id"]),
                    "subset_key": _resolve_subset_key(seg),
                    "is_small": int(is_small),
                    "mean_abs_logits_diff": mean_abs,
                    "binary_disagree_ratio": disagree,
                    "pred_mask_iou_ab": float(iou_ab),
                    "pred_area_change_ratio": area_change,
                    "iou_a_vs_gt": float(iou_a_gt),
                    "iou_b_vs_gt": float(iou_b_gt),
                    "delta_iou_b_minus_a": delta_iou,
                }
            )

    if sequential_single_gpu is not None:
        g = sequential_single_gpu
        dt = torch.float32
        model_a.to(device=g, dtype=dt)
        model_a.eval()
        ia = {k: (v.to(g) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        ia["token_refer_id"] = [ids.to(g) for ids in ia["token_refer_id"]]
        common_a = dict(
            input_ids=ia["input_ids"],
            attention_mask=ia["attention_mask"],
            images=ia["images"].to(device=g, dtype=dt),
            images_clip=ia["images_clip"].to(device=g, dtype=dt),
            seg_info=inputs["seg_info"],
            token_refer_id=ia["token_refer_id"],
            SEG_token_embedding_indices=ia["SEG_token_embedding_indices"],
            labels=ia["labels"],
            mask_num=ia["mask_num"],
            return_mask_logits_all=True,
        )
        with torch.inference_mode():
            out_a = model_a.eval_seg(**common_a)
        model_a.cpu()
        if g.type == "cuda":
            torch.cuda.synchronize(device=g)
            torch.cuda.empty_cache()

        model_b.to(device=g, dtype=dt)
        model_b.eval()
        ib = {k: (v.to(g) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        ib["token_refer_id"] = [ids.to(g) for ids in ib["token_refer_id"]]
        common_b = dict(
            input_ids=ib["input_ids"],
            attention_mask=ib["attention_mask"],
            images=ib["images"].to(device=g, dtype=dt),
            images_clip=ib["images_clip"].to(device=g, dtype=dt),
            seg_info=inputs["seg_info"],
            token_refer_id=ib["token_refer_id"],
            SEG_token_embedding_indices=ib["SEG_token_embedding_indices"],
            labels=ib["labels"],
            mask_num=ib["mask_num"],
            return_mask_logits_all=True,
        )
        with torch.inference_mode():
            out_b = model_b.eval_seg(**common_b)
        model_b.cpu()
        if g.type == "cuda":
            torch.cuda.synchronize(device=g)
            torch.cuda.empty_cache()

        _run_pair(out_a, out_b)
        return rows

    inputs_a = {k: (v.to(device_a) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    inputs_b = {k: (v.to(device_b) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    inputs_a["token_refer_id"] = [ids.to(device_a) for ids in inputs_a["token_refer_id"]]
    inputs_b["token_refer_id"] = [ids.to(device_b) for ids in inputs_b["token_refer_id"]]
    dt_a = _model_param_dtype(model_a)
    dt_b = _model_param_dtype(model_b)

    common_kw = dict(
        input_ids=inputs_a["input_ids"],
        attention_mask=inputs_a["attention_mask"],
        images=inputs_a["images"].to(device=device_a, dtype=dt_a),
        images_clip=inputs_a["images_clip"].to(device=device_a, dtype=dt_a),
        seg_info=inputs["seg_info"],
        token_refer_id=inputs_a["token_refer_id"],
        SEG_token_embedding_indices=inputs_a["SEG_token_embedding_indices"],
        labels=inputs_a["labels"],
        mask_num=inputs_a["mask_num"],
        return_mask_logits_all=True,
    )
    out_a = model_a.eval_seg(**common_kw)

    common_kw_b = dict(
        input_ids=inputs_b["input_ids"],
        attention_mask=inputs_b["attention_mask"],
        images=inputs_b["images"].to(device=device_b, dtype=dt_b),
        images_clip=inputs_b["images_clip"].to(device=device_b, dtype=dt_b),
        seg_info=inputs["seg_info"],
        token_refer_id=inputs_b["token_refer_id"],
        SEG_token_embedding_indices=inputs_b["SEG_token_embedding_indices"],
        labels=inputs_b["labels"],
        mask_num=inputs_b["mask_num"],
        return_mask_logits_all=True,
    )
    out_b = model_b.eval_seg(**common_kw_b)

    _run_pair(out_a, out_b)
    return rows


def _aggregate(rows: List[Dict[str, Any]], effect_eps: float) -> Dict[str, Any]:
    if len(rows) == 0:
        return {"n": 0}

    def mean(k: str) -> float:
        return float(np.mean([r[k] for r in rows]))

    mean_abs = np.array([r["mean_abs_logits_diff"] for r in rows], dtype=np.float64)
    delta = np.array([r["delta_iou_b_minus_a"] for r in rows], dtype=np.float64)
    is_small = np.array([r["is_small"] for r in rows], dtype=bool)

    def bucket(mask: np.ndarray, key: str) -> Optional[float]:
        if not mask.any():
            return None
        return float(np.mean([r[key] for i, r in enumerate(rows) if mask[i]]))

    out: Dict[str, Any] = {
        "n": len(rows),
        "output_effect_mean": mean("mean_abs_logits_diff"),
        "output_effect_nonzero_ratio": float(np.mean(mean_abs > effect_eps)),
        "binary_mask_disagree_ratio_mean": mean("binary_disagree_ratio"),
        "pred_mask_iou_ab_mean": mean("pred_mask_iou_ab"),
        "pred_area_change_ratio_mean": mean("pred_area_change_ratio"),
        "iou_a_vs_gt_mean": mean("iou_a_vs_gt"),
        "iou_b_vs_gt_mean": mean("iou_b_vs_gt"),
        "delta_iou_mean": float(np.mean(delta)),
        "delta_iou_positive_ratio": float(np.mean(delta > 1e-9)),
        "delta_iou_negative_ratio": float(np.mean(delta < -1e-9)),
        "delta_iou_near_zero_ratio": float(np.mean(np.abs(delta) <= 1e-9)),
        "output_diff_small_mean_abs": bucket(is_small, "mean_abs_logits_diff"),
        "output_diff_non_small_mean_abs": bucket(~is_small, "mean_abs_logits_diff"),
        "delta_iou_small_mean": bucket(is_small, "delta_iou_b_minus_a"),
        "delta_iou_non_small_mean": bucket(~is_small, "delta_iou_b_minus_a"),
    }
    return out


def _interpretation_hint(summary: Dict[str, Any]) -> Dict[str, str]:
    """Heuristic triage for structured vs baseline style experiments (B = model B)."""
    if summary.get("n", 0) == 0:
        return {"pattern": "empty", "notes_zh": "无有效样本（检查 GT 是否可从 seg_info 解析）。"}
    nz = float(summary.get("output_effect_nonzero_ratio", 0.0))
    mabs = float(summary.get("output_effect_mean", 0.0))
    dmean = float(summary.get("delta_iou_mean", 0.0))
    pos = float(summary.get("delta_iou_positive_ratio", 0.0))
    neg = float(summary.get("delta_iou_negative_ratio", 0.0))
    if nz < 0.02 and mabs < 1e-3:
        return {
            "pattern": "almost_no_output_change",
            "notes_zh": "A/B 预测几乎一致：最终决策层面差异极弱（在阈值与选 query 方式下）。",
        }
    if dmean > 0.005 or pos > 0.55:
        return {
            "pattern": "change_with_net_gain",
            "notes_zh": "B 相对 A 在 IoU 上整体偏正：存在有效决策改进信号（需结合具体任务再确认显著性）。",
        }
    if nz > 0.05 and dmean <= 0.0 and pos < 0.45:
        return {
            "pattern": "change_likely_perturbation_or_no_gain",
            "notes_zh": "输出差异明显但 ΔIoU 未整体为正：更像扰动或混合效应，不等于稳定收益。",
        }
    return {"pattern": "mixed", "notes_zh": "介于「几乎不变」与明确正/负收益之间，建议看 per-sample 与 by_subset。"}


def _aggregate_by_subset(rows: List[Dict[str, Any]], effect_eps: float) -> Dict[str, Any]:
    from collections import defaultdict

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[str(r.get("subset_key", "UNK"))].append(r)
    return {k: _aggregate(v, effect_eps) for k, v in buckets.items()}


def main():
    args = parse_args()
    device_b = args.model_b_device or args.model_a_device
    device_a = torch.device(args.model_a_device)
    device_b = torch.device(device_b)
    same_gpu = device_a == device_b and device_a.type == "cuda"
    sequential_gpu: Optional[torch.device] = device_a if same_gpu else None

    os.makedirs(args.output_dir, exist_ok=True)
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.version]

    data_args = CompareDataArgs(
        base_data_path=args.base_data_path,
        dataset_name=args.dataset_name,
        split=args.split,
        mask_config=args.mask_config,
        vision_tower=args.vision_tower,
        eval_batch_size=args.eval_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        max_eval_samples=args.max_samples,
    )

    tokenizer_a, model_a, _, _ = load_pretrained_model(
        os.path.expanduser(args.model_a_path),
        model_args=data_args,
        mask_config=args.mask_config,
        device="cpu",
    )
    tokenizer_b, model_b, _, _ = load_pretrained_model(
        os.path.expanduser(args.model_b_path),
        model_args=data_args,
        mask_config=args.mask_config,
        device="cpu",
    )
    if len(tokenizer_a) != len(tokenizer_b):
        print("[warn] tokenizer vocab sizes differ; using tokenizer from model A for the loader.")

    if same_gpu:
        print(
            f"[info] model A/B share {device_a}: sequential single-GPU (fp32 per forward) to avoid OOM / dtype issues."
        )
        model_a.cpu()
        model_b.cpu()
    else:
        _move_model(model_a, device_a)
        _move_model(model_b, device_b)

    clip_image_processor = SiglipImageProcessor.from_pretrained(args.vision_tower)
    data_collator = DataCollatorForCOCODatasetV2(
        tokenizer=tokenizer_a,
        clip_image_processor=clip_image_processor,
    )

    eval_sets = build_eval_datasets(data_args, tokenizer_a)
    all_rows: List[Dict[str, Any]] = []

    for split, eval_dataset in eval_sets:
        loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.dataloader_num_workers,
            pin_memory=False,
            collate_fn=data_collator,
        )
        for inputs in tqdm(loader, desc=f"compare[{split}]"):
            rows = _process_batch_pair(
                model_a,
                model_b,
                inputs,
                device_a,
                device_b,
                logit_thresh=args.logit_thresh,
                small_th=args.small_area_ratio_threshold,
                sequential_single_gpu=sequential_gpu,
            )
            all_rows.extend(rows)
            if args.max_samples > 0 and len(all_rows) >= args.max_samples:
                all_rows = all_rows[: args.max_samples]
                break
        if args.max_samples > 0 and len(all_rows) >= args.max_samples:
            break

    effect_eps = 1e-6
    summary = _aggregate(all_rows, effect_eps=effect_eps)
    summary["by_subset"] = _aggregate_by_subset(all_rows, effect_eps=effect_eps)
    summary["interpretation_hint"] = _interpretation_hint(summary)
    summary["model_a_path"] = os.path.abspath(args.model_a_path)
    summary["model_b_path"] = os.path.abspath(args.model_b_path)
    summary["split"] = args.split
    summary["dataset_name"] = args.dataset_name
    summary["small_area_ratio_threshold"] = args.small_area_ratio_threshold
    summary["logit_thresh"] = args.logit_thresh
    summary["effect_eps"] = effect_eps

    with open(os.path.join(args.output_dir, "compare_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(os.path.join(args.output_dir, "compare_per_sample.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
