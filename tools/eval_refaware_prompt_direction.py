#!/usr/bin/env python3
"""
Phase 2.5 diagnostic: paired eval raw prompt vs refaware prompt (same merged model, same GT).

Not the standard benchmark (inference normally uses raw prompt). Do not import as production eval.

Usage (from repo resegearth+source root, with PYTHONPATH including this repo):
  CUDA_VISIBLE_DEVICES=0 python tools/eval_refaware_prompt_direction.py \\
    --model_path ... --base_data_path ... --library_path ... \\
    --output_root outputs/source/refaware_prompt_direction_eval

Optional: --max_samples 500 for pilot (sets pilot_only in summary).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import tifffile as tiff
from tqdm import tqdm
from transformers import SiglipImageProcessor

# repo root = parent of tools/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from segearth_r2.datasets.dataset import (  # noqa: E402
    DataCollatorForCOCODatasetV2,
    RRSISDDataset,
)
from segearth_r2.utils import conversation as conversation_lib  # noqa: E402
from segearth_r2.utils.builder import load_pretrained_model  # noqa: E402
from segearth_r2.utils.concept_public_grounding_train import (  # noqa: E402
    build_rrsisd_refaware_exclusion_only_human_value,
    retrieve_matched_public_grounding,
    split_matched_concepts_target_and_reference,
)
from segearth_r2.utils.constants import REFER_TOKEN_INDEX  # noqa: E402


def shallow_copy_sample(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            out[k] = v.clone()
        else:
            out[k] = v
    return out


def prior_group_and_has_matched_from_rf(rf: Dict[str, Any]) -> Tuple[str, bool]:
    matched = [x for x in rf["matched_concepts"] if x]
    tgt = [x for x in rf["target_concepts"] if x]
    refc = [x for x in rf["reference_concepts"] if x]
    has_matched_prior = len(matched) > 0
    if not has_matched_prior:
        return "no_prior", False
    if tgt and not refc:
        return "target_prior", True
    if refc and not tgt:
        return "reference_only", True
    if tgt and refc:
        return "mixed", True
    return "mixed", True


def build_refaware_fields(
    ds: RRSISDDataset,
    *,
    instruction: str,
    category_name_str: Optional[str],
    lib_path: str,
) -> Dict[str, Any]:
    """Aligned with tools/probe_refaware_seg_hidden_effect.build_refaware_fields."""
    matched_all = retrieve_matched_public_grounding(instruction, lib_path)
    tgt_rows, ref_rows = split_matched_concepts_target_and_reference(
        matched_all, str(category_name_str).strip() if category_name_str else None
    )
    hv = build_rrsisd_refaware_exclusion_only_human_value(
        instruction,
        lib_path,
        str(category_name_str).strip() if category_name_str else None,
        matched_precalc=matched_all,
    )
    answer = "[SEG]"
    sources = [[{"from": "human", "value": hv}, {"from": "gpt", "value": "\n" + answer}]]
    text_dict = ds.preprocess_llama2(sources, ds.tokenizer)
    input_ids = text_dict["input_ids"][0]
    seg_id = ds.SEG_token_id
    seg_indices = torch.zeros_like(input_ids)
    seg_indices[input_ids == seg_id] = 1
    refer_indices = torch.zeros_like(input_ids)
    refer_indices[input_ids == REFER_TOKEN_INDEX] = 1
    return {
        "input_ids": input_ids,
        "labels": text_dict["labels"][0],
        "SEG_token_embedding_indices": seg_indices,
        "refer_embedding_indices": refer_indices,
        "matched_concepts": [str(r.get("concept") or "").strip() for r in matched_all if isinstance(r, dict)],
        "target_concepts": [str(r.get("concept") or "").strip() for r in tgt_rows if isinstance(r, dict)],
        "reference_concepts": [str(r.get("concept") or "").strip() for r in ref_rows if isinstance(r, dict)],
    }

# Metrics aligned with eval_val_metrics.py (same file-level helpers)
_RESEG_ROOT = os.path.dirname(REPO_ROOT)  # parent of resegearth+source = huangziyi/reseg
_EVAL_METRICS_PATH = os.path.join(_RESEG_ROOT, "eval_val_metrics.py")
if os.path.isfile(_EVAL_METRICS_PATH):
    import importlib.util

    _spec = importlib.util.spec_from_file_location("eval_val_metrics_phase25", _EVAL_METRICS_PATH)
    _em = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(_em)
    merge_masks = _em.merge_masks
    load_pred_mask = _em.load_pred_mask
    resize_pred_to_gt = _em.resize_pred_to_gt
    mask_iou = _em.mask_iou
    dice_score = _em.dice_score
    mask_recall = _em.mask_recall
    mask_precision = _em.mask_precision
    mask_inter_union = _em.mask_inter_union
    precision_at_threshold = _em.precision_at_threshold
    cumulative_iou = _em.cumulative_iou
    mask_to_bbox = _em.mask_to_bbox
    bbox_iou = _em.bbox_iou
    box_giou = _em.box_giou
    box_ciou = _em.box_ciou
else:
    raise FileNotFoundError(f"eval_val_metrics.py not found at {_EVAL_METRICS_PATH}")


def load_rrsisd_test_samples(base_data_path: str) -> Tuple[List[Dict[str, Any]], str]:
    """Same ordering as RRSISDDataset split=test (refs with split==test)."""
    import pickle

    refs_path = os.path.join(base_data_path, "rrsisd", "refs(unc).p")
    instances_path = os.path.join(base_data_path, "rrsisd", "instances.json")
    with open(refs_path, "rb") as f:
        refs = pickle.load(f)
    with open(instances_path, "r", encoding="utf-8") as f:
        instances = json.load(f)
    ann_dict = {ann["id"]: ann for ann in instances["annotations"]}
    test_refs = [r for r in refs if r["split"] == "test"]
    img_dir = os.path.join(base_data_path, "images", "rrsisd", "JPEGImages")
    rows = []
    for ref in test_refs:
        ann = ann_dict[ref["ann_id"]]
        sentence = ""
        if ref.get("sentences"):
            sentence = ref["sentences"][0].get("sent", ref["sentences"][0].get("raw", ""))
        rows.append(
            {
                "image_name": ref["file_name"],
                "id": ref["ref_id"],
                "description": sentence,
                "mask": ann["segmentation"],
            }
        )
    return rows, img_dir


def ref_instruction_category(dataset: RRSISDDataset, idx: int) -> Tuple[str, Optional[str], int, str]:
    ref = dataset.reason_file[idx]
    data_id = int(ref["ref_id"])
    if len(ref["sentences"]) > 0 and "sent" in ref["sentences"][0]:
        instruction = ref["sentences"][0]["sent"].strip()
    elif len(ref["sentences"]) > 0 and "raw" in ref["sentences"][0]:
        instruction = ref["sentences"][0]["raw"].strip()
    else:
        ann = dataset.ann_dict[ref["ann_id"]]
        cat_id = ann.get("categories_id", ann.get("category_id"))
        cat_name = dataset.category_dict.get(int(cat_id)) if cat_id is not None else "target"
        instruction = f"segment the {cat_name} in this remote sensing image"
    cid = ref.get("category_id")
    if cid is None:
        ann_probe = dataset.ann_dict.get(ref.get("ann_id"))
        if isinstance(ann_probe, dict):
            cid = ann_probe.get("categories_id")
            if cid is None:
                cid = ann_probe.get("category_id")
    try:
        cid_int = int(cid) if cid is not None else None
    except (TypeError, ValueError):
        cid_int = None
    category_name_str = dataset.category_dict.get(cid_int) if cid_int is not None else None
    return instruction, category_name_str, data_id, ref["file_name"]


def prior_group_from_instruction(
    instruction: str, category_name_str: Optional[str], lib_path: str
) -> Tuple[str, Dict[str, Any]]:
    """Build rf-like dict for grouping without needing full token tensors."""
    matched_all = retrieve_matched_public_grounding(instruction, lib_path)
    tgt_rows, ref_rows = split_matched_concepts_target_and_reference(
        matched_all, str(category_name_str).strip() if category_name_str else None
    )
    rf = {
        "matched_concepts": [
            str(r.get("concept") or "").strip() for r in matched_all if isinstance(r, dict)
        ],
        "target_concepts": [str(r.get("concept") or "").strip() for r in tgt_rows if isinstance(r, dict)],
        "reference_concepts": [str(r.get("concept") or "").strip() for r in ref_rows if isinstance(r, dict)],
    }
    pg, _ = prior_group_and_has_matched_from_rf(rf)
    return pg, rf


def direction_label(delta_iou: float) -> str:
    if delta_iou >= 0.01:
        return "improved"
    if delta_iou <= -0.01:
        return "degraded"
    return "stable"


def _pred_tif_candidates(image_file: str, data_id: int, split_stem: str) -> List[str]:
    stem = os.path.splitext(image_file)[0]
    return [
        f"{image_file}_{data_id}_{split_stem}_0.tif",
        f"{stem}_{data_id}_{split_stem}_0.tif",
    ]


def paired_preds_exist(raw_dir: str, ref_dir: str, image_file: str, data_id: int, split_stem: str) -> bool:
    """True if at least one candidate exists in BOTH dirs (resume/skip forward)."""
    cands = _pred_tif_candidates(image_file, data_id, split_stem)

    def _dir_has(d: str) -> bool:
        return any(os.path.isfile(os.path.join(d, c)) for c in cands)

    return _dir_has(raw_dir) and _dir_has(ref_dir)


@dataclass
class ScriptDataArgs:
    vision_tower: str = ""
    vision_tower_mask: str = ""
    lazy_preprocess: bool = False
    base_data_path: str = ""
    model_path: str = ""
    mask_config: str = ""
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = None
    model_map_name: str = "segearth_r2"
    version: str = "llava_phi"
    output_dir: str = ""
    concept_public_semantic_library: Optional[str] = None
    concept_match_strict: bool = False
    concept_refaware_prior: bool = False
    seg_task: str = "instance"


def run_forward_save_tifs(
    model,
    device: torch.device,
    infer_dtype: torch.dtype,
    batch: Dict[str, Any],
    save_dir: str,
    split_stem: str,
) -> None:
    model.eval()
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    inputs["token_refer_id"] = [ids.to(device) for ids in inputs["token_refer_id"]]
    with torch.no_grad():
        outputs = model.eval_seg(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            images=inputs["images"].to(device=device, dtype=infer_dtype),
            images_clip=inputs["images_clip"].to(device=device, dtype=infer_dtype),
            seg_info=inputs["seg_info"],
            token_refer_id=inputs["token_refer_id"],
            SEG_token_embedding_indices=inputs["SEG_token_embedding_indices"],
            labels=inputs["labels"],
            mask_num=inputs["mask_num"],
        )
    os.makedirs(save_dir, exist_ok=True)
    for output in outputs:
        pred_mask = output["pred"]
        image_name = output["image_name"]
        sample_id = output["id"]
        mask_id = output["mask_id"]
        mask_save_name = f"{image_name}_{sample_id}_{split_stem}_{mask_id}.tif"
        if pred_mask.ndim > 2:
            pred_mask = np.squeeze(pred_mask)
        out_path = os.path.join(save_dir, mask_save_name)
        tiff.imwrite(out_path, pred_mask.astype(np.uint8))


def aggregate_metrics_from_preds(
    val_rows: List[Dict[str, Any]], pred_dir: str
) -> Tuple[Dict[str, float], List[Dict[str, Any]], int, int]:
    """Same skip rules as eval_val_metrics: skip empty GT; skip missing pred."""
    results: List[Dict[str, Any]] = []
    total_inter = 0
    total_union = 0
    skipped_empty_gt = 0
    missing_pred = 0
    for idx, sample in enumerate(val_rows):
        gt_mask = merge_masks(sample["mask"])
        if gt_mask.sum() == 0:
            skipped_empty_gt += 1
            continue
        image_name = sample["image_name"]
        sample_id = sample["id"]
        image_stem = os.path.splitext(image_name)[0]
        candidates = [
            os.path.join(pred_dir, f"{image_name}_{sample_id}_test_0.tif"),
            os.path.join(pred_dir, f"{image_stem}_{sample_id}_test_0.tif"),
        ]
        pred_path = None
        for p in candidates:
            if os.path.isfile(p):
                pred_path = p
                break
        if pred_path is None:
            missing_pred += 1
            continue
        pred_mask = load_pred_mask(pred_path)
        pred_mask = resize_pred_to_gt(pred_mask, gt_mask, sample["image_name"])
        sample_iou = mask_iou(pred_mask, gt_mask)
        sample_dice = dice_score(pred_mask, gt_mask)
        sample_recall = mask_recall(pred_mask, gt_mask)
        sample_precision = mask_precision(pred_mask, gt_mask)
        sample_inter, sample_union = mask_inter_union(pred_mask, gt_mask)
        total_inter += sample_inter
        total_union += sample_union
        gt_box = mask_to_bbox(gt_mask)
        pred_box = mask_to_bbox(pred_mask)
        if pred_box is not None and gt_box is not None:
            sample_box_iou = bbox_iou(pred_box, gt_box)
            sample_box_giou = box_giou(pred_box, gt_box)
            sample_box_ciou = box_ciou(pred_box, gt_box)
        else:
            sample_box_iou = 0.0
            sample_box_giou = -1.0
            sample_box_ciou = -1.0
        results.append(
            {
                "idx": idx,
                "id": sample_id,
                "image_name": image_name,
                "description": sample["description"],
                "pred_path": pred_path,
                "iou": sample_iou,
                "dice": sample_dice,
                "recall": sample_recall,
                "precision": sample_precision,
                "box_iou": sample_box_iou,
                "box_giou": sample_box_giou,
                "box_ciou": sample_box_ciou,
                "inter": sample_inter,
                "union": sample_union,
                "pred_area": int(pred_mask.sum()),
            }
        )
    if not results:
        return {}, [], skipped_empty_gt, missing_pred
    iou_list = [x["iou"] for x in results]
    miou = float(np.mean(iou_list))
    mdice = float(np.mean([x["dice"] for x in results]))
    mrecall = float(np.mean([x["recall"] for x in results]))
    mprec = float(np.mean([x["precision"] for x in results]))
    giou = miou
    ciou = cumulative_iou(total_inter, total_union)
    oiou = ciou
    pr_05 = precision_at_threshold(iou_list, 0.5)
    pr_06 = precision_at_threshold(iou_list, 0.6)
    pr_07 = precision_at_threshold(iou_list, 0.7)
    pr_08 = precision_at_threshold(iou_list, 0.8)
    pr_09 = precision_at_threshold(iou_list, 0.9)
    mean_box_iou = float(np.mean([x["box_iou"] for x in results]))
    mean_box_giou = float(np.mean([x["box_giou"] for x in results]))
    mean_box_ciou = float(np.mean([x["box_ciou"] for x in results]))
    metrics = {
        "mIoU": miou,
        "gIoU": giou,
        "oIoU": oiou,
        "cIoU": ciou,
        "mDice": mdice,
        "mRecall": mrecall,
        "mPrecision": mprec,
        "Pr@0.5": pr_05,
        "Pr@0.6": pr_06,
        "Pr@0.7": pr_07,
        "Pr@0.8": pr_08,
        "Pr@0.9": pr_09,
        "box_level_iou": mean_box_iou,
        "box_level_giou": mean_box_giou,
        "box_level_ciou": mean_box_ciou,
    }
    return metrics, results, skipped_empty_gt, missing_pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_path",
        type=str,
        default="/home/wangchengjun/huangziyi/reseg/output/source/rrsisd_public_semantic_v2_refaware_exclusion_only_28w/rrsisd_public_semantic_v2_refaware_exclusion_only_28w_merged_best",
    )
    ap.add_argument("--base_data_path", type=str, default="/home/wangchengjun/huangziyi/data/RRSISD")
    ap.add_argument(
        "--library_path",
        type=str,
        default=os.path.join(REPO_ROOT, "configs", "concept_public_semantic_library_v2.json"),
    )
    ap.add_argument(
        "--mask_config",
        type=str,
        default=os.path.join(REPO_ROOT, "segearth_r2", "model", "mask_decoder", "mask_config", "maskformer2_swin_base_384_bs16_50ep.yaml"),
    )
    ap.add_argument("--output_root", type=str, default=os.path.join(REPO_ROOT, "outputs", "source", "refaware_prompt_direction_eval"))
    ap.add_argument("--max_samples", type=int, default=0, help="0 = full test split; e.g. 500 for pilot")
    ap.add_argument("--dataloader_num_workers", type=int, default=4)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip samples whose raw+refaware pred .tif files already exist (restart-safe)",
    )
    args = ap.parse_args()

    model_path = os.path.expanduser(args.model_path)
    if not os.path.isdir(model_path) or not os.path.isfile(os.path.join(model_path, "config.json")):
        print(f"[FATAL] model_path invalid or missing config.json: {model_path}", file=sys.stderr)
        return 2
    if "checkpoint" in os.path.basename(model_path).lower():
        print(f"[FATAL] model_path looks like a checkpoint dir: {model_path}", file=sys.stderr)
        return 2

    base_data_path = os.path.expanduser(args.base_data_path)
    refs_p = os.path.join(base_data_path, "rrsisd", "refs(unc).p")
    if not os.path.isfile(refs_p):
        print(f"[FATAL] RRSISD refs not found: {refs_p}", file=sys.stderr)
        return 2

    lib_path = os.path.expanduser(args.library_path)
    if not os.path.isfile(lib_path):
        print(f"[FATAL] library_path not found: {lib_path}", file=sys.stderr)
        return 2

    raw_dir = os.path.join(args.output_root, "raw_prompt", "test_results")
    ref_dir = os.path.join(args.output_root, "refaware_prompt", "test_results")
    metrics_dir = os.path.join(args.output_root, "metrics")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Load model config for vision tower paths
    with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
        cfg_json = json.load(f)
    mm_vision = cfg_json.get("mm_vision_tower") or cfg_json.get("vision_tower")
    vision_mask_default = os.path.join(_RESEG_ROOT, "pretrained_model", "mask2former", "model_final_54b88a.pkl")
    data_args = ScriptDataArgs(
        vision_tower=mm_vision or "/home/wangchengjun/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384",
        vision_tower_mask=vision_mask_default,
        base_data_path=base_data_path,
        model_path=model_path,
        mask_config=args.mask_config,
        concept_public_semantic_library=None,
        concept_refaware_prior=False,
    )
    if not os.path.isdir(data_args.vision_tower) and not os.path.isfile(data_args.vision_tower):
        # HF id or local dir
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("[FATAL] CUDA required for this eval script (matches eval.py).", file=sys.stderr)
        return 2

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        model_args=data_args,
        mask_config=args.mask_config,
        device="cuda",
    )
    # fp16 reduces VRAM vs eval.py float32; sufficient for direction-of-change diagnostic
    infer_dtype = torch.float16
    model.to(dtype=infer_dtype, device=device)
    model.eval()

    conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version]
    clip_image_processor = SiglipImageProcessor.from_pretrained(data_args.vision_tower)
    collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer, clip_image_processor=clip_image_processor)

    raw_ds = RRSISDDataset(
        base_data_path=base_data_path,
        tokenizer=tokenizer,
        data_args=data_args,
        split="test",
    )
    n_total = len(raw_ds)
    n_run = n_total if args.max_samples <= 0 else min(n_total, args.max_samples)
    pilot_only = bool(args.max_samples > 0 and args.max_samples < n_total)

    split_stem = "test"
    for idx in tqdm(range(n_run), desc="paired infer (raw then refaware)", unit="sample"):
        _, _, data_id, img_file = ref_instruction_category(raw_ds, idx)
        if args.resume and paired_preds_exist(raw_dir, ref_dir, img_file, data_id, split_stem):
            continue
        raw_item = raw_ds[idx]
        instruction, category_name_str, _, _ = ref_instruction_category(raw_ds, idx)
        batch_raw = collator([shallow_copy_sample(raw_item)])
        run_forward_save_tifs(model, device, infer_dtype, batch_raw, raw_dir, split_stem)

        rf = build_refaware_fields(
            raw_ds,
            instruction=instruction,
            category_name_str=category_name_str,
            lib_path=lib_path,
        )
        ref_item = shallow_copy_sample(raw_item)
        ref_item["input_ids"] = rf["input_ids"]
        ref_item["labels"] = rf["labels"]
        ref_item["SEG_token_embedding_indices"] = rf["SEG_token_embedding_indices"]
        ref_item["refer_embedding_indices"] = rf["refer_embedding_indices"]
        batch_ref = collator([shallow_copy_sample(ref_item)])
        run_forward_save_tifs(model, device, infer_dtype, batch_ref, ref_dir, split_stem)

    # --- metrics on same val_rows ordering as dataset indices 0..n_run-1 ---
    val_rows_full, _ = load_rrsisd_test_samples(base_data_path)
    val_rows = val_rows_full[:n_run]

    raw_m, raw_details, sk_r, miss_r = aggregate_metrics_from_preds(val_rows, raw_dir)
    ref_m, ref_details, sk_a, miss_a = aggregate_metrics_from_preds(val_rows, ref_dir)
    if not raw_details or not ref_details:
        print("[FATAL] No overlapping metrics rows.", file=sys.stderr)
        return 3

    # index raw/ref details by idx
    raw_by_idx = {r["idx"]: r for r in raw_details}
    ref_by_idx = {r["idx"]: r for r in ref_details}
    common_idx = sorted(set(raw_by_idx.keys()) & set(ref_by_idx.keys()))

    # per-sample CSV + direction counts
    per_sample_path = os.path.join(args.output_root, "per_sample_direction_diff.csv")
    focus_cats = [
        "airport",
        "overpass",
        "vehicle",
        "ship",
        "harbor",
        "road",
        "bridge",
        "building",
        "parking lot",
        "playground",
        "chimney",
        "golffield",
        "basketballcourt",
    ]

    improved = degraded = stable = 0
    strong_improved = strong_degraded = 0
    per_sample_rows: List[Dict[str, Any]] = []

    # load probe for optional merge
    probe_by_ref: Dict[int, Dict[str, Any]] = {}
    probe_csv = os.path.join(REPO_ROOT, "outputs", "source", "refaware_hidden_probe_100", "hidden_probe_per_sample.csv")
    if os.path.isfile(probe_csv):
        with open(probe_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rid = int(row.get("ref_id", -1))
                except (TypeError, ValueError):
                    continue
                probe_by_ref[rid] = row

    cat_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "sample_count": 0,
            "raw_ious": [],
            "ref_ious": [],
            "raw_dice": [],
            "ref_dice": [],
            "raw_prec": [],
            "ref_prec": [],
            "raw_rec": [],
            "ref_rec": [],
            "raw_area": [],
            "ref_area": [],
            "improved": 0,
            "degraded": 0,
            "stable": 0,
        }
    )
    pg_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "sample_count": 0,
            "raw_ious": [],
            "ref_ious": [],
            "raw_dice": [],
            "ref_dice": [],
            "improved": 0,
            "degraded": 0,
            "stable": 0,
            "strong_improved": 0,
            "strong_degraded": 0,
            "delta_areas": [],
        }
    )

    for i in common_idx:
        rr = raw_by_idx[i]
        ar = ref_by_idx[i]
        diou = float(ar["iou"]) - float(rr["iou"])
        ddice = float(ar["dice"]) - float(rr["dice"])
        dprec = float(ar["precision"]) - float(rr["precision"])
        drec = float(ar["recall"]) - float(rr["recall"])
        darea = int(ar["pred_area"]) - int(rr["pred_area"])
        dl = direction_label(diou)
        if dl == "improved":
            improved += 1
        elif dl == "degraded":
            degraded += 1
        else:
            stable += 1
        if diou >= 0.05:
            strong_improved += 1
        if diou <= -0.05:
            strong_degraded += 1

        instruction, category_name_str, ref_id, _ = ref_instruction_category(raw_ds, i)
        pg, rf = prior_group_from_instruction(instruction, category_name_str, lib_path)
        cat_name = (category_name_str or "").strip() or "unknown"

        prow = {
            "idx": i,
            "ref_id": ref_id,
            "category_name": cat_name,
            "expression": instruction.replace("\n", " ")[:500],
            "matched_concepts": json.dumps(rf["matched_concepts"], ensure_ascii=False),
            "target_concepts": json.dumps(rf["target_concepts"], ensure_ascii=False),
            "reference_concepts": json.dumps(rf["reference_concepts"], ensure_ascii=False),
            "prior_group": pg,
            "raw_iou": rr["iou"],
            "refaware_iou": ar["iou"],
            "delta_iou": diou,
            "raw_dice": rr["dice"],
            "refaware_dice": ar["dice"],
            "delta_dice": ddice,
            "raw_pred_area": rr["pred_area"],
            "refaware_pred_area": ar["pred_area"],
            "delta_pred_area": darea,
            "raw_precision": rr["precision"],
            "refaware_precision": ar["precision"],
            "delta_precision": dprec,
            "raw_recall": rr["recall"],
            "refaware_recall": ar["recall"],
            "delta_recall": drec,
            "direction_label": dl,
            "strong_improved": diou >= 0.05,
            "strong_degraded": diou <= -0.05,
        }
        if ref_id in probe_by_ref:
            p = probe_by_ref[ref_id]
            for k in (
                "signal_survival",
                "projected_query_cosine_raw_vs_refaware",
                "mask_iou_raw_vs_refaware",
                "llm_seg_cosine_raw_vs_refaware",
            ):
                if k in p:
                    prow[f"probe_{k}"] = p.get(k)
        per_sample_rows.append(prow)

        cs = cat_stats[cat_name]
        cs["sample_count"] += 1
        cs["raw_ious"].append(float(rr["iou"]))
        cs["ref_ious"].append(float(ar["iou"]))
        cs["raw_dice"].append(float(rr["dice"]))
        cs["ref_dice"].append(float(ar["dice"]))
        cs["raw_prec"].append(float(rr["precision"]))
        cs["ref_prec"].append(float(ar["precision"]))
        cs["raw_rec"].append(float(rr["recall"]))
        cs["ref_rec"].append(float(ar["recall"]))
        cs["raw_area"].append(float(rr["pred_area"]))
        cs["ref_area"].append(float(ar["pred_area"]))
        if dl == "improved":
            cs["improved"] += 1
        elif dl == "degraded":
            cs["degraded"] += 1
        else:
            cs["stable"] += 1

        gs = pg_stats[pg]
        gs["sample_count"] += 1
        gs["raw_ious"].append(float(rr["iou"]))
        gs["ref_ious"].append(float(ar["iou"]))
        gs["raw_dice"].append(float(rr["dice"]))
        gs["ref_dice"].append(float(ar["dice"]))
        gs["delta_areas"].append(float(darea))
        if dl == "improved":
            gs["improved"] += 1
        elif dl == "degraded":
            gs["degraded"] += 1
        else:
            gs["stable"] += 1
        if diou >= 0.05:
            gs["strong_improved"] += 1
        if diou <= -0.05:
            gs["strong_degraded"] += 1

    fieldnames_set = set()
    for row in per_sample_rows:
        fieldnames_set.update(row.keys())
    base_order = [
        "idx", "ref_id", "category_name", "expression", "matched_concepts", "target_concepts",
        "reference_concepts", "prior_group", "raw_iou", "refaware_iou", "delta_iou",
        "raw_dice", "refaware_dice", "delta_dice", "raw_pred_area", "refaware_pred_area",
        "delta_pred_area", "raw_precision", "refaware_precision", "delta_precision",
        "raw_recall", "refaware_recall", "delta_recall", "direction_label",
        "strong_improved", "strong_degraded",
    ]
    fieldnames = [k for k in base_order if k in fieldnames_set]
    fieldnames += sorted(fieldnames_set - set(fieldnames))
    with open(per_sample_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in per_sample_rows:
            w.writerow(row)

    # overall metrics table
    keys = [
        "mIoU",
        "gIoU",
        "oIoU",
        "cIoU",
        "mDice",
        "mPrecision",
        "mRecall",
        "Pr@0.5",
        "box_level_giou",
    ]
    overall_rows = []
    delta_miou = float(ref_m["mIoU"]) - float(raw_m["mIoU"])
    for k in keys:
        rv = raw_m.get(k, float("nan"))
        av = ref_m.get(k, float("nan"))
        dv = float(av) - float(rv) if math.isfinite(rv) and math.isfinite(av) else float("nan")
        rel = dv / rv if math.isfinite(rv) and abs(rv) > 1e-12 and math.isfinite(dv) else float("nan")
        overall_rows.append(
            {
                "metric": k,
                "raw_prompt": rv,
                "refaware_prompt": av,
                "delta_refaware_minus_raw": dv,
                "relative_delta": rel,
            }
        )
    overall_csv = os.path.join(metrics_dir, "overall_metrics.csv")
    with open(overall_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(overall_rows[0].keys()))
        w.writeheader()
        for row in overall_rows:
            w.writerow(row)

    # per-category CSV
    cat_csv_path = os.path.join(metrics_dir, "per_category_metrics.csv")
    cat_rows_out = []
    for cat, cs in sorted(cat_stats.items(), key=lambda x: -x[1]["sample_count"]):
        n = cs["sample_count"]
        if n == 0:
            continue
        r_miou = float(np.mean(cs["raw_ious"]))
        a_miou = float(np.mean(cs["ref_ious"]))
        r_md = float(np.mean(cs["raw_dice"]))
        a_md = float(np.mean(cs["ref_dice"]))
        r_mp = float(np.mean(cs["raw_prec"]))
        a_mp = float(np.mean(cs["ref_prec"]))
        r_mr = float(np.mean(cs["raw_rec"]))
        a_mr = float(np.mean(cs["ref_rec"]))
        cat_rows_out.append(
            {
                "category_name": cat,
                "sample_count": n,
                "raw_mIoU": r_miou,
                "refaware_mIoU": a_miou,
                "delta_mIoU": a_miou - r_miou,
                "raw_mDice": r_md,
                "refaware_mDice": a_md,
                "delta_mDice": a_md - r_md,
                "raw_precision": r_mp,
                "refaware_precision": a_mp,
                "delta_precision": a_mp - r_mp,
                "raw_recall": r_mr,
                "refaware_recall": a_mr,
                "delta_recall": a_mr - r_mr,
                "raw_pred_area_mean": float(np.mean(cs["raw_area"])),
                "refaware_pred_area_mean": float(np.mean(cs["ref_area"])),
                "delta_pred_area_mean": float(np.mean(cs["ref_area"])) - float(np.mean(cs["raw_area"])),
                "improved_sample_count": cs["improved"],
                "degraded_sample_count": cs["degraded"],
                "stable_sample_count": cs["stable"],
            }
        )
    with open(cat_csv_path, "w", newline="", encoding="utf-8") as f:
        fn = list(cat_rows_out[0].keys()) if cat_rows_out else []
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for row in cat_rows_out:
            w.writerow(row)

    # prior group CSV
    pg_csv = os.path.join(metrics_dir, "per_prior_group_metrics.csv")
    pg_out = []
    for pg, gs in sorted(pg_stats.items()):
        n = gs["sample_count"]
        if n == 0:
            continue
        r_miou = float(np.mean(gs["raw_ious"]))
        a_miou = float(np.mean(gs["ref_ious"]))
        r_md = float(np.mean(gs["raw_dice"]))
        a_md = float(np.mean(gs["ref_dice"]))
        pg_out.append(
            {
                "prior_group": pg,
                "sample_count": n,
                "raw_mIoU": r_miou,
                "refaware_mIoU": a_miou,
                "delta_mIoU": a_miou - r_miou,
                "raw_mDice": r_md,
                "refaware_mDice": a_md,
                "delta_mDice": a_md - r_md,
                "improved_count": gs["improved"],
                "degraded_count": gs["degraded"],
                "stable_count": gs["stable"],
                "strong_improved_count": gs["strong_improved"],
                "strong_degraded_count": gs["strong_degraded"],
                "mean_delta_pred_area": float(np.mean(gs["delta_areas"])) if gs["delta_areas"] else 0.0,
            }
        )
    with open(pg_csv, "w", newline="", encoding="utf-8") as f:
        fn = list(pg_out[0].keys()) if pg_out else []
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for row in pg_out:
            w.writerow(row)

    # Decision per user rules (overall delta mIoU first)
    d_miou = delta_miou
    if pilot_only:
        decision = "D"
        decision_note = "pilot_only=True: not valid for final A/B/C gate"
        stage0_rec = "do_not_enter_stage0_pending_full_eval"
    elif abs(d_miou) < 0.001:
        decision = "C"
        decision_note = "prior_direction_weak_or_negligible"
        stage0_rec = "do_not_enter_stage0"
    elif d_miou > 0:
        if improved < degraded:
            decision = "B"
            decision_note = "positive overall mIoU but improved_count < degraded_count (skewed sample direction)"
            stage0_rec = "caution_positive_mean_mIoU_but_improved_lt_degraded"
        else:
            decision = "A"
            decision_note = "prior_direction_positive"
            stage0_rec = "may_discuss_stage0_projector_after_reviewing_category_mix"
    else:
        decision = "B"
        decision_note = "prior_direction_negative_or_non_positive_overall_mIoU"
        stage0_rec = "do_not_enter_stage0"

    # top categories by delta mIoU
    cat_sorted_pos = sorted(cat_rows_out, key=lambda x: x["delta_mIoU"], reverse=True)[:10]
    cat_sorted_neg = sorted(cat_rows_out, key=lambda x: x["delta_mIoU"])[:10]

    summary = {
        "model_path": model_path,
        "resolved_note": (
            "Actual model_path used (user-provided path missing one parent segment): "
            "output/source/rrsisd_public_semantic_v2_refaware_exclusion_only_28w/rrsisd_public_semantic_v2_refaware_exclusion_only_28w_merged_best"
        ),
        "split": "test",
        "sample_count": len(common_idx),
        "n_run_requested": n_run,
        "pilot_only": pilot_only,
        "skipped_empty_gt_raw": sk_r,
        "skipped_empty_gt_refaware": sk_a,
        "missing_pred_raw": miss_r,
        "missing_pred_refaware": miss_a,
        "overall_delta_miou": d_miou,
        "overall_delta_summary": decision_note,
        "raw_mIoU": raw_m["mIoU"],
        "refaware_mIoU": ref_m["mIoU"],
        "category_delta_top10_positive": cat_sorted_pos,
        "category_delta_top10_negative": cat_sorted_neg,
        "prior_group_delta_summary": pg_out,
        "improved_count": improved,
        "degraded_count": degraded,
        "stable_count": stable,
        "strong_improved_count": strong_improved,
        "strong_degraded_count": strong_degraded,
        "decision": decision,
        "stage0_recommendation": stage0_rec,
        "recommendation": (
            "If decision A and full split: prior may help under refaware inference; still not standard eval. "
            "If B/C/D or non-positive mIoU: pause prior-aware projector and prompt-prior emphasis; fix library/prompt before structure."
        ),
        "focus_categories_present": {fc: (fc in cat_stats) for fc in focus_cats},
    }

    # probe correlation note (optional)
    if os.path.isfile(probe_csv) and per_sample_rows:
        xs = []
        ys = []
        for row in per_sample_rows:
            if row.get("probe_signal_survival") not in (None, "", "nan"):
                try:
                    xs.append(float(row["probe_signal_survival"]))
                    ys.append(float(row["delta_iou"]))
                except (TypeError, ValueError):
                    pass
        if len(xs) >= 5:
            summary["probe_signal_survival_vs_delta_iou_n"] = len(xs)
            summary["probe_signal_survival_vs_delta_iou_corr"] = float(np.corrcoef(np.array(xs), np.array(ys))[0, 1])
        else:
            summary["probe_merge_note"] = "Too few probe-aligned ref_id rows for correlation"
    if not os.path.isfile(probe_csv):
        summary["probe_merge_note"] = "hidden_probe_per_sample.csv not found at expected path"

    summ_path = os.path.join(args.output_root, "direction_eval_summary.json")
    with open(summ_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Markdown report
    doc_path = os.path.join(REPO_ROOT, "docs", "REFAWARE_PROMPT_DIRECTION_EVAL.md")
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("# Refaware prompt direction eval (Phase 2.5)\n\n")
        f.write("**Diagnostic only** — not standard benchmark inference.\n\n")
        f.write(f"- **model_path**: `{model_path}`\n")
        f.write(f"- **split**: test\n")
        f.write(f"- **sample_count (paired)**: {len(common_idx)}\n")
        f.write(f"- **pilot_only**: {pilot_only}\n\n")
        f.write("## Highest-priority result\n\n")
        f.write(f"- **overall delta mIoU** (refaware − raw): **{d_miou:.6f}**\n")
        f.write(f"- **decision**: **{decision}** — {decision_note}\n")
        f.write(f"- **stage0_recommendation**: **{stage0_rec}**\n\n")
        if not pilot_only and d_miou <= 0:
            f.write("> **do_not_enter_stage0** — overall delta mIoU ≤ 0 per project gate.\n\n")
        f.write("## Overall metrics (excerpt)\n\n")
        f.write("| metric | raw | refaware | delta |\n")
        f.write("|--------|-----|----------|-------|\n")
        for row in overall_rows:
            f.write(
                f"| {row['metric']} | {row['raw_prompt']:.6f} | {row['refaware_prompt']:.6f} | {row['delta_refaware_minus_raw']:.6f} |\n"
            )
        f.write("\n## Per-prior-group delta mIoU\n\n")
        for row in pg_out:
            f.write(
                f"- **{row['prior_group']}**: n={row['sample_count']}, delta_mIoU={row['delta_mIoU']:.6f}, "
                f"improved/degraded/stable={row['improved_count']}/{row['degraded_count']}/{row['stable_count']}\n"
            )
        f.write("\n## Sample direction counts\n\n")
        f.write(
            f"- improved / degraded / stable: **{improved}** / **{degraded}** / **{stable}**\n"
            f"- strong_improved / strong_degraded: **{strong_improved}** / **{strong_degraded}**\n"
        )
        f.write("\n## Outputs\n\n")
        f.write(f"- raw preds: `{raw_dir}`\n")
        f.write(f"- refaware preds: `{ref_dir}`\n")
        f.write(f"- metrics: `{metrics_dir}`\n")
        f.write(f"- per-sample: `{per_sample_path}`\n")
        f.write(f"- summary json: `{summ_path}`\n")

    print(json.dumps({"overall_delta_miou": d_miou, "decision": decision, "stage0_recommendation": stage0_rec}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
