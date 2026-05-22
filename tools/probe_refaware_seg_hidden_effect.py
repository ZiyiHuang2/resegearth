#!/usr/bin/env python3
"""
Phase-2 mechanism probe (read-only): compare [SEG] LLM hidden states and projected
segmentation queries under raw vs refaware prompts (Probe A) and baseline vs refaware
merged weights (Probe B). Does not train, save checkpoints, or modify repo files.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from detectron2.structures import ImageList
from transformers import SiglipImageProcessor

# Repo root (parent of tools/)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from segearth_r2.datasets.dataset import (  # noqa: E402
    DataCollatorForCOCODatasetV2,
    RRSISDDataset,
)
from segearth_r2.eval.eval import DataArguments as EvalDataArguments  # noqa: E402
from segearth_r2.model.language_model.llava_phi import SegEarthR2  # noqa: E402
from segearth_r2.utils import conversation as conversation_lib  # noqa: E402
from segearth_r2.utils.builder import load_pretrained_model  # noqa: E402
from segearth_r2.utils.constants import REFER_TOKEN_INDEX  # noqa: E402
from segearth_r2.utils.concept_public_grounding_train import (  # noqa: E402
    build_rrsisd_refaware_exclusion_only_human_value,
    retrieve_matched_public_grounding,
    split_matched_concepts_target_and_reference,
)

FATAL_MERGED = "FATAL: Probe requires merged HF model directory, not LoRA checkpoint."

FOCUS_CATEGORIES = (
    "airport",
    "vehicle",
    "ship",
    "harbor",
    "overpass",
    "chimney",
    "golffield",
    "basketballcourt",
    "road",
    "building",
)


def _fatal(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def assert_merged_hf_dir(path: str) -> None:
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(path):
        _fatal(f"{FATAL_MERGED}\n  Path is not a directory: {path}")
    if not os.path.isfile(os.path.join(path, "config.json")):
        _fatal(f"{FATAL_MERGED}\n  Missing config.json under: {path}")
    base = os.path.basename(path.rstrip(os.sep))
    if re.search(r"checkpoint-\d+$", base):
        _fatal(f"{FATAL_MERGED}\n  Path basename looks like a training checkpoint dir: {base}")
    has_adapter_cfg = os.path.isfile(os.path.join(path, "adapter_config.json"))
    has_adapter_weights = os.path.isfile(os.path.join(path, "adapter_model.safetensors")) or os.path.isfile(
        os.path.join(path, "adapter_model.bin")
    )
    has_full_weights = os.path.isfile(os.path.join(path, "model.safetensors")) or os.path.isfile(
        os.path.join(path, "pytorch_model.bin")
    )
    if has_adapter_cfg and has_adapter_weights and not has_full_weights:
        _fatal(
            f"{FATAL_MERGED}\n  Found adapter_config + adapter weights but no full merged weight file "
            f"(model.safetensors / pytorch_model.bin): {path}"
        )


def torch_dtype_from_arg(s: str) -> torch.dtype:
    s = (s or "bf16").lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    _fatal(f"Unknown --dtype {s}")


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    na, nb = a.norm(), b.norm()
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float((a @ b) / (na * nb))


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a.detach().float() - b.detach().float()) ** 2).mean())


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    n = a.norm()
    if n < 1e-12:
        return float("nan")
    return float((a - b).norm() / n)


def shallow_copy_sample(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            out[k] = v.clone()
        else:
            out[k] = v
    return out


def mask_iou(pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    a = pred_a.astype(bool).reshape(-1)
    b = pred_b.astype(bool).reshape(-1)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / union)


def pred_area(pred: np.ndarray) -> int:
    return int((pred.astype(bool)).sum())


@torch.no_grad()
def forward_eval_seg_tensors(
    model: SegEarthR2,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    images: torch.Tensor,
    images_clip: torch.Tensor,
    seg_info: list,
    token_refer_id: list,
    SEG_token_embedding_indices: torch.Tensor,
    labels: torch.Tensor,
    mask_num: list,
    dtype: torch.dtype,
    compute_device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Mirrors segearth_r2/eval/eval.py → model.eval_seg and llava_phi.eval_seg forward,
    returning (seg_llm_hidden_pre_proj, projected_seg_query, pred_mask_uint8).

    If compute_device is CUDA and the model currently lives on CPU, the model is
    temporarily moved to CUDA for this forward only (then moved back) so two
    merged checkpoints never simultaneously occupy VRAM.
    """
    model.eval()
    staging_cuda = compute_device.type == "cuda" and next(model.parameters()).device.type == "cpu"
    try:
        if staging_cuda:
            model.to(compute_device)
            if dtype == torch.float32:
                model.float()
            elif dtype == torch.bfloat16:
                model.to(dtype=torch.bfloat16)
            elif dtype == torch.float16:
                model.half()
        device = next(model.parameters()).device

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        images = images.to(device=device, dtype=torch.float32)
        images_clip = images_clip.to(device=device, dtype=torch.float32)
        labels = labels.to(device)
        SEG_token_embedding_indices = SEG_token_embedding_indices.to(device)
        token_refer_id = [t.to(device) for t in token_refer_id]

        image_features = model.get_vision_tower_feature(images)

        (
            input_ids2,
            attention_mask2,
            past_key_values,
            inputs_embeds,
            labels2,
            seg_indices2,
            image_features_indices,
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids,
            attention_mask,
            None,
            labels,
            images_clip,
            token_refer_id=token_refer_id,
            SEG_token_embedding_indices=SEG_token_embedding_indices,
        )

        if dtype in (torch.float16, torch.bfloat16):
            inputs_embeds = inputs_embeds.to(dtype=dtype)

        outputs = model.model(
            input_ids=input_ids2,
            attention_mask=attention_mask2,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        seg_hidden = model.get_SEG_embedding(hidden_states, seg_indices2)
        seg_query = model.SEG_token_projector(seg_hidden)

        mask_features, _t, multi_scale_features = model.pixel_decoder.forward_features(image_features)
        images_rep = [image.repeat((num, 1, 1, 1)) for image, num in zip(images, mask_num)]
        images_rep = [s[0] for image_repeat in images_rep for s in torch.split(image_repeat, 1, dim=0)]
        mask_num_t = torch.tensor(mask_num, device=mask_features.device)
        mask_features = torch.repeat_interleave(mask_features, repeats=mask_num_t, dim=0)
        multi_scale_features = [
            torch.repeat_interleave(feat, repeats=mask_num_t, dim=0) for feat in multi_scale_features
        ]
        mask_outputs = model.predictor(multi_scale_features, mask_features, None, None, seg_query)
        mask_pred_results = mask_outputs["pred_masks"]
        images_il = ImageList.from_tensors(images_rep, model.size_divisibility)
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images_il.tensor.shape[-2], images_il.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        mask_pred_result = mask_pred_results[0]
        pred = ((mask_pred_result.detach().float().cpu().numpy() > 0) * 255).astype(np.uint8)
        if pred.ndim > 2:
            pred = np.squeeze(pred)
        return seg_hidden.detach().cpu(), seg_query.detach().cpu(), pred
    finally:
        if staging_cuda:
            model.cpu()
            torch.cuda.empty_cache()


def first_seg_index(seg_indices_1d: torch.Tensor) -> int:
    """First [SEG] position in padded batch row (1D)."""
    hits = torch.where(seg_indices_1d == 1)[0]
    if hits.numel() == 0:
        return -1
    return int(hits[0].item())


def count_valid_tokens(input_ids_row: torch.Tensor, pad_id: int) -> int:
    return int((input_ids_row != pad_id).sum().item())


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


def focus_category_rank(category_name: str) -> int:
    c = (category_name or "").lower()
    if c in FOCUS_CATEGORIES:
        return FOCUS_CATEGORIES.index(c)
    return len(FOCUS_CATEGORIES)


def pearson_r(x: List[float], y: List[float]) -> Tuple[Optional[float], Optional[str]]:
    a = np.array(x, dtype=np.float64)
    b = np.array(y, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 2:
        return None, "insufficient_finite_pairs"
    a, b = a[m], b[m]
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return None, "zero_variance"
    return float(np.corrcoef(a, b)[0, 1]), None


def finite_mean(vals: List[Any]) -> float:
    vs = [float(x) for x in vals if x is not None and np.isfinite(float(x))]
    return float(np.mean(vs)) if vs else float("nan")


def finite_median(vals: List[Any]) -> float:
    vs = [float(x) for x in vals if x is not None and np.isfinite(float(x))]
    return float(np.median(vs)) if vs else float("nan")


def finite_mean_abs(vals: List[Any]) -> float:
    vs = [abs(float(x)) for x in vals if x is not None and np.isfinite(float(x))]
    return float(np.mean(vs)) if vs else float("nan")


def signal_survival_fields(llm_cos: float, q_cos: float) -> Tuple[Optional[float], Optional[float], bool]:
    """(signal_survival, attenuation_ratio, ratio_undefined). JSON null → None."""
    if not np.isfinite(llm_cos) or not np.isfinite(q_cos):
        return None, None, True
    if llm_cos < 0.99:
        den = 1.0 - float(llm_cos)
        if abs(den) < 1e-12:
            return None, None, True
        surv = (1.0 - float(q_cos)) / den
        return float(surv), float(1.0 - surv), False
    return None, None, True


def degradation_from_probe_b_iou(iou: float) -> Tuple[str, bool, bool]:
    if not np.isfinite(iou):
        return "stable", False, False
    if iou < 0.7:
        return "severe_degradation", True, False
    if iou < 0.85:
        return "moderate_degradation", False, True
    return "stable", False, False


def build_refaware_fields(
    ds: RRSISDDataset,
    *,
    instruction: str,
    category_name_str: Optional[str],
    lib_path: str,
) -> Dict[str, Any]:
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


def select_prior_hit_stratified(
    dataset: RRSISDDataset,
    lib_path: str,
    jsonl_path: str,
    sample_count: int,
    seed: int,
) -> Tuple[List[int], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    rng = random.Random(seed)
    candidates: List[Dict[str, Any]] = []
    for row in rows:
        idx = int(row.get("idx", -1))
        if idx < 0 or idx >= len(dataset):
            continue
        ref = dataset.reason_file[idx]
        ann = dataset.ann_dict[ref["ann_id"]]
        cid = ann.get("categories_id", ann.get("category_id"))
        category_name_str = dataset.category_dict.get(int(cid)) if cid is not None else None
        cat_lower = (str(category_name_str) if category_name_str else "").lower()
        instruction = (
            ref["sentences"][0].get("sent", "").strip()
            if ref.get("sentences") and "sent" in ref["sentences"][0]
            else ref["sentences"][0].get("raw", "").strip()
        )
        rf = build_refaware_fields(
            dataset,
            instruction=instruction,
            category_name_str=category_name_str,
            lib_path=lib_path,
        )
        prior_group, has_matched_prior = prior_group_and_has_matched_from_rf(rf)
        candidates.append(
            {
                "idx": idx,
                "ref_id": int(ref.get("ref_id", -1)),
                "prior_group": prior_group,
                "has_matched_prior": has_matched_prior,
                "category_name": str(category_name_str or ""),
                "category_lower": cat_lower,
                "focus_rank": focus_category_rank(str(category_name_str or "")),
            }
        )

    by_g: Dict[str, List[Dict[str, Any]]] = {g: [] for g in ("target_prior", "reference_only", "mixed", "no_prior")}
    for c in candidates:
        by_g[c["prior_group"]].append(c)

    for g in by_g:
        lst = by_g[g]
        if g == "reference_only":
            lst.sort(key=lambda x: (0 if x["category_lower"] == "overpass" else 1, x["focus_rank"], rng.random()))
        else:
            lst.sort(key=lambda x: (x["focus_rank"], rng.random()))

    quotas = {"target_prior": 30, "reference_only": 20, "mixed": 20, "no_prior": 10}
    used: set = set()
    selected: List[Dict[str, Any]] = []

    for g, quota in quotas.items():
        taken = 0
        for item in by_g[g]:
            if len(selected) >= sample_count:
                break
            if taken >= quota:
                break
            if item["idx"] in used:
                continue
            used.add(item["idx"])
            selected.append(item)
            taken += 1

    if len(selected) < sample_count:
        leftover = [c for c in candidates if c["idx"] not in used]
        leftover.sort(key=lambda x: (x["focus_rank"], rng.random()))
        for c in leftover:
            if len(selected) >= sample_count:
                break
            if c["idx"] in used:
                continue
            used.add(c["idx"])
            selected.append(c)

    notes: List[str] = []
    for g, q in quotas.items():
        avail = len(by_g[g])
        if avail < q:
            notes.append(f"shortfall: group {g} requested {q} but jsonl only has {avail} after enrichment")

    if len(selected) < sample_count:
        rng2 = random.Random(seed + 1)
        pool = [i for i in range(len(dataset)) if i not in used]
        rng2.shuffle(pool)
        for i in pool:
            if len(selected) >= sample_count:
                break
            ref = dataset.reason_file[i]
            ann = dataset.ann_dict[ref["ann_id"]]
            cid = ann.get("categories_id", ann.get("category_id"))
            category_name_str = dataset.category_dict.get(int(cid)) if cid is not None else None
            instruction = (
                ref["sentences"][0].get("sent", "").strip()
                if ref.get("sentences") and "sent" in ref["sentences"][0]
                else ref["sentences"][0].get("raw", "").strip()
            )
            rf = build_refaware_fields(
                dataset,
                instruction=instruction,
                category_name_str=category_name_str,
                lib_path=lib_path,
            )
            prior_group, has_matched_prior = prior_group_and_has_matched_from_rf(rf)
            used.add(i)
            selected.append(
                {
                    "idx": i,
                    "ref_id": int(ref.get("ref_id", -1)),
                    "prior_group": prior_group,
                    "has_matched_prior": has_matched_prior,
                    "category_name": str(category_name_str or ""),
                    "category_lower": (str(category_name_str or "")).lower(),
                    "focus_rank": focus_category_rank(str(category_name_str or "")),
                    "from_fallback_random": True,
                }
            )
        fb = sum(1 for s in selected if s.get("from_fallback_random"))
        if fb:
            notes.append(f"filled {fb} samples via random fallback (not in prior-hit jsonl)")

    pg_counts = Counter(s["prior_group"] for s in selected)
    cat_counts = Counter((s["category_name"] or "").lower() for s in selected if s["category_name"])
    focus_hits = [fc for fc in FOCUS_CATEGORIES if cat_counts.get(fc, 0) > 0]

    meta: Dict[str, Any] = {
        "prior_hit_jsonl": os.path.abspath(jsonl_path),
        "stratified": True,
        "quotas_requested": quotas,
        "jsonl_candidates_total": len(candidates),
        "actual_prior_group_counts": dict(pg_counts),
        "category_counts_in_selection": dict(cat_counts),
        "focus_categories_represented": focus_hits,
        "sampling_notes": notes,
    }
    return [s["idx"] for s in selected[:sample_count]], meta


def select_samples(
    *,
    dataset: RRSISDDataset,
    sample_source: str,
    sample_count: int,
    fixed_ref_ids: Optional[List[int]],
    jsonl_path: Optional[str],
    lib_path: str,
) -> Tuple[List[int], Optional[Dict[str, Any]]]:
    if sample_source == "fixed_ids":
        if not fixed_ref_ids:
            _fatal("--sample-source fixed_ids requires --fixed-ref-ids")
        idxs = []
        for rid in fixed_ref_ids:
            found = None
            for i, r in enumerate(dataset.reason_file):
                if int(r.get("ref_id", -1)) == int(rid):
                    found = i
                    break
            if found is None:
                _fatal(f"ref_id {rid} not found in RRSISDDataset split")
            idxs.append(found)
        return idxs[:sample_count], None

    if sample_source == "random":
        n = len(dataset)
        rng = random.Random(42)
        return [rng.randint(0, n - 1) for _ in range(sample_count)], {"stratified": False, "sample_source": "random"}

    # prior_hit
    if jsonl_path and os.path.isfile(jsonl_path) and sample_count >= 30:
        return select_prior_hit_stratified(dataset, lib_path, jsonl_path, sample_count, seed=42)

    if jsonl_path and os.path.isfile(jsonl_path):
        rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        hit_prior = None
        ref_only = None
        for row in rows:
            mc = row.get("matched_concepts") or []
            if hit_prior is None and isinstance(mc, list) and len(mc) > 0:
                cat = str(row.get("category_name") or "").lower()
                if any(x in cat for x in ("airport", "vehicle", "ship", "harbor")):
                    hit_prior = row
            expr = str(row.get("expression") or "").lower()
            catn = str(row.get("category_name") or "").lower()
            if ref_only is None and "overpass" in catn and "vehicle" in expr:
                ref_only = row
        picks = []
        if hit_prior is not None:
            picks.append(int(hit_prior["idx"]))
        if ref_only is not None:
            picks.append(int(ref_only["idx"]))
        out: List[int] = []
        for p in picks:
            if p not in out:
                out.append(p)
        if len(out) < sample_count:
            for row in rows:
                i = int(row.get("idx", -1))
                if i >= 0 and i not in out:
                    out.append(i)
                if len(out) >= sample_count:
                    break
        return out[:sample_count], {"stratified": False, "sample_source": "prior_hit_legacy_jsonl"}

    rng = random.Random(42)
    n = len(dataset)
    return [rng.randint(0, n - 1) for _ in range(sample_count)], {
        "stratified": False,
        "sample_source": "prior_hit_missing_jsonl_random_fallback",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-data-path", default="/home/wangchengjun/huangziyi/data/RRSISD")
    parser.add_argument("--baseline-model-path", required=True)
    parser.add_argument("--refaware-model-path", required=True)
    parser.add_argument(
        "--semantic-library",
        default=os.path.join(REPO_ROOT, "configs/concept_public_semantic_library_v2.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(REPO_ROOT, "outputs/source/refaware_hidden_probe"),
    )
    parser.add_argument("--sample-count", type=int, default=2)
    parser.add_argument("--sample-source", default="prior_hit", choices=("prior_hit", "random", "fixed_ids"))
    parser.add_argument("--fixed-ref-ids", default=None, help="Comma-separated ref_id list")
    parser.add_argument(
        "--prior-hit-jsonl",
        default=os.path.join(
            REPO_ROOT, "outputs/source/public_semantic_v2_prior_hit_analysis/per_sample_prior_hit_diff.jsonl"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--save-mask-diff", action="store_true")
    parser.add_argument("--split", default="test")
    parser.add_argument("--mask-config", default="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
    parser.add_argument(
        "--vision-tower",
        default="/home/wangchengjun/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384",
    )
    parser.add_argument(
        "--vision-tower-mask",
        default="/home/wangchengjun/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl",
    )
    args = parser.parse_args()

    assert_merged_hf_dir(args.baseline_model_path)
    assert_merged_hf_dir(args.refaware_model_path)

    lib_path = args.semantic_library
    if not os.path.isabs(lib_path):
        lib_path = os.path.join(REPO_ROOT, lib_path)
    if not os.path.isfile(lib_path):
        _fatal(f"Semantic library not found: {lib_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    dtype = torch_dtype_from_arg(args.dtype)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    mconf = args.mask_config
    if not os.path.isabs(mconf):
        mconf = os.path.join(REPO_ROOT, mconf)
    data_args = replace(
        EvalDataArguments(),
        base_data_path=args.base_data_path,
        dataset_name="rrsisd",
        split=args.split,
        vision_tower=args.vision_tower,
        vision_tower_mask=args.vision_tower_mask,
        mask_config=mconf,
    )
    setattr(data_args, "concept_public_semantic_library", None)
    setattr(data_args, "concept_refaware_prior", False)
    setattr(data_args, "concept_match_strict", False)
    setattr(data_args, "debug_concept_match_strict", False)
    setattr(data_args, "debug_concept_refaware_prior", False)
    if not os.path.isdir(data_args.base_data_path):
        _fatal(f"base_data_path not found: {data_args.base_data_path}")

    conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version]

    # --- Load baseline (for Probe B) ---
    tok_b, model_b, _, _ctx = load_pretrained_model(
        args.baseline_model_path,
        model_args=data_args,
        mask_config=data_args.mask_config,
        device="cpu",
    )
    tok_r, model_r, _, _ctx2 = load_pretrained_model(
        args.refaware_model_path,
        model_args=data_args,
        mask_config=data_args.mask_config,
        device="cpu",
    )
    # Keep both checkpoints on CPU; each forward stages one model to GPU temporarily (VRAM).
    model_b = model_b.cpu()
    model_r = model_r.cpu()
    if dtype == torch.float32:
        model_b = model_b.float()
        model_r = model_r.float()

    if len(tok_b) != len(tok_r):
        print("[WARN] Tokenizer vocab differs between baseline and refaware models.", file=sys.stderr)

    tokenizer = tok_r
    clip_processor = SiglipImageProcessor.from_pretrained(data_args.vision_tower)
    collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer, clip_image_processor=clip_processor)

    ds_raw = RRSISDDataset(
        base_data_path=data_args.base_data_path,
        tokenizer=tokenizer,
        data_args=data_args,
        split=data_args.split,
    )

    fixed_ids = None
    if args.fixed_ref_ids:
        fixed_ids = [int(x.strip()) for x in args.fixed_ref_ids.split(",") if x.strip()]

    idx_list, sampling_meta = select_samples(
        dataset=ds_raw,
        sample_source=args.sample_source,
        sample_count=args.sample_count,
        fixed_ref_ids=fixed_ids,
        jsonl_path=args.prior_hit_jsonl,
        lib_path=lib_path,
    )

    rows_csv: List[Dict[str, Any]] = []
    stab_cosines: List[float] = []
    pos_mismatch_ct = 0
    hook_stability_failed_count = 0
    probe_a_llm_cos: List[float] = []
    probe_a_q_cos: List[float] = []
    probe_a_iou: List[float] = []
    probe_b_llm_cos: List[float] = []
    probe_b_q_cos: List[float] = []
    probe_b_iou: List[float] = []
    failure_samples: List[Dict[str, Any]] = []

    hook_failed = False
    projected_query_confirmed = True
    hook_module_llm = "SegEarthR2.model (HF backbone) → outputs.last_hidden_state"
    hook_module_q = "SegEarthR2.SEG_token_projector(Linear) applied to get_SEG_embedding(last_hidden_state, SEG_token_embedding_indices); same tensor fed to predictor() as in llava_phi.eval_seg line ~825"

    llm_shape_s = ""
    q_shape_s = ""

    for row_idx, idx in enumerate(idx_list):
        raw_item = shallow_copy_sample(ds_raw[idx])
        ref = ds_raw.reason_file[idx]
        ref_id = int(ref.get("ref_id", -1))
        instruction = (
            ref["sentences"][0].get("sent", "").strip()
            if ref.get("sentences") and "sent" in ref["sentences"][0]
            else ref["sentences"][0].get("raw", "").strip()
        )
        ann = ds_raw.ann_dict[ref["ann_id"]]
        cid = ann.get("categories_id", ann.get("category_id"))
        category_name_str = ds_raw.category_dict.get(int(cid)) if cid is not None else None

        rf = build_refaware_fields(
            ds_raw,
            instruction=instruction,
            category_name_str=category_name_str,
            lib_path=lib_path,
        )
        prior_group, has_matched_prior = prior_group_and_has_matched_from_rf(rf)

        raw_batch = collator([shallow_copy_sample(raw_item)])
        ref_item = shallow_copy_sample(ds_raw[idx])
        ref_item["input_ids"] = rf["input_ids"].clone()
        ref_item["labels"] = rf["labels"].clone()
        ref_item["SEG_token_embedding_indices"] = rf["SEG_token_embedding_indices"].clone()
        ref_item["refer_embedding_indices"] = rf["refer_embedding_indices"].clone()
        refaware_batch = collator([ref_item])

        def to_dev(b):
            out = {}
            for k, v in b.items():
                if torch.is_tensor(v):
                    out[k] = v.to(device)
                else:
                    out[k] = v
            return out

        rb = to_dev(raw_batch)
        ab = to_dev(refaware_batch)

        raw_seg = first_seg_index(raw_batch["SEG_token_embedding_indices"][0])
        ref_seg = first_seg_index(refaware_batch["SEG_token_embedding_indices"][0])
        if raw_seg < 0 or ref_seg < 0:
            _fatal("Could not locate [SEG] in input_ids / SEG_token_embedding_indices for this sample.")

        pad_id = tokenizer.pad_token_id
        raw_tok_n = count_valid_tokens(raw_batch["input_ids"][0], pad_id)
        ref_tok_n = count_valid_tokens(refaware_batch["input_ids"][0], pad_id)
        prior_token_count = max(0, ref_tok_n - raw_tok_n)
        seg_delta = ref_seg - raw_seg
        position_mismatch_risk = bool(prior_token_count > 0 and seg_delta <= 0)

        refer_raw = torch.where(raw_item["refer_embedding_indices"] == 1)[0]
        refer_ref = torch.where(rf["refer_embedding_indices"] == 1)[0]
        token_refer_confirmed = refer_raw.numel() > 0 and refer_ref.numel() > 0

        # --- Hook stability: same refaware model, same raw-eval batch, two forwards ---
        h1, q1, p1 = forward_eval_seg_tensors(
            model_r,
            input_ids=rb["input_ids"],
            attention_mask=rb["attention_mask"],
            images=rb["images"],
            images_clip=rb["images_clip"],
            seg_info=rb["seg_info"],
            token_refer_id=[x for x in rb["token_refer_id"]],
            SEG_token_embedding_indices=rb["SEG_token_embedding_indices"],
            labels=rb["labels"],
            mask_num=rb["mask_num"],
            dtype=dtype,
            compute_device=device,
        )
        h2, q2, p2 = forward_eval_seg_tensors(
            model_r,
            input_ids=rb["input_ids"],
            attention_mask=rb["attention_mask"],
            images=rb["images"],
            images_clip=rb["images_clip"],
            seg_info=rb["seg_info"],
            token_refer_id=[x for x in rb["token_refer_id"]],
            SEG_token_embedding_indices=rb["SEG_token_embedding_indices"],
            labels=rb["labels"],
            mask_num=rb["mask_num"],
            dtype=dtype,
            compute_device=device,
        )
        stab_c = cosine_sim(h1, h2)
        stab_m = mse(h1, h2)
        stab_r = relative_l2(h1, h2)
        stab_cosines.append(stab_c)
        stability_failed = (not np.isfinite(stab_c)) or stab_c < 0.999 or (h1.abs().sum() == 0)
        if stability_failed:
            hook_stability_failed_count += 1
            hook_failed = True

        llm_shape_s = str(tuple(h1.shape))
        q_shape_s = str(tuple(q1.shape))
        if h1.ndim != 3 or h1.shape[1] != 1:
            hook_failed = True
        if q1.ndim != 3 or q1.shape[1] != 1:
            hook_failed = True

        row_pq_confirmed = (
            h1.ndim == 3
            and h1.shape[1] == 1
            and q1.ndim == 3
            and q1.shape[1] == 1
            and (not stability_failed)
        )
        projected_query_confirmed = projected_query_confirmed and row_pq_confirmed

        # --- Probe A: same refaware model, raw vs refaware prompt ---
        h_raw_a, q_raw_a, pred_raw_a = forward_eval_seg_tensors(
            model_r,
            input_ids=rb["input_ids"],
            attention_mask=rb["attention_mask"],
            images=rb["images"],
            images_clip=rb["images_clip"],
            seg_info=rb["seg_info"],
            token_refer_id=[x for x in rb["token_refer_id"]],
            SEG_token_embedding_indices=rb["SEG_token_embedding_indices"],
            labels=rb["labels"],
            mask_num=rb["mask_num"],
            dtype=dtype,
            compute_device=device,
        )
        h_ref_a, q_ref_a, pred_ref_a = forward_eval_seg_tensors(
            model_r,
            input_ids=ab["input_ids"],
            attention_mask=ab["attention_mask"],
            images=ab["images"],
            images_clip=ab["images_clip"],
            seg_info=ab["seg_info"],
            token_refer_id=[x for x in ab["token_refer_id"]],
            SEG_token_embedding_indices=ab["SEG_token_embedding_indices"],
            labels=ab["labels"],
            mask_num=ab["mask_num"],
            dtype=dtype,
            compute_device=device,
        )
        pa_llm_c = cosine_sim(h_raw_a, h_ref_a)
        pa_llm_m = mse(h_raw_a, h_ref_a)
        pa_llm_r = relative_l2(h_raw_a, h_ref_a)
        pa_q_c = cosine_sim(q_raw_a, q_ref_a)
        pa_q_m = mse(q_raw_a, q_ref_a)
        pa_q_r = relative_l2(q_raw_a, q_ref_a)
        pa_iou = mask_iou(pred_raw_a, pred_ref_a)
        pa_area_d = pred_area(pred_ref_a) - pred_area(pred_raw_a)
        pred_area_raw = pred_area(pred_raw_a)
        pred_area_refaware = pred_area(pred_ref_a)
        probe_a_llm_cos.append(float(pa_llm_c))
        probe_a_q_cos.append(float(pa_q_c))
        probe_a_iou.append(float(pa_iou))

        if position_mismatch_risk:
            pos_mismatch_ct += 1

        # --- Probe B: same raw prompt, baseline vs refaware weights ---
        h_raw_b, q_raw_b, pred_raw_b = forward_eval_seg_tensors(
            model_b,
            input_ids=rb["input_ids"],
            attention_mask=rb["attention_mask"],
            images=rb["images"],
            images_clip=rb["images_clip"],
            seg_info=rb["seg_info"],
            token_refer_id=[x for x in rb["token_refer_id"]],
            SEG_token_embedding_indices=rb["SEG_token_embedding_indices"],
            labels=rb["labels"],
            mask_num=rb["mask_num"],
            dtype=dtype,
            compute_device=device,
        )
        pb_llm_c = cosine_sim(h_raw_b, h_raw_a)
        pb_llm_m = mse(h_raw_b, h_raw_a)
        pb_llm_r = relative_l2(h_raw_b, h_raw_a)
        pb_q_c = cosine_sim(q_raw_b, q_raw_a)
        pb_q_m = mse(q_raw_b, q_raw_a)
        pb_q_r = relative_l2(q_raw_b, q_raw_a)
        pb_iou = mask_iou(pred_raw_b, pred_raw_a)
        pb_area_d = pred_area(pred_raw_a) - pred_area(pred_raw_b)
        probe_b_llm_cos.append(float(pb_llm_c))
        probe_b_q_cos.append(float(pb_q_c))
        probe_b_iou.append(float(pb_iou))

        sig_s, sig_a, ratio_u = signal_survival_fields(pa_llm_c, pa_q_c)
        deg_level, sev_b, mod_b = degradation_from_probe_b_iou(pb_iou)

        fail_reasons: List[str] = []
        if position_mismatch_risk:
            fail_reasons.append("position_mismatch_risk")
        if stability_failed:
            fail_reasons.append("hook_stability_cosine_below_0.999_or_nonfinite_or_zero_seg_hidden")
        if not row_pq_confirmed:
            fail_reasons.append("projected_query_confirmed_false")
        if not np.isfinite(pa_llm_c) or not np.isfinite(pa_q_c) or not np.isfinite(pa_iou):
            fail_reasons.append("probe_a_nan")
        if not np.isfinite(pb_llm_c) or not np.isfinite(pb_q_c) or not np.isfinite(pb_iou):
            fail_reasons.append("probe_b_nan")
        if fail_reasons:
            failure_samples.append({"ref_id": ref_id, "idx": idx, "reasons": fail_reasons})

        row = {
            "ref_id": ref_id,
            "idx": idx,
            "category_name": category_name_str or "",
            "expression": instruction[:500],
            "matched_concepts": "|".join(rf["matched_concepts"]),
            "target_concepts": "|".join(rf["target_concepts"]),
            "reference_concepts": "|".join(rf["reference_concepts"]),
            "prior_group": prior_group,
            "has_matched_prior": has_matched_prior,
            "refer_placeholder_indices_raw": ",".join(
                str(int(x)) for x in torch.where(raw_item["refer_embedding_indices"] == 1)[0].tolist()
            ),
            "refer_placeholder_indices_refaware": ",".join(
                str(int(x)) for x in torch.where(rf["refer_embedding_indices"] == 1)[0].tolist()
            ),
            "raw_seg_index": raw_seg,
            "refaware_seg_index": ref_seg,
            "seg_index_delta": seg_delta,
            "prior_token_count": prior_token_count,
            "position_mismatch_risk": position_mismatch_risk,
            "token_refer_id_position_confirmed": token_refer_confirmed,
            "hook_stability_cosine": stab_c,
            "hook_stability_mse": stab_m,
            "hook_stability_relative_l2": stab_r,
            "llm_seg_hidden_shape": llm_shape_s,
            "projected_query_shape": q_shape_s,
            "projected_query_confirmed": row_pq_confirmed,
            "llm_seg_cosine_raw_vs_refaware": pa_llm_c,
            "llm_seg_mse_raw_vs_refaware": pa_llm_m,
            "llm_seg_relative_l2_raw_vs_refaware": pa_llm_r,
            "projected_query_cosine_raw_vs_refaware": pa_q_c,
            "projected_query_mse_raw_vs_refaware": pa_q_m,
            "projected_query_relative_l2_raw_vs_refaware": pa_q_r,
            "mask_iou_raw_vs_refaware": pa_iou,
            "pred_area_raw": pred_area_raw,
            "pred_area_refaware": pred_area_refaware,
            "pred_area_delta_raw_vs_refaware": pa_area_d,
            "signal_survival": sig_s,
            "attenuation_ratio": sig_a,
            "ratio_undefined": ratio_u,
            "llm_seg_cosine_baseline_vs_refaware_model": pb_llm_c,
            "llm_seg_mse_baseline_vs_refaware_model": pb_llm_m,
            "llm_seg_relative_l2_baseline_vs_refaware_model": pb_llm_r,
            "projected_query_cosine_baseline_vs_refaware_model": pb_q_c,
            "projected_query_mse_baseline_vs_refaware_model": pb_q_m,
            "projected_query_relative_l2_baseline_vs_refaware_model": pb_q_r,
            "mask_iou_baseline_vs_refaware_model": pb_iou,
            "pred_area_delta_baseline_vs_refaware_model": pb_area_d,
            "degradation_level": deg_level,
            "severe_degradation": sev_b,
            "moderate_degradation": mod_b,
        }
        rows_csv.append(row)

        if args.save_mask_diff:
            d = np.abs(pred_raw_a.astype(np.int16) - pred_ref_a.astype(np.int16))
            out_png = os.path.join(args.output_dir, f"mask_diff_sample_{row_idx}_ref{ref_id}.npy")
            np.save(out_png, d)


    def mean(xs: List[float]) -> float:
        xs = [x for x in xs if np.isfinite(x)]
        return float(sum(xs) / len(xs)) if xs else float("nan")

    n = len(rows_csv)
    ratio_undefined_count = sum(1 for r in rows_csv if r.get("ratio_undefined"))
    severe_degradation_count = sum(1 for r in rows_csv if r.get("severe_degradation"))
    moderate_degradation_count = sum(1 for r in rows_csv if r.get("moderate_degradation"))
    stable_count = sum(1 for r in rows_csv if r.get("degradation_level") == "stable")
    surv_defined = [
        float(r["signal_survival"])
        for r in rows_csv
        if (not r.get("ratio_undefined")) and r.get("signal_survival") is not None and np.isfinite(r["signal_survival"])
    ]
    probe_a_mean_signal_survival = finite_mean(surv_defined)
    probe_a_median_signal_survival = finite_median(surv_defined)

    degradation_by_category: Dict[str, Dict[str, int]] = {}
    for r in rows_csv:
        ck = (r.get("category_name") or "").strip().lower() or "_unknown"
        degradation_by_category.setdefault(
            ck, {"severe_degradation": 0, "moderate_degradation": 0, "stable": 0}
        )
        if r.get("severe_degradation"):
            degradation_by_category[ck]["severe_degradation"] += 1
        elif r.get("moderate_degradation"):
            degradation_by_category[ck]["moderate_degradation"] += 1
        else:
            degradation_by_category[ck]["stable"] += 1

    degradation_by_prior_group: Dict[str, Dict[str, int]] = {}
    for r in rows_csv:
        gk = str(r.get("prior_group") or "unknown")
        degradation_by_prior_group.setdefault(
            gk, {"severe_degradation": 0, "moderate_degradation": 0, "stable": 0}
        )
        if r.get("severe_degradation"):
            degradation_by_prior_group[gk]["severe_degradation"] += 1
        elif r.get("moderate_degradation"):
            degradation_by_prior_group[gk]["moderate_degradation"] += 1
        else:
            degradation_by_prior_group[gk]["stable"] += 1

    def rich_stats(sub: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not sub:
            return {
                "sample_count": 0,
                "probe_a_mean_llm_seg_cosine": float("nan"),
                "probe_a_mean_projected_query_cosine": float("nan"),
                "probe_a_mean_signal_survival": float("nan"),
                "probe_a_median_signal_survival": float("nan"),
                "ratio_undefined_count": 0,
                "probe_a_mean_mask_iou": float("nan"),
                "probe_a_mean_abs_pred_area_delta_raw_vs_refaware": float("nan"),
                "probe_b_mean_llm_seg_cosine": float("nan"),
                "probe_b_mean_projected_query_cosine": float("nan"),
                "probe_b_mean_mask_iou": float("nan"),
                "severe_degradation_count": 0,
                "moderate_degradation_count": 0,
                "stable_count": 0,
                "probe_b_mean_abs_pred_area_delta_baseline_vs_refaware_model": float("nan"),
            }
        survs = [
            float(r["signal_survival"])
            for r in sub
            if (not r.get("ratio_undefined"))
            and r.get("signal_survival") is not None
            and np.isfinite(r["signal_survival"])
        ]
        return {
            "sample_count": len(sub),
            "probe_a_mean_llm_seg_cosine": finite_mean([r["llm_seg_cosine_raw_vs_refaware"] for r in sub]),
            "probe_a_mean_projected_query_cosine": finite_mean(
                [r["projected_query_cosine_raw_vs_refaware"] for r in sub]
            ),
            "probe_a_mean_signal_survival": finite_mean(survs),
            "probe_a_median_signal_survival": finite_median(survs),
            "ratio_undefined_count": sum(1 for r in sub if r.get("ratio_undefined")),
            "probe_a_mean_mask_iou": finite_mean([r["mask_iou_raw_vs_refaware"] for r in sub]),
            "probe_a_mean_abs_pred_area_delta_raw_vs_refaware": finite_mean_abs(
                [r["pred_area_delta_raw_vs_refaware"] for r in sub]
            ),
            "probe_b_mean_llm_seg_cosine": finite_mean([r["llm_seg_cosine_baseline_vs_refaware_model"] for r in sub]),
            "probe_b_mean_projected_query_cosine": finite_mean(
                [r["projected_query_cosine_baseline_vs_refaware_model"] for r in sub]
            ),
            "probe_b_mean_mask_iou": finite_mean([r["mask_iou_baseline_vs_refaware_model"] for r in sub]),
            "severe_degradation_count": sum(1 for r in sub if r.get("severe_degradation")),
            "moderate_degradation_count": sum(1 for r in sub if r.get("moderate_degradation")),
            "stable_count": sum(1 for r in sub if r.get("degradation_level") == "stable"),
            "probe_b_mean_abs_pred_area_delta_baseline_vs_refaware_model": finite_mean_abs(
                [r["pred_area_delta_baseline_vs_refaware_model"] for r in sub]
            ),
        }

    prior_group_summary = {
        g: rich_stats([r for r in rows_csv if r.get("prior_group") == g])
        for g in ("target_prior", "reference_only", "mixed", "no_prior")
    }
    cats_sorted = sorted(set((r.get("category_name") or "").strip().lower() for r in rows_csv if (r.get("category_name") or "").strip()))
    category_summary = {
        c: rich_stats([r for r in rows_csv if (r.get("category_name") or "").strip().lower() == c]) for c in cats_sorted
    }

    def group_csv_row(gtype: str, gname: str, sub: List[Dict[str, Any]]) -> Dict[str, Any]:
        st = rich_stats(sub)
        return {
            "group_type": gtype,
            "group_name": gname,
            "sample_count": st["sample_count"],
            "probe_a_mean_llm_seg_cosine": st["probe_a_mean_llm_seg_cosine"],
            "probe_a_mean_projected_query_cosine": st["probe_a_mean_projected_query_cosine"],
            "probe_a_mean_signal_survival": st["probe_a_mean_signal_survival"],
            "probe_a_median_signal_survival": st["probe_a_median_signal_survival"],
            "ratio_undefined_count": st["ratio_undefined_count"],
            "probe_a_mean_mask_iou": st["probe_a_mean_mask_iou"],
            "probe_b_mean_mask_iou": st["probe_b_mean_mask_iou"],
            "severe_degradation_count": st["severe_degradation_count"],
            "moderate_degradation_count": st["moderate_degradation_count"],
            "stable_count": st["stable_count"],
        }

    group_summary_rows: List[Dict[str, Any]] = [group_csv_row("all", "all", rows_csv)]
    for g in ("target_prior", "reference_only", "mixed", "no_prior"):
        group_summary_rows.append(group_csv_row("prior_group", g, [r for r in rows_csv if r.get("prior_group") == g]))
    for ph in (True, False):
        group_summary_rows.append(
            group_csv_row("prior_hit", str(ph), [r for r in rows_csv if bool(r.get("has_matched_prior")) == ph])
        )
    for c in cats_sorted:
        group_summary_rows.append(
            group_csv_row(
                "category_name",
                c,
                [r for r in rows_csv if (r.get("category_name") or "").strip().lower() == c],
            )
        )

    xs1 = [1 - float(r["projected_query_cosine_raw_vs_refaware"]) for r in rows_csv]
    ys1 = [1 - float(r["mask_iou_raw_vs_refaware"]) for r in rows_csv]
    c1, c1r = pearson_r(xs1, ys1)

    rows_sig = [
        r
        for r in rows_csv
        if (not r.get("ratio_undefined"))
        and r.get("signal_survival") is not None
        and np.isfinite(r["signal_survival"])
        and np.isfinite(r["mask_iou_raw_vs_refaware"])
    ]
    c2, c2r = pearson_r(
        [float(r["signal_survival"]) for r in rows_sig],
        [1 - float(r["mask_iou_raw_vs_refaware"]) for r in rows_sig],
    )

    c3, c3r = pearson_r(
        [abs(float(r["pred_area_delta_raw_vs_refaware"])) for r in rows_csv],
        [1 - float(r["projected_query_cosine_raw_vs_refaware"]) for r in rows_csv],
    )

    c4, c4r = pearson_r(
        [1 - float(r["projected_query_cosine_baseline_vs_refaware_model"]) for r in rows_csv],
        [1 - float(r["mask_iou_baseline_vs_refaware_model"]) for r in rows_csv],
    )

    correlations = {
        "corr_one_minus_projected_query_cosine_vs_one_minus_mask_iou_probe_a": c1,
        "corr_one_minus_projected_query_cosine_vs_one_minus_mask_iou_probe_a_reason": c1r,
        "corr_signal_survival_vs_one_minus_mask_iou_probe_a_ratio_defined_only": c2,
        "corr_signal_survival_vs_one_minus_mask_iou_probe_a_reason": c2r,
        "corr_abs_pred_area_delta_vs_one_minus_projected_query_cosine_probe_a": c3,
        "corr_abs_pred_area_delta_vs_one_minus_projected_query_cosine_probe_a_reason": c3r,
        "corr_one_minus_projected_query_cosine_vs_one_minus_mask_iou_probe_b": c4,
        "corr_one_minus_projected_query_cosine_vs_one_minus_mask_iou_probe_b_reason": c4r,
    }

    top_degradation_samples = sorted(rows_csv, key=lambda r: float(r["mask_iou_baseline_vs_refaware_model"]))[:25]
    airport_severe = [
        {"ref_id": r["ref_id"], "idx": r["idx"], "mask_iou_baseline_vs_refaware_model": r["mask_iou_baseline_vs_refaware_model"]}
        for r in rows_csv
        if (r.get("category_name") or "").strip().lower() == "airport" and r.get("severe_degradation")
    ]

    mean_pa_llm = mean(probe_a_llm_cos)
    mean_pa_q = mean(probe_a_q_cos)
    mean_pa_iou = mean(probe_a_iou)
    mean_pb_iou = mean(probe_b_iou)
    mod_rate = moderate_degradation_count / max(n, 1)

    verdict_tags: List[str] = []
    hook_disaster = (
        hook_failed
        or pos_mismatch_ct > 0
        or hook_stability_failed_count > 0
        or (not projected_query_confirmed)
    )
    if hook_disaster:
        verdict_tags.append("hook_failed")
    if not hook_disaster:
        if (
            np.isfinite(mean_pa_llm)
            and mean_pa_llm < 0.98
            and np.isfinite(mean_pa_q)
            and mean_pa_q > 0.985
            and np.isfinite(probe_a_mean_signal_survival)
            and probe_a_mean_signal_survival < 0.3
        ):
            verdict_tags.append("prompt_signal_blocked_by_projector")
        if (np.isfinite(mean_pa_q) and mean_pa_q < 0.97) or (np.isfinite(mean_pa_iou) and mean_pa_iou < 0.95):
            verdict_tags.append("prompt_affects_mask")
        if severe_degradation_count > 0 or mod_rate > 0.15:
            verdict_tags.append("degradation_risk")
        verdict_tags.append("mechanism_plausible_but_needs_control")

    dtype_str = str(args.dtype).lower()
    dtype_numeric_note = (
        "fp32: 余弦/IoU 等标量为 float32 累加，数值可复现性较好。"
        if dtype_str in ("fp32", "float32")
        else "bf16/fp16: 中间激活与部分归约为低精度，余弦与 IoU 可能出现微小漂移；解释阈值时应留更大容差。"
    )

    summary = {
        "sample_count": n,
        "dtype": dtype_str,
        "dtype_numeric_note": dtype_numeric_note,
        "baseline_model_path": os.path.abspath(args.baseline_model_path),
        "refaware_model_path": os.path.abspath(args.refaware_model_path),
        "sampling_metadata": sampling_meta,
        "hook_module_llm_last_layer": hook_module_llm,
        "hook_module_projected_query": hook_module_q,
        "llm_seg_hidden_shape": llm_shape_s,
        "projected_query_shape": q_shape_s,
        "projected_query_confirmed": projected_query_confirmed,
        "hook_stability_mean_cosine": mean(stab_cosines),
        "hook_stability_failed": hook_stability_failed_count > 0,
        "hook_stability_failed_count": hook_stability_failed_count,
        "position_mismatch_count": pos_mismatch_ct,
        "ratio_undefined_count": ratio_undefined_count,
        "probe_a_mean_llm_seg_cosine": mean_pa_llm,
        "probe_a_mean_projected_query_cosine": mean_pa_q,
        "probe_a_mean_signal_survival": probe_a_mean_signal_survival,
        "probe_a_median_signal_survival": probe_a_median_signal_survival,
        "probe_a_mean_mask_iou": mean_pa_iou,
        "probe_b_mean_llm_seg_cosine": mean(probe_b_llm_cos),
        "probe_b_mean_projected_query_cosine": mean(probe_b_q_cos),
        "probe_b_mean_mask_iou": mean_pb_iou,
        "severe_degradation_count": severe_degradation_count,
        "moderate_degradation_count": moderate_degradation_count,
        "stable_count": stable_count,
        "severe_degradation_rate": float(severe_degradation_count / max(n, 1)),
        "moderate_degradation_rate": float(moderate_degradation_count / max(n, 1)),
        "degradation_by_category": degradation_by_category,
        "degradation_by_prior_group": degradation_by_prior_group,
        "prior_group_summary": prior_group_summary,
        "category_summary": category_summary,
        "correlations": correlations,
        "top_degradation_samples": [
            {
                "ref_id": r["ref_id"],
                "idx": r["idx"],
                "category_name": r.get("category_name"),
                "prior_group": r.get("prior_group"),
                "mask_iou_baseline_vs_refaware_model": r["mask_iou_baseline_vs_refaware_model"],
            }
            for r in top_degradation_samples
        ],
        "airport_severe_degradation_samples": airport_severe,
        "failure_samples": failure_samples,
        "verdict": verdict_tags,
        "recommendation_next_steps": (
            "Probe B 为 baseline_7w merged 对 refaware_28w merged，训练步数与配方均不同，不能将 mask 差异归因于 refaware 策略本身。"
            " 若要做因果对比，应优先跑 baseline_28w merged、v2_28w merged 与 refaware_28w merged 的三方控制实验，再解读 Probe B 类指标。"
        ),
        "indices_used": idx_list,
    }

    csv_path = os.path.join(args.output_dir, "hidden_probe_per_sample.csv")
    if rows_csv:
        def _csv_row(r: Dict[str, Any]) -> Dict[str, Any]:
            o: Dict[str, Any] = {}
            for k, v in r.items():
                if v is None:
                    o[k] = ""
                else:
                    o[k] = v
            return o

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
            w.writeheader()
            for r in rows_csv:
                w.writerow(_csv_row(r))

    gcsv = os.path.join(args.output_dir, "hidden_probe_group_summary.csv")
    if group_summary_rows:
        with open(gcsv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(group_summary_rows[0].keys()))
            w.writeheader()
            w.writerows(group_summary_rows)

    deg_rows = [r for r in rows_csv if r.get("degradation_level") != "stable"]
    dcsv = os.path.join(args.output_dir, "hidden_probe_degradation_samples.csv")
    deg_fields = [
        "degradation_level",
        "ref_id",
        "idx",
        "category_name",
        "expression",
        "prior_group",
        "matched_concepts",
        "target_concepts",
        "reference_concepts",
        "mask_iou_baseline_vs_refaware_model",
        "pred_area_delta_baseline_vs_refaware_model",
        "projected_query_cosine_baseline_vs_refaware_model",
    ]
    with open(dcsv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=deg_fields, extrasaction="ignore")
        w.writeheader()
        for r in deg_rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in deg_fields})

    json_path = os.path.join(args.output_dir, "hidden_probe_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    md_path = os.path.join(args.output_dir, "hidden_probe_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Refaware hidden / projected-query / mask mechanism probe (Phase 2)\n\n")
        f.write("## 任务目的\n\n")
        f.write(
            "在只读 merged HF 权重、仅前向推理的前提下，量化 refaware 语义先验对 LLM 层 [SEG] hidden、"
            "`SEG_token_projector` 后 segmentation query、以及最终 mask 的影响（Probe A），"
            "并记录 baseline_7w 与 refaware_28w 权重差异（Probe B，**不能因果归因于 refaware 策略**）。\n\n"
        )
        f.write(f"- **dtype**: `{dtype_str}` — {dtype_numeric_note}\n")
        f.write(f"- **baseline_model_path**: `{summary['baseline_model_path']}`\n")
        f.write(f"- **refaware_model_path**: `{summary['refaware_model_path']}`\n")
        f.write("- **merged_model_check**: passed `assert_merged_hf_dir`.\n")
        if sampling_meta:
            f.write(f"- **sampling_metadata**: ```json\n{json.dumps(sampling_meta, indent=2, ensure_ascii=False)}\n```\n")
        f.write(f"- **sample_count**: {n}\n\n")

        f.write("## Hook / shape 验收\n\n")
        f.write(
            f"- LLM 最后一层 hidden 经 `get_SEG_embedding` 取单 token，本 run 典型 shape: `{llm_shape_s}`。\n"
            f"- Projected query 经 `SEG_token_projector`，典型 shape: `{q_shape_s}`。\n"
            f"- `projected_query_confirmed`（全体样本）: `{projected_query_confirmed}`\n"
            f"- `hook_stability_mean_cosine`: {summary['hook_stability_mean_cosine']}\n"
            f"- `hook_stability_failed_count`: {hook_stability_failed_count}\n\n"
        )

        f.write("## Position / mismatch\n\n")
        f.write(f"- `position_mismatch_count`: {pos_mismatch_ct}\n")
        if len(rows_csv) <= 40:
            f.write("\nPer-sample:\n\n")
            for r in rows_csv:
                f.write(
                    f"- ref_id={r['ref_id']}: raw_seg={r['raw_seg_index']}, refaware_seg={r['refaware_seg_index']}, "
                    f"delta={r['seg_index_delta']}, prior_token_count={r['prior_token_count']}, "
                    f"position_mismatch_risk={r['position_mismatch_risk']}\n"
                )
        else:
            f.write("\n（样本较多，省略逐条 position 行；详见 `hidden_probe_per_sample.csv`。）\n")
        if failure_samples:
            f.write("\n### failure_samples\n\n")
            f.write(f"```json\n{json.dumps(failure_samples, indent=2, ensure_ascii=False)}\n```\n\n")

        f.write("## signal_survival / attenuation_ratio\n\n")
        f.write(
            "- 定义：当 `llm_seg_cosine_raw_vs_refaware < 0.99` 时，"
            "`signal_survival = (1 - projected_query_cosine) / (1 - llm_seg_cosine)`，"
            "`attenuation_ratio = 1 - signal_survival`；否则二者为 null 且 `ratio_undefined=True`。\n"
        )
        f.write(
            f"- 全体 `ratio_undefined_count`: {ratio_undefined_count}\n"
            f"- 全体 mean / median signal_survival（仅 ratio 有定义子集）: "
            f"{probe_a_mean_signal_survival} / {probe_a_median_signal_survival}\n\n"
        )

        f.write("## Probe B 退化分级（仅描述权重差异，非因果）\n\n")
        f.write(
            f"- severe / moderate / stable: {severe_degradation_count} / {moderate_degradation_count} / {stable_count}\n"
            f"- severe_rate: {summary['severe_degradation_rate']}, moderate_rate: {summary['moderate_degradation_rate']}\n"
        )
        if airport_severe:
            f.write(f"\n### airport + severe_degradation\n\n```json\n{json.dumps(airport_severe, indent=2, ensure_ascii=False)}\n```\n")
        f.write("\n")

        f.write("## prior_group / category 摘要\n\n")
        f.write(f"```json\n{json.dumps(prior_group_summary, indent=2, ensure_ascii=False)}\n```\n\n")
        f.write("## correlations\n\n")
        f.write(f"```json\n{json.dumps(correlations, indent=2, ensure_ascii=False)}\n```\n\n")

        f.write("## Probe A 均值\n\n")
        f.write(
            f"- mean llm cosine: {mean_pa_llm}\n"
            f"- mean projected-query cosine: {mean_pa_q}\n"
            f"- mean mask IoU: {mean_pa_iou}\n\n"
        )
        f.write("## Probe B 均值（7w vs 28w confounded）\n\n")
        f.write(
            f"- mean llm cosine: {summary['probe_b_mean_llm_seg_cosine']}\n"
            f"- mean projected-query cosine: {summary['probe_b_mean_projected_query_cosine']}\n"
            f"- mean mask IoU: {mean_pb_iou}\n\n"
        )
        f.write("## Verdict\n\n")
        for t in verdict_tags:
            f.write(f"- `{t}`\n")
        f.write("\n## 后续实验建议\n\n")
        f.write(f"{summary['recommendation_next_steps']}\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
