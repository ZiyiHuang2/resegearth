"""Lightweight NaN/Inf diagnostics for grouped SET++ training (env-gated)."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)

_MASKED_SOFTMAX_FILL = -1e4
_DEBUG_ENV = "GROUPED_SETPP_NAN_DEBUG"
_DUMP_DIR_ENV = "GROUPED_SETPP_NAN_DUMP_DIR"


def enabled() -> bool:
    return os.environ.get(_DEBUG_ENV, "0") == "1"


_current_step = 0


def set_step(step: int) -> None:
    global _current_step
    _current_step = int(step)


def current_step() -> int:
    return _current_step


def masked_softmax_fill() -> float:
    return _MASKED_SOFTMAX_FILL


def tensor_stats(tensor: Optional[torch.Tensor], name: str) -> Dict[str, Any]:
    if tensor is None:
        return {f"{name}": None}
    if not torch.is_tensor(tensor):
        return {f"{name}": repr(tensor)}
    t = tensor.detach()
    if t.numel() == 0:
        return {f"{name}/empty": True}
    tf = t.float()
    return {
        f"{name}/min": float(tf.min().item()),
        f"{name}/max": float(tf.max().item()),
        f"{name}/mean": float(tf.mean().item()),
        f"{name}/isnan": bool(torch.isnan(tf).any().item()),
        f"{name}/isinf": bool(torch.isinf(tf).any().item()),
    }


def has_nan_or_inf(tensor: Optional[torch.Tensor]) -> bool:
    if tensor is None or not torch.is_tensor(tensor) or tensor.numel() == 0:
        return False
    t = tensor.detach()
    return bool(torch.isnan(t).any().item() or torch.isinf(t).any().item())


def _dump_path(step: int) -> Optional[Path]:
    dump_dir = os.environ.get(_DUMP_DIR_ENV, "")
    if not dump_dir:
        return None
    root = Path(dump_dir)
    root.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return root / f"nan_step{step}_{ts}.json"


def log_nan_report(tag: str, step: int, report: Dict[str, Any]) -> None:
    if not enabled():
        return
    flat: Dict[str, Any] = {}
    for key, section in report.items():
        if isinstance(section, dict):
            flat.update(section)
        else:
            flat[key] = section
    parts = [f"{k}={v}" for k, v in flat.items()]
    logger.warning("[NaN audit step %d] %s | %s", step, tag, " | ".join(parts))

    dump_file = _dump_path(step)
    if dump_file is not None:
        payload = {"tag": tag, "step": step, "report": report}
        dump_file.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        logger.warning("[NaN audit] saved %s", dump_file)


def audit_mask_outputs(outputs: Dict[str, Any], step: int) -> bool:
    if not enabled():
        return False
    report: Dict[str, Any] = {}
    for key in (
        "pred_set_union_mask",
        "pred_seg_masks_grouped",
        "pred_seg_masks",
        "pred_masks",
    ):
        if key in outputs:
            report[key] = tensor_stats(outputs[key], key)

    bad = any(
        section.get(f"{k}/isnan") or section.get(f"{k}/isinf")
        for k, section in ((k, report[k]) for k in report)
    )
    if bad:
        log_nan_report("predictor_outputs", step, report)
    return bad


def audit_set_union_head(
    slot_score: torch.Tensor,
    slot_weight: torch.Tensor,
    slot_valid_mask: torch.Tensor,
    set_feature: torch.Tensor,
    mask_embed_set: Optional[torch.Tensor],
    step: Optional[int] = None,
) -> bool:
    if not enabled():
        return False
    step = current_step() if step is None else step
    valid_counts = slot_valid_mask.sum(dim=1).tolist() if slot_valid_mask is not None else []
    report = {
        "slot": tensor_stats(slot_score, "slot_score"),
        "weight": tensor_stats(slot_weight, "slot_weight"),
        "set_feature": tensor_stats(set_feature, "set_feature"),
        "valid_counts": {"slot_valid_per_row": valid_counts},
    }
    if mask_embed_set is not None:
        report["mask_embed_set"] = tensor_stats(mask_embed_set, "mask_embed_set")
    if slot_weight is not None and slot_weight.numel():
        report["weight_sum"] = {
            "slot_weight_row_sum/min": float(slot_weight.sum(dim=1).min().item()),
            "slot_weight_row_sum/max": float(slot_weight.sum(dim=1).max().item()),
        }
    bad = any(
        has_nan_or_inf(t)
        for t in (slot_score, slot_weight, set_feature, mask_embed_set)
    )
    if bad:
        log_nan_report("SetUnionMaskHead", step, report)
    return bad


def audit_criterion_grouped(
    outputs: Dict[str, Any],
    losses: Dict[str, torch.Tensor],
    gt_masks_flat: torch.Tensor,
    gt_union_b: torch.Tensor,
    step: int,
) -> bool:
    if not enabled():
        return False
    report: Dict[str, Any] = {
        "gt_masks_flat": tensor_stats(gt_masks_flat, "gt_masks_flat"),
        "gt_union_b": tensor_stats(gt_union_b, "gt_union_b"),
    }
    first_nan = None
    for name, value in losses.items():
        if value is None or not torch.is_tensor(value):
            continue
        if torch.isnan(value).any() or torch.isinf(value).any():
            if first_nan is None:
                first_nan = name
            report[f"loss/{name}"] = float(value.detach().float().item())
    for key in ("pred_set_union_mask", "pred_seg_masks_grouped", "pred_seg_masks"):
        if key in outputs:
            report[key] = tensor_stats(outputs[key], key)

    bad = first_nan is not None or report["gt_masks_flat"].get("gt_masks_flat/isnan", False)
    if bad:
        report["first_nan_loss"] = {"name": first_nan}
        log_nan_report("criterion_grouped", step, report)
    return bad
