"""Helpers for A3-frozen training (only set modules trainable)."""

from __future__ import annotations

import sys
from typing import Iterable, List, Sequence, Tuple

A3_TRAINABLE_KEYWORDS = (
    "set_conditioner",
    "count_head",
    "category_set_head",
)

A3_FORBIDDEN_KEYWORDS = (
    "pixel_decoder",
    "predictor",
    "SEG_token_projector",
    "lm_head",
    "lora",
    "vision_tower",
    "mm_projector",
)


def _matches_any(name: str, keywords: Sequence[str]) -> bool:
    return any(k in name for k in keywords)


def freeze_all_parameters(model) -> None:
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_by_keywords(model, keywords: Sequence[str]) -> List[str]:
    enabled: List[str] = []
    for name, param in model.named_parameters():
        if _matches_any(name, keywords):
            param.requires_grad = True
            enabled.append(name)
    return enabled


def collect_trainable_names(model) -> List[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad]


def assert_no_forbidden_trainables(
    model,
    forbidden_keywords: Sequence[str] = A3_FORBIDDEN_KEYWORDS,
    strict: bool = True,
) -> List[str]:
    bad = [
        n
        for n, p in model.named_parameters()
        if p.requires_grad and _matches_any(n, forbidden_keywords)
    ]
    if bad:
        msg = (
            "[A3-FROZEN ERROR] Forbidden trainable parameters detected:\n"
            + "\n".join(f"  - {n}" for n in bad[:30])
        )
        if len(bad) > 30:
            msg += f"\n  ... and {len(bad) - 30} more"
        if strict:
            raise RuntimeError(msg)
        print(msg, file=sys.stderr)
    return bad


def print_trainable_parameter_report(model, title: str = "A3 trainable parameters") -> Tuple[int, int]:
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    total = sum(p.numel() for _, p in model.named_parameters())
    trainable_numel = sum(p.numel() for _, p in trainable)
    print("=" * 60)
    print(title)
    print(f"trainable params: {len(trainable)} tensors, {trainable_numel:,} elements")
    print(f"total params:     {total:,} elements")
    if trainable_numel > 0:
        print(f"trainable ratio:  {100.0 * trainable_numel / max(total, 1):.4f}%")
    for name, _ in trainable:
        print(f"  [train] {name}")
    print("=" * 60)
    return trainable_numel, total


def apply_a3_frozen_training(model, strict_forbidden: bool = True) -> List[str]:
    freeze_all_parameters(model)
    enabled = unfreeze_by_keywords(model, A3_TRAINABLE_KEYWORDS)
    assert_no_forbidden_trainables(model, strict=strict_forbidden)
    return enabled
