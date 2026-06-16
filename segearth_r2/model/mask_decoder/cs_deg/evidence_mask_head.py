"""Evidence-aware mask head additive refinement for CS-DEG++."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def apply_evidence_mask_head(
    mask_logits: torch.Tensor,
    target_evidence: torch.Tensor,
    context_evidence: torch.Tensor,
    gamma: torch.Tensor,
    eta: torch.Tensor,
) -> torch.Tensor:
    """
    Additive logit refinement:
        mask_logits + gamma * target - eta * context
    """
    size = mask_logits.shape[-2:]
    if target_evidence.shape[-2:] != size:
        target_evidence = F.interpolate(
            target_evidence.float(), size=size, mode="bilinear", align_corners=False
        ).to(mask_logits.dtype)
    if context_evidence.shape[-2:] != size:
        context_evidence = F.interpolate(
            context_evidence.float(), size=size, mode="bilinear", align_corners=False
        ).to(mask_logits.dtype)
    return mask_logits + gamma * target_evidence - eta * context_evidence
