"""Soft additive evidence bias for Mask2Former cross-attention."""
from __future__ import annotations

import torch


def normalize_per_sample(x: torch.Tensor) -> torch.Tensor:
    """Per-(B,Q) spatial z-score + tanh. x: [B, Q, H, W]."""
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return torch.tanh((x - mean) / std)


def bool_mask_to_additive(
    bool_mask: torch.Tensor,
    block_value: float = -1e4,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """True (blocked) -> block_value, False -> 0."""
    mask_dtype = dtype or torch.float32
    out = torch.zeros(bool_mask.shape, dtype=mask_dtype, device=bool_mask.device)
    return out.masked_fill(bool_mask, block_value)


def compute_bias_scale(
    alpha: torch.Tensor,
    global_step: int,
    bias_max: float,
    bias_warmup_steps: int,
) -> torch.Tensor:
    if bias_warmup_steps <= 0:
        warmup = 1.0
    else:
        warmup = min(1.0, max(0.0, float(global_step) / float(bias_warmup_steps)))
    return bias_max * warmup * alpha


def fuse_evidence_attention_mask(
    bool_attn_mask: torch.Tensor,
    target_logits: torch.Tensor,
    context_logits: torch.Tensor,
    alpha: torch.Tensor,
    global_step: int,
    num_heads: int,
    context_lambda: float = 1.0,
    bias_max: float = 5.0,
    bias_warmup_steps: int = 1000,
    mask_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fuse bool Mask2Former mask with soft evidence bias.

    bool_attn_mask: [B*nheads, Q, HW] bool
    target/context_logits: [B, Q, H, W]
    Returns (float_attn_mask, bias_scale_scalar)
    """
    e = target_logits - context_lambda * context_logits
    e = normalize_per_sample(e)
    bias_scale = compute_bias_scale(alpha, global_step, bias_max, bias_warmup_steps)
    evidence_bias = bias_scale * e
    evidence_bias = evidence_bias.clamp(-bias_max, bias_max)

    batch_size, q, _, _ = target_logits.shape
    evidence_bias_flat = (
        evidence_bias.flatten(2)
        .unsqueeze(1)
        .repeat(1, num_heads, 1, 1)
        .flatten(0, 1)
    )
    attn_dtype = mask_dtype or target_logits.dtype
    bool_additive = bool_mask_to_additive(bool_attn_mask, dtype=attn_dtype)
    return bool_additive + evidence_bias_flat.to(attn_dtype), bias_scale
