from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_4d_masks(masks: torch.Tensor) -> torch.Tensor:
    if masks.dim() == 2:
        masks = masks.unsqueeze(0).unsqueeze(0)
    elif masks.dim() == 3:
        masks = masks.unsqueeze(1)
    elif masks.dim() != 4:
        raise ValueError(f"Unsupported mask shape: {tuple(masks.shape)}")
    return masks.float()


def masks_to_boundary_targets(masks: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Morphological gradient boundary target in [0,1], output shape [N,1,H,W]."""
    masks = _ensure_4d_masks(masks)
    pad = kernel_size // 2
    dilated = F.max_pool2d(masks, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-masks, kernel_size=kernel_size, stride=1, padding=pad)
    boundary = (dilated - eroded).clamp_(0.0, 1.0)
    return boundary


def mask_logits_to_boundary_prob(mask_logits: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Convert predicted mask logits to soft boundary probability map."""
    probs = _ensure_4d_masks(mask_logits.sigmoid())
    return masks_to_boundary_targets(probs, kernel_size=kernel_size)


def compute_small_object_weights(
    masks: torch.Tensor,
    percentile: float = 30.0,
    small_weight: float = 1.8,
    normal_weight: float = 1.0,
) -> torch.Tensor:
    """Per-sample weight by object area percentile (P30 by default)."""
    masks = _ensure_4d_masks(masks)
    area = masks.flatten(1).sum(dim=1)
    valid = area > 0
    weights = torch.full_like(area, float(normal_weight), dtype=torch.float)
    if valid.any():
        p = torch.quantile(area[valid].float(), float(percentile) / 100.0)
        weights = torch.where(area <= p, torch.full_like(weights, float(small_weight)), weights)
    return weights


class BoundaryProposalHead(nn.Module):
    """BPH: predict boundary logits from last two multi-scale decoder features."""

    def __init__(self, in_channels: int = 256, hidden_channels: int = 256):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.boundary_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.boundary_proj = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)

    def forward(self, feat_last: torch.Tensor, feat_prev: torch.Tensor):
        if feat_last.dim() != 4 or feat_prev.dim() != 4:
            raise ValueError("BoundaryProposalHead expects 4D features")
        if feat_last.shape[0] != feat_prev.shape[0]:
            raise ValueError("Batch mismatch between last two multi-scale features")

        if feat_prev.shape[-2:] != feat_last.shape[-2:]:
            feat_last = F.interpolate(feat_last, size=feat_prev.shape[-2:], mode='bilinear', align_corners=False)

        fused = self.fuse(torch.cat([feat_last, feat_prev], dim=1))
        boundary_logits = self.boundary_head(fused)
        boundary_tokens = self.boundary_proj(fused).flatten(2).permute(2, 0, 1).contiguous()  # [S,B,C]
        return boundary_logits, boundary_tokens

    def forward_with_fused_feature(self, feat_last: torch.Tensor, feat_prev: torch.Tensor):
        """Return boundary outputs plus fused feature map for bi-lite Q2B modulation."""
        if feat_last.dim() != 4 or feat_prev.dim() != 4:
            raise ValueError("BoundaryProposalHead expects 4D features")
        if feat_last.shape[0] != feat_prev.shape[0]:
            raise ValueError("Batch mismatch between last two multi-scale features")

        if feat_prev.shape[-2:] != feat_last.shape[-2:]:
            feat_last = F.interpolate(feat_last, size=feat_prev.shape[-2:], mode='bilinear', align_corners=False)

        fused = self.fuse(torch.cat([feat_last, feat_prev], dim=1))
        boundary_logits = self.boundary_head(fused)
        boundary_tokens = self.boundary_proj(fused).flatten(2).permute(2, 0, 1).contiguous()  # [S,B,C]
        return boundary_logits, boundary_tokens, fused

    def features_to_boundary(self, fused_feature: torch.Tensor):
        """Project fused boundary feature to logits and boundary memory tokens."""
        boundary_logits = self.boundary_head(fused_feature)
        boundary_tokens = self.boundary_proj(fused_feature).flatten(2).permute(2, 0, 1).contiguous()  # [S,B,C]
        return boundary_logits, boundary_tokens


class _QBRLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=0.0, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor):
        attn_out, _ = self.cross_attn(q, kv, kv)
        q = self.norm1(q + attn_out)
        q = self.norm2(q + self.ffn(q))
        return q


class QueryBoundaryRefiner(nn.Module):
    """QBR: refine SEG query with boundary memory tokens, depth=2 by default."""

    def __init__(self, dim: int = 256, num_heads: int = 8, depth: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([_QBRLayer(dim, num_heads) for _ in range(int(depth))])

    def forward(self, seg_query: torch.Tensor, boundary_tokens: torch.Tensor):
        # seg_query [B,Q,C], boundary_tokens [S,B,C]
        if seg_query is None or boundary_tokens is None:
            return seg_query
        if seg_query.dim() != 3 or boundary_tokens.dim() != 3:
            raise ValueError(
                f"QueryBoundaryRefiner expects [B,Q,C] and [S,B,C], got {tuple(seg_query.shape)} and {tuple(boundary_tokens.shape)}"
            )
        kv = boundary_tokens.permute(1, 0, 2)
        out = seg_query
        for layer in self.layers:
            out = layer(out, kv)
        return out


class QueryToBoundaryModulator(nn.Module):
    """Bi-lite Q2B: use refined query to weakly modulate boundary feature channels."""

    def __init__(self, dim: int = 256, alpha: float = 0.1):
        super().__init__()
        self.dim = dim
        self.alpha = float(alpha)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim * 2),
        )

    def forward(self, fused_feature: torch.Tensor, refined_query: torch.Tensor, alpha: Optional[float] = None):
        # fused_feature: [B,C,H,W], refined_query: [B,Q,C]
        if fused_feature is None or refined_query is None:
            return fused_feature
        if fused_feature.dim() != 4 or refined_query.dim() != 3:
            raise ValueError(
                f"QueryToBoundaryModulator expects [B,C,H,W] and [B,Q,C], got "
                f"{tuple(fused_feature.shape)} and {tuple(refined_query.shape)}"
            )
        if fused_feature.shape[0] != refined_query.shape[0]:
            raise ValueError("Batch mismatch between fused feature and refined query")

        q = refined_query.mean(dim=1)  # [B,C]
        gb = self.proj(q)  # [B,2C]
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        mod_alpha = self.alpha if alpha is None else float(alpha)
        return fused_feature * (1.0 + mod_alpha * torch.tanh(gamma)) + mod_alpha * beta


def boundary_token_drift_loss(tokens_before: torch.Tensor, tokens_after: torch.Tensor) -> torch.Tensor:
    if tokens_before is None or tokens_after is None:
        raise ValueError("tokens_before/tokens_after must not be None")
    return F.l1_loss(tokens_after, tokens_before)
