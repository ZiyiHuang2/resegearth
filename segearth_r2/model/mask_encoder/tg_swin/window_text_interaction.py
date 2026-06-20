"""Window Text Interaction — v1 key-bias / v1.5 head-aware low-rank pairwise bias."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class StageWTIKeyBias(nn.Module):
    """v1: per-key token bias [BW, 1, 1, N]."""

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        num_heads: int,
        window_size: int,
        rank: int = 16,
        bias_max: float = 4.0,
        alpha_init: float = 0.0,
        stage_scale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.num_heads = num_heads
        self.N = window_size * window_size
        self.bias_max = bias_max
        self.stage_scale = stage_scale
        self.text_proj = nn.Linear(cond_dim, rank, bias=False)
        self.visual_proj = nn.Linear(dim, rank, bias=False)
        self.score_proj = nn.Linear(rank, self.N, bias=False)
        nn.init.normal_(self.score_proj.weight, std=1e-3)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    def compute_raw_bias(self, x_windows, text_cond, reliability):
        bw, _, _ = x_windows.shape
        batch_size = text_cond.shape[0]
        num_windows = bw // batch_size
        tc = text_cond.unsqueeze(1).expand(batch_size, num_windows, -1).reshape(bw, -1)
        visual_ctx = x_windows.mean(dim=1)
        low_rank = self.text_proj(tc) * self.visual_proj(visual_ctx)
        token_score = self.score_proj(low_rank)
        return torch.tanh(token_score) * self.bias_max

    def forward(self, x_windows, text_cond, reliability):
        bw, _, _ = x_windows.shape
        batch_size = text_cond.shape[0]
        num_windows = bw // batch_size
        rel = reliability.view(batch_size, -1)[:, :1]
        rel = rel.unsqueeze(1).expand(batch_size, num_windows, 1).reshape(bw, 1)
        raw = self.compute_raw_bias(x_windows, text_cond, reliability)
        gate = torch.tanh(self.alpha) * self.stage_scale * rel
        return (raw * gate).unsqueeze(1).unsqueeze(1), raw, gate

    def stats(self, attn_bias, raw_bias, gate, reliability):
        return {
            "alpha": float(torch.tanh(self.alpha).item()),
            "bias_abs_mean": float(attn_bias.abs().mean().item()) if attn_bias is not None else 0.0,
            "bias_abs_max": float(attn_bias.abs().max().item()) if attn_bias is not None else 0.0,
            "raw_bias_abs_mean": float(raw_bias.abs().mean().item()) if raw_bias is not None else 0.0,
            "reliability_mean": float(reliability.mean().item()) if reliability is not None else 0.0,
        }


class StageWTIHeadAware(nn.Module):
    """v1.5: head-aware low-rank pairwise bias [BW, H, N, N]."""

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        num_heads: int,
        window_size: int,
        rank: int = 16,
        bias_max: float = 4.0,
        alpha_init: float = 0.0,
        stage_scale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.num_heads = num_heads
        self.N = window_size * window_size
        self.rank = rank
        self.bias_max = bias_max
        self.stage_scale = stage_scale
        hr = num_heads * rank

        self.visual_q = nn.Linear(dim, hr, bias=False)
        self.visual_k = nn.Linear(dim, hr, bias=False)
        self.text_q = nn.Linear(cond_dim, hr, bias=False)
        self.text_k = nn.Linear(cond_dim, hr, bias=False)
        for m in (self.visual_q, self.visual_k, self.text_q, self.text_k):
            nn.init.normal_(m.weight, std=1e-3)

        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    def compute_raw_bias(
        self,
        x_windows: torch.Tensor,
        text_cond: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        bw, n_tokens, _ = x_windows.shape
        batch_size = text_cond.shape[0]
        H, r = self.num_heads, self.rank
        num_windows = bw // batch_size

        tc = text_cond.unsqueeze(1).expand(batch_size, num_windows, -1).reshape(bw, -1)

        visual_q = self.visual_q(x_windows).view(bw, n_tokens, H, r).permute(0, 2, 1, 3)
        visual_k = self.visual_k(x_windows).view(bw, n_tokens, H, r).permute(0, 2, 1, 3)
        text_q = self.text_q(tc).view(bw, H, r)
        text_k = self.text_k(tc).view(bw, H, r)

        q_bias = visual_q * text_q.unsqueeze(2)
        k_bias = visual_k * text_k.unsqueeze(2)
        bias = torch.einsum("bhnr,bhmr->bhnm", q_bias, k_bias) / math.sqrt(r)
        bias = torch.tanh(bias) * self.bias_max
        return bias

    def forward(
        self,
        x_windows: torch.Tensor,
        text_cond: torch.Tensor,
        reliability: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bw = x_windows.shape[0]
        batch_size = text_cond.shape[0]
        num_windows = bw // batch_size
        rel = reliability.view(batch_size, -1)[:, :1]
        rel = rel.unsqueeze(1).expand(batch_size, num_windows, 1).reshape(bw, 1, 1, 1)

        raw_bias = self.compute_raw_bias(x_windows, text_cond, reliability)
        gate = torch.tanh(self.alpha) * self.stage_scale * rel
        attn_bias = raw_bias * gate
        return attn_bias, raw_bias, gate

    def stats(self, attn_bias, raw_bias, gate, reliability):
        head_mean = attn_bias.abs().mean(dim=(0, 2, 3)) if attn_bias is not None else None
        ent = 0.0
        if head_mean is not None and head_mean.sum() > 0:
            p = head_mean / head_mean.sum().clamp_min(1e-8)
            ent = float(-(p * (p + 1e-8).log()).sum().item())
        return {
            "alpha": float(torch.tanh(self.alpha).item()),
            "bias_abs_mean": float(attn_bias.abs().mean().item()) if attn_bias is not None else 0.0,
            "bias_abs_max": float(attn_bias.abs().max().item()) if attn_bias is not None else 0.0,
            "raw_bias_abs_mean": float(raw_bias.abs().mean().item()) if raw_bias is not None else 0.0,
            "reliability_mean": float(reliability.mean().item()) if reliability is not None else 0.0,
            "head_bias_entropy": ent,
        }


class TGSwimController(nn.Module):
    """Per-stage WTI registry; selects stage slice from [B,S,C] text cond."""

    STAGE_DIMS_BASE = [128, 256, 512, 1024]
    STAGE_HEADS_BASE = [4, 8, 16, 32]

    def __init__(
        self,
        cond_dim: int = 256,
        wti_rank: int = 16,
        wti_stages: Optional[List[int]] = None,
        wti_start_layer: int = 0,
        bias_max: float = 4.0,
        alpha_init: float = 0.0,
        window_size: int = 12,
        swin_type: str = "base",
        stage_scales: Optional[List[float]] = None,
        log_stats: bool = False,
        head_aware: bool = True,
        num_text_stages: int = 4,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.wti_stages = set(wti_stages if wti_stages is not None else [1, 2, 3])
        self.wti_start_layer = wti_start_layer
        self.window_size = window_size
        self.log_stats = log_stats
        self.head_aware = head_aware
        self.num_text_stages = num_text_stages
        self._last_stats: Dict[str, float] = {}

        if swin_type == "large":
            stage_dims = [192, 384, 768, 1536]
            stage_heads = [6, 12, 24, 48]
        else:
            stage_dims = self.STAGE_DIMS_BASE
            stage_heads = self.STAGE_HEADS_BASE

        if stage_scales is None:
            stage_scales = [0.25, 0.5, 0.75, 1.0]

        wti_cls = StageWTIHeadAware if head_aware else StageWTIKeyBias
        self.wti_blocks = nn.ModuleDict()
        for stage_idx in self.wti_stages:
            if stage_idx >= len(stage_dims):
                continue
            self.wti_blocks[str(stage_idx)] = wti_cls(
                dim=stage_dims[stage_idx],
                cond_dim=cond_dim,
                num_heads=stage_heads[stage_idx],
                window_size=window_size,
                rank=wti_rank,
                bias_max=bias_max,
                alpha_init=alpha_init,
                stage_scale=stage_scales[stage_idx] if stage_idx < len(stage_scales) else 1.0,
            )

    def _select_stage_cond(
        self,
        text_cond: torch.Tensor,
        reliability: torch.Tensor,
        swin_stage_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if text_cond.dim() == 3:
            s = min(swin_stage_idx, text_cond.shape[1] - 1)
            tc = text_cond[:, s, :]
            if reliability.dim() == 3:
                rel = reliability[:, s, :]
            else:
                rel = reliability
        else:
            tc = text_cond
            rel = reliability.view(reliability.shape[0], -1)[:, :1] if reliability.dim() > 2 else reliability
        return tc, rel

    def enabled_for(self, stage_idx: int, layer_idx: int) -> bool:
        return stage_idx in self.wti_stages and layer_idx >= self.wti_start_layer

    def compute_bias(
        self,
        stage_idx: int,
        layer_idx: int,
        x_windows: torch.Tensor,
        text_cond: Optional[torch.Tensor],
        reliability: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if text_cond is None or reliability is None:
            return None
        if not self.enabled_for(stage_idx, layer_idx):
            return None
        key = str(stage_idx)
        if key not in self.wti_blocks:
            return None

        tc, rel = self._select_stage_cond(text_cond, reliability, stage_idx)
        module = self.wti_blocks[key]
        attn_bias, raw_bias, gate = module(x_windows, tc, rel)

        if self.log_stats:
            st = module.stats(attn_bias, raw_bias, gate, rel)
            st["stage_idx"] = float(stage_idx)
            st["layer_idx"] = float(layer_idx)
            st["active"] = 1.0
            self._last_stats.update(st)
        return attn_bias

    def pop_stats(self) -> Dict[str, float]:
        stats = dict(self._last_stats)
        self._last_stats = {}
        return stats
