"""DR-EWTI: Dynamic Relational Evidence-guided Window-Text Interaction."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .window_text_interaction import StageWTIHeadAware


class EvidenceTokenFusion(nn.Module):
    """Fuse per-window visual tokens with coarse probability evidence."""

    def __init__(self, dim: int, evidence_dim: int = 32, eps: float = 1e-4, logit_clip: float = 4.0):
        super().__init__()
        self.dim = dim
        self.evidence_dim = evidence_dim
        self.eps = eps
        self.logit_clip = logit_clip

        self.evidence_feature_proj = nn.Linear(3, evidence_dim)
        self.evidence_visual_proj = nn.Linear(dim, evidence_dim)
        self.evidence_add_proj = nn.Linear(evidence_dim, evidence_dim)
        self.evidence_gate_visual = nn.Linear(dim, evidence_dim, bias=False)
        self.evidence_gate_mask = nn.Linear(evidence_dim, evidence_dim, bias=False)
        self.evidence_norm = nn.LayerNorm(evidence_dim)

        for m in (
            self.evidence_feature_proj,
            self.evidence_visual_proj,
            self.evidence_add_proj,
        ):
            nn.init.normal_(m.weight, std=1e-3)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.evidence_gate_visual.weight, std=1e-3)
        nn.init.normal_(self.evidence_gate_mask.weight, std=1e-3)

    def forward(self, x_windows: torch.Tensor, evidence_windows: torch.Tensor) -> torch.Tensor:
        orig_dtype = x_windows.dtype
        p = evidence_windows.clamp(self.eps, 1.0 - self.eps).float()
        uncertainty = 4.0 * p * (1.0 - p)
        bounded_logit = torch.logit(p).clamp(-self.logit_clip, self.logit_clip) / self.logit_clip
        evidence_base = torch.stack([p, uncertainty, bounded_logit], dim=-1).to(orig_dtype)

        evidence_feat = self.evidence_feature_proj(evidence_base)
        visual_feat = self.evidence_visual_proj(x_windows)
        gate = self.evidence_gate_visual(x_windows) * self.evidence_gate_mask(evidence_feat)
        return self.evidence_norm(
            visual_feat + self.evidence_add_proj(evidence_feat) + gate
        )


class StageDynamicRelationalWTI(nn.Module):
    """Evidence-only extension; base v1.5 WTI params live in sibling wti_blocks entry."""

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        num_heads: int,
        rank: int,
        bias_max: float,
        evidence_dim: int = 32,
        eps: float = 1e-4,
        logit_clip: float = 4.0,
        evidence_relation_gate_init: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.num_heads = num_heads
        self.rank = rank
        self.bias_max = bias_max

        self.evidence_fusion = EvidenceTokenFusion(
            dim=dim,
            evidence_dim=evidence_dim,
            eps=eps,
            logit_clip=logit_clip,
        )
        hr = num_heads * rank
        self.evidence_q = nn.Linear(evidence_dim, hr, bias=False)
        self.evidence_k = nn.Linear(evidence_dim, hr, bias=False)
        nn.init.normal_(self.evidence_q.weight, std=1e-3)
        nn.init.normal_(self.evidence_k.weight, std=1e-3)

        self.evidence_relation_gate = nn.Parameter(
            torch.tensor(evidence_relation_gate_init, dtype=torch.float32)
        )

    @staticmethod
    def _broadcast_text_cond(text_cond: torch.Tensor, bw: int) -> torch.Tensor:
        batch_size = text_cond.shape[0]
        num_windows = bw // batch_size
        return text_cond.unsqueeze(1).expand(batch_size, num_windows, -1).reshape(bw, -1)

    def compute_raw_bias(
        self,
        base_wti: StageWTIHeadAware,
        x_windows: torch.Tensor,
        text_cond: torch.Tensor,
        evidence_windows: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if evidence_windows is None:
            return base_wti.compute_raw_bias(x_windows, text_cond, reliability=None)

        bw, n_tokens, _ = x_windows.shape
        H, r = self.num_heads, self.rank
        tc = self._broadcast_text_cond(text_cond, bw)

        visual_q = base_wti.visual_q(x_windows).view(bw, n_tokens, H, r).permute(0, 2, 1, 3)
        visual_k = base_wti.visual_k(x_windows).view(bw, n_tokens, H, r).permute(0, 2, 1, 3)
        text_q = base_wti.text_q(tc).view(bw, H, r)
        text_k = base_wti.text_k(tc).view(bw, H, r)
        tq = text_q.unsqueeze(2)
        tk = text_k.unsqueeze(2)

        q_base = visual_q * tq
        k_base = visual_k * tk

        delta = torch.tanh(self.evidence_relation_gate)
        z = self.evidence_fusion(x_windows, evidence_windows)
        evidence_q = self.evidence_q(z).view(bw, n_tokens, H, r).permute(0, 2, 1, 3)
        evidence_k = self.evidence_k(z).view(bw, n_tokens, H, r).permute(0, 2, 1, 3)

        q_dynamic = q_base + delta * (evidence_q * tq)
        k_dynamic = k_base + delta * (evidence_k * tk)

        raw_bias = torch.einsum("bhnr,bhmr->bhnm", q_dynamic, k_dynamic) / math.sqrt(r)
        return torch.tanh(raw_bias) * self.bias_max

    def forward(
        self,
        base_wti: StageWTIHeadAware,
        x_windows: torch.Tensor,
        text_cond: torch.Tensor,
        reliability: torch.Tensor,
        evidence_windows: Optional[torch.Tensor] = None,
        log_stats: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict[str, float]]]:
        bw = x_windows.shape[0]
        batch_size = text_cond.shape[0]
        num_windows = bw // batch_size
        rel = reliability.view(batch_size, -1)[:, :1]
        rel = rel.unsqueeze(1).expand(batch_size, num_windows, 1).reshape(bw, 1, 1, 1)

        raw_bias = self.compute_raw_bias(base_wti, x_windows, text_cond, evidence_windows)
        gate = torch.tanh(base_wti.alpha) * base_wti.stage_scale * rel
        attn_bias = raw_bias * gate

        diag = None
        if log_stats:
            p = evidence_windows.clamp(self.evidence_fusion.eps, 1.0 - self.evidence_fusion.eps) if evidence_windows is not None else None
            v15_raw = base_wti.compute_raw_bias(x_windows, text_cond, reliability)
            diag = {
                "used_dr_path": 1.0 if evidence_windows is not None else 0.0,
                "evidence_relation_gate": float(torch.tanh(self.evidence_relation_gate).detach().cpu()),
                "evidence_prob_mean": float(p.mean().detach().cpu()) if p is not None else 0.0,
                "evidence_uncertainty_mean": float((4.0 * p * (1.0 - p)).mean().detach().cpu()) if p is not None else 0.0,
                "dynamic_bias_abs_mean": float(raw_bias.detach().abs().mean().cpu()),
                "v15_bias_abs_mean": float(v15_raw.detach().abs().mean().cpu()),
                "alpha": float(torch.tanh(base_wti.alpha).detach().cpu()),
                "bias_abs_mean": float(attn_bias.detach().abs().mean().cpu()),
                "raw_bias_abs_mean": float(raw_bias.detach().abs().mean().cpu()),
            }
        return attn_bias, raw_bias, gate, diag
