"""Text Condition Factory (TCF) — v1 flat cond / v1.5 stage-wise router."""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class StageWiseTextRouter(nn.Module):
    """Stage-wise text router: distinct condition per Swin stage."""

    def __init__(
        self,
        text_dim: int,
        cond_dim: int = 256,
        num_stages: int = 4,
        router_hidden_dim: int = 512,
        reliability_init: float = 0.0,
        use_relation_pool: bool = True,
        fallback_warn_limit: int = 5,
        context_radius: int = 8,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.num_stages = num_stages
        self.use_relation_pool = use_relation_pool
        self.context_radius = context_radius
        self._fallback_warn_remaining = fallback_warn_limit

        self.seg_norm = nn.LayerNorm(text_dim)
        self.seg_proj = nn.Linear(text_dim, cond_dim)
        self.phrase_norm = nn.LayerNorm(text_dim)
        self.phrase_proj = nn.Linear(text_dim, cond_dim)

        if use_relation_pool:
            self.rel_scale = cond_dim ** -0.5
            self.rel_query = nn.Linear(cond_dim, cond_dim)
            self.rel_key = nn.Linear(text_dim, cond_dim)
            self.rel_value = nn.Linear(text_dim, cond_dim)
            nn.init.normal_(self.rel_query.weight, std=1e-3)
            nn.init.normal_(self.rel_key.weight, std=1e-3)
            nn.init.normal_(self.rel_value.weight, std=1e-3)

        self.stage_embed = nn.Parameter(torch.randn(num_stages, cond_dim) * 0.02)
        self.stage_router = nn.Sequential(
            nn.Linear(cond_dim * 3, router_hidden_dim),
            nn.GELU(),
            nn.Linear(router_hidden_dim, num_stages * 3),
        )
        nn.init.normal_(self.stage_router[-1].weight, std=1e-3)
        nn.init.zeros_(self.stage_router[-1].bias)

        self.out_norm = nn.LayerNorm(cond_dim)
        self.reliability_head = nn.Linear(cond_dim, 1)
        nn.init.normal_(self.reliability_head.weight, std=1e-3)
        nn.init.constant_(self.reliability_head.bias, reliability_init)

    def _pool_local_context(self, hidden_states, seg_indices):
        pooled = []
        for seq_hidden, seq_mask in zip(hidden_states, seg_indices):
            positions = seq_mask.nonzero(as_tuple=False).squeeze(-1)
            for pos in positions:
                pos = int(pos.item())
                lo = max(0, pos - self.context_radius)
                hi = min(seq_hidden.shape[0], pos + self.context_radius + 1)
                pooled.append(seq_hidden[lo:hi].mean(dim=0))
        if not pooled:
            return torch.zeros(0, hidden_states.shape[-1], device=hidden_states.device, dtype=hidden_states.dtype)
        return torch.stack(pooled, dim=0)

    def _relation_pool(self, seg_feat, phrase_hidden, phrase_mask):
        """Seg-query attention over phrase tokens."""
        n = seg_feat.shape[0]
        if phrase_hidden is None or phrase_mask is None or not phrase_mask.any():
            return torch.zeros_like(seg_feat)

        max_len = phrase_hidden.shape[1]
        q = self.rel_query(seg_feat).unsqueeze(1)
        k = self.rel_key(phrase_hidden)
        v = self.rel_value(phrase_hidden)
        attn_logits = (q * k).sum(dim=-1) * self.rel_scale
        attn_logits = attn_logits.masked_fill(~phrase_mask, float("-inf"))
        attn = torch.softmax(attn_logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        relation_ctx = torch.bmm(attn.unsqueeze(1), v).squeeze(1)
        return relation_ctx

    def forward(
        self,
        seg_hidden: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        seg_indices: Optional[torch.Tensor] = None,
        phrase_hidden: Optional[torch.Tensor] = None,
        phrase_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_target = seg_hidden.shape[0]
        device = seg_hidden.device
        dtype = seg_hidden.dtype

        seg_feat = self.seg_proj(self.seg_norm(seg_hidden))

        param_dtype = self.seg_proj.weight.dtype
        has_refer_span = (
            phrase_hidden is not None
            and phrase_mask is not None
            and phrase_hidden.numel() > 0
            and phrase_mask.any()
        )
        if has_refer_span:
            mask = phrase_mask.float().unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            phrase_ctx = (phrase_hidden * mask).sum(dim=1) / denom
        elif hidden_states is not None and seg_indices is not None:
            phrase_ctx = self._pool_local_context(hidden_states, seg_indices)
            if phrase_ctx.shape[0] != n_target:
                phrase_ctx = phrase_ctx[:n_target]
            if self._fallback_warn_remaining > 0:
                self._fallback_warn_remaining -= 1
                logger.warning("[TG_SWIN v1.5] refer span missing; local pool fallback")
        else:
            phrase_ctx = torch.zeros_like(seg_hidden)

        phrase_ctx = phrase_ctx.to(dtype=param_dtype)
        phrase_feat = self.phrase_proj(self.phrase_norm(phrase_ctx))
        if self.use_relation_pool and has_refer_span:
            relation_ctx = self._relation_pool(seg_feat, phrase_hidden, phrase_mask)
        else:
            relation_ctx = phrase_feat
        relation_ctx = relation_ctx.to(dtype=param_dtype)

        router_in = torch.cat([seg_feat, phrase_feat, relation_ctx], dim=-1)
        gates = self.stage_router(router_in).view(n_target, self.num_stages, 3)
        gate_seg = torch.sigmoid(gates[:, :, 0:1])
        gate_phr = torch.sigmoid(gates[:, :, 1:2])
        gate_rel = torch.sigmoid(gates[:, :, 2:3])

        stage_embed = self.stage_embed.unsqueeze(0).expand(n_target, -1, -1)
        seg_exp = seg_feat.unsqueeze(1).expand(-1, self.num_stages, -1)
        phr_exp = phrase_feat.unsqueeze(1).expand(-1, self.num_stages, -1)
        rel_exp = relation_ctx.unsqueeze(1).expand(-1, self.num_stages, -1)

        mixed = gate_seg * seg_exp + gate_phr * phr_exp + gate_rel * rel_exp + stage_embed
        stage_text_cond = self.out_norm(mixed)
        reliability = torch.sigmoid(self.reliability_head(stage_text_cond))

        return stage_text_cond.to(device=device, dtype=dtype), reliability.to(device=device, dtype=dtype)


class TextConditionFactory(nn.Module):
    """v1: [N,C] flat cond; v1.5: [N,S,C] stage-wise router."""

    def __init__(
        self,
        text_dim: int,
        cond_dim: int = 256,
        reliability_init: float = 0.0,
        use_phrase_pool: bool = True,
        context_radius: int = 8,
        fallback_warn_limit: int = 5,
        version: str = "v1",
        num_stages: int = 4,
        stage_router: bool = False,
        router_hidden_dim: int = 512,
        use_relation_pool: bool = True,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.version = version
        self.use_v15 = version == "v1.5" or stage_router

        if self.use_v15:
            self.router = StageWiseTextRouter(
                text_dim=text_dim,
                cond_dim=cond_dim,
                num_stages=num_stages,
                router_hidden_dim=router_hidden_dim,
                reliability_init=reliability_init,
                use_relation_pool=use_relation_pool,
                fallback_warn_limit=fallback_warn_limit,
                context_radius=context_radius,
            )
        else:
            self.use_phrase_pool = use_phrase_pool
            self.context_radius = context_radius
            self._fallback_warn_remaining = fallback_warn_limit
            self.seg_norm = nn.LayerNorm(text_dim)
            self.seg_proj = nn.Linear(text_dim, cond_dim)
            if use_phrase_pool:
                self.phrase_norm = nn.LayerNorm(text_dim)
                self.phrase_proj = nn.Linear(text_dim, cond_dim)
                self.reliability_logit = nn.Parameter(torch.tensor(reliability_init, dtype=torch.float32))
                self.phrase_gate = nn.Linear(cond_dim, cond_dim)
                nn.init.zeros_(self.phrase_gate.weight)
                nn.init.zeros_(self.phrase_gate.bias)
            else:
                self.phrase_norm = None
                self.phrase_proj = None
                self.reliability_logit = None
                self.phrase_gate = None
            self.out_norm = nn.LayerNorm(cond_dim)
            self.out_proj = nn.Linear(cond_dim, cond_dim)
            nn.init.normal_(self.out_proj.weight, std=1e-3)
            nn.init.zeros_(self.out_proj.bias)

    def _pool_local_context(self, hidden_states, seg_indices):
        pooled = []
        for seq_hidden, seq_mask in zip(hidden_states, seg_indices):
            positions = seg_mask.nonzero(as_tuple=False).squeeze(-1)
            for pos in positions:
                pos = int(pos.item())
                lo = max(0, pos - self.context_radius)
                hi = min(seq_hidden.shape[0], pos + self.context_radius + 1)
                pooled.append(seq_hidden[lo:hi].mean(dim=0))
        if not pooled:
            return torch.zeros(0, hidden_states.shape[-1], device=hidden_states.device, dtype=hidden_states.dtype)
        return torch.stack(pooled, dim=0)

    def _forward_v1(self, seg_hidden, hidden_states, seg_indices, phrase_hidden, phrase_mask):
        n_target = seg_hidden.shape[0]
        device = seg_hidden.device
        dtype = seg_hidden.dtype
        seg_feat = self.seg_proj(self.seg_norm(seg_hidden))
        if self.use_phrase_pool:
            has_refer_span = (
                phrase_hidden is not None and phrase_mask is not None
                and phrase_hidden.numel() > 0 and phrase_mask.any()
            )
            if has_refer_span:
                mask = phrase_mask.float().unsqueeze(-1)
                denom = mask.sum(dim=1).clamp_min(1.0)
                phrase_ctx = (phrase_hidden * mask).sum(dim=1) / denom
            elif hidden_states is not None and seg_indices is not None:
                phrase_ctx = self._pool_local_context(hidden_states, seg_indices)
                if phrase_ctx.shape[0] != n_target:
                    phrase_ctx = phrase_ctx[:n_target]
            else:
                phrase_ctx = torch.zeros_like(seg_hidden)
            phrase_feat = self.phrase_proj(self.phrase_norm(phrase_ctx))
            reliability = torch.sigmoid(self.reliability_logit).expand(n_target, 1).to(device=device, dtype=dtype)
            mixed = seg_feat + reliability * self.phrase_gate(phrase_feat)
        else:
            mixed = seg_feat
            reliability = torch.ones(n_target, 1, device=device, dtype=dtype)
        text_cond = self.out_proj(self.out_norm(mixed))
        return text_cond, reliability

    def forward(
        self,
        seg_hidden: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        seg_indices: Optional[torch.Tensor] = None,
        phrase_hidden: Optional[torch.Tensor] = None,
        phrase_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_v15:
            param_dtype = self.router.seg_proj.weight.dtype
        else:
            param_dtype = self.seg_proj.weight.dtype
        seg_hidden = seg_hidden.to(dtype=param_dtype)
        if phrase_hidden is not None:
            phrase_hidden = phrase_hidden.to(dtype=param_dtype)
        if hidden_states is not None:
            hidden_states = hidden_states.to(dtype=param_dtype)
        if self.use_v15:
            return self.router(
                seg_hidden,
                hidden_states=hidden_states,
                seg_indices=seg_indices,
                phrase_hidden=phrase_hidden,
                phrase_mask=phrase_mask,
            )
        return self._forward_v1(seg_hidden, hidden_states, seg_indices, phrase_hidden, phrase_mask)
