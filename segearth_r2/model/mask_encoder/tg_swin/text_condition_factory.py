"""Text Condition Factory (TCF) — v1 flat cond / v1.5 stage-wise router."""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class StageWiseTextRouter(nn.Module):
    """Stage-wise text router: one condition vector per WTI injection stage."""

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
        use_stage_phrase: bool = False,
        use_hybrid_reliability: bool = False,
        stage_phrase_temp: float = 1.0,
        hybrid_reliability_temp: float = 1.0,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.num_stages = num_stages
        self.use_relation_pool = use_relation_pool
        self.context_radius = context_radius
        self._fallback_warn_remaining = fallback_warn_limit
        self.use_stage_phrase = use_stage_phrase
        self.use_hybrid_reliability = use_hybrid_reliability
        self.stage_phrase_temp = stage_phrase_temp
        self.hybrid_reliability_temp = hybrid_reliability_temp
        self._last_phrase_attn: Optional[torch.Tensor] = None

        self.seg_norm = nn.LayerNorm(text_dim)
        self.seg_proj = nn.Linear(text_dim, cond_dim)
        self.phrase_norm = nn.LayerNorm(text_dim)
        self.phrase_proj = nn.Linear(text_dim, cond_dim)

        if use_stage_phrase:
            self.phrase_attn_scale = text_dim ** -0.5
            self.stage_phrase_query = nn.ModuleList(
                [nn.Linear(text_dim, text_dim) for _ in range(num_stages)]
            )
            self.stage_phrase_key = nn.ModuleList(
                [nn.Linear(text_dim, text_dim) for _ in range(num_stages)]
            )
            self.stage_phrase_embed = nn.Parameter(torch.randn(num_stages, text_dim) * 0.02)
            for q, k in zip(self.stage_phrase_query, self.stage_phrase_key):
                nn.init.normal_(q.weight, std=1e-3)
                nn.init.normal_(k.weight, std=1e-3)

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

        if use_hybrid_reliability:
            self.hybrid_global_proj = nn.Linear(cond_dim, 1)
            nn.init.normal_(self.hybrid_global_proj.weight, std=1e-3)
            nn.init.zeros_(self.hybrid_global_proj.bias)

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

        q = self.rel_query(seg_feat).unsqueeze(1)
        k = self.rel_key(phrase_hidden)
        v = self.rel_value(phrase_hidden)
        attn_logits = (q * k).sum(dim=-1) * self.rel_scale
        attn_logits = attn_logits.masked_fill(~phrase_mask, float("-inf"))
        attn = torch.softmax(attn_logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        relation_ctx = torch.bmm(attn.unsqueeze(1), v).squeeze(1)
        return relation_ctx

    def _global_phrase_pool(self, phrase_hidden, phrase_mask, n_target, seg_hidden, hidden_states, seg_indices):
        """v1.5: global phrase mean-pool with local fallback."""
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
                logger.warning("[TG_SWIN] refer span missing; local pool fallback")
        else:
            phrase_ctx = torch.zeros_like(seg_hidden)
        return phrase_ctx, has_refer_span

    def _stage_phrase_attention(self, seg_hidden, phrase_hidden, phrase_mask):
        """v1.6: per-stage phrase attention → [N, S, text_dim]."""
        n_target = seg_hidden.shape[0]
        seg_ln = self.seg_norm(seg_hidden)
        phr_ln = self.phrase_norm(phrase_hidden)
        phrase_feats = []
        attn_weights = []
        for s in range(self.num_stages):
            q = self.stage_phrase_query[s](seg_ln) + self.stage_phrase_embed[s]
            k = self.stage_phrase_key[s](phr_ln)
            attn_logits = (q.unsqueeze(1) * k).sum(dim=-1) * self.phrase_attn_scale
            attn_logits = attn_logits / max(self.stage_phrase_temp, 1e-6)
            attn_logits = attn_logits.masked_fill(~phrase_mask, float("-inf"))
            attn = torch.softmax(attn_logits, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)
            h_phr_s = torch.bmm(attn.unsqueeze(1), phrase_hidden).squeeze(1)
            phrase_feats.append(h_phr_s)
            attn_weights.append(attn)
        self._last_phrase_attn = torch.stack(attn_weights, dim=1)
        return torch.stack(phrase_feats, dim=1)

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
        phrase_ctx, has_refer_span = self._global_phrase_pool(
            phrase_hidden, phrase_mask, n_target, seg_hidden, hidden_states, seg_indices
        )
        phrase_ctx = phrase_ctx.to(dtype=param_dtype)

        if self.use_stage_phrase and has_refer_span:
            stage_phrase_ctx = self._stage_phrase_attention(seg_hidden, phrase_hidden, phrase_mask)
            stage_phrase_ctx = stage_phrase_ctx.to(dtype=param_dtype)
            phrase_feat_per_stage = self.phrase_proj(self.phrase_norm(stage_phrase_ctx))
            phrase_feat = phrase_feat_per_stage.mean(dim=1)
        else:
            phrase_feat = self.phrase_proj(self.phrase_norm(phrase_ctx))
            phrase_feat_per_stage = None

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
        if phrase_feat_per_stage is not None:
            phr_exp = phrase_feat_per_stage
        else:
            phr_exp = phrase_feat.unsqueeze(1).expand(-1, self.num_stages, -1)
        rel_exp = relation_ctx.unsqueeze(1).expand(-1, self.num_stages, -1)

        mixed = gate_seg * seg_exp + gate_phr * phr_exp + gate_rel * rel_exp + stage_embed
        stage_text_cond = self.out_norm(mixed)

        local_rel = torch.sigmoid(self.reliability_head(stage_text_cond))
        if self.use_hybrid_reliability:
            global_logits = self.hybrid_global_proj(stage_text_cond).squeeze(-1)
            global_rel = F.softmax(
                global_logits / max(self.hybrid_reliability_temp, 1e-6), dim=1
            ).unsqueeze(-1)
            reliability = local_rel * global_rel
        else:
            reliability = local_rel

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
        use_stage_phrase: bool = False,
        use_hybrid_reliability: bool = False,
        stage_phrase_temp: float = 1.0,
        hybrid_reliability_temp: float = 1.0,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.version = version
        self.use_v15 = version in ("v1.5", "v1.6") or stage_router

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
                use_stage_phrase=use_stage_phrase,
                use_hybrid_reliability=use_hybrid_reliability,
                stage_phrase_temp=stage_phrase_temp,
                hybrid_reliability_temp=hybrid_reliability_temp,
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
