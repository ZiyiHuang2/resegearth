"""Window Text Interaction — v1 key-bias / v1.5 head-aware / DR-EWTI dynamic relational."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def _scalar_param(value: float) -> nn.Parameter:
    """1-element parameter for HF safetensors / meta-init compatibility."""
    return nn.Parameter(torch.tensor([value], dtype=torch.float32))


def _alpha_gate_term(alpha: torch.Tensor, gate_floor: float) -> torch.Tensor:
    """Non-zero gate floor so main loss can backprop through raw_bias at alpha=0."""
    return gate_floor + torch.tanh(alpha)


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
        gate_floor: float = 0.05,
    ):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.num_heads = num_heads
        self.N = window_size * window_size
        self.bias_max = bias_max
        self.stage_scale = stage_scale
        self.gate_floor = gate_floor
        self.text_proj = nn.Linear(cond_dim, rank, bias=False)
        self.visual_proj = nn.Linear(dim, rank, bias=False)
        self.score_proj = nn.Linear(rank, self.N, bias=False)
        nn.init.normal_(self.score_proj.weight, std=1e-3)
        self.alpha = _scalar_param(alpha_init)

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
        gate = _alpha_gate_term(self.alpha, self.gate_floor) * self.stage_scale * rel
        return (raw * gate).unsqueeze(1).unsqueeze(1), raw, gate

    def stats(self, attn_bias, raw_bias, gate, reliability):
        alpha_term = _alpha_gate_term(self.alpha, self.gate_floor)
        return {
            "alpha": float(torch.tanh(self.alpha).item()),
            "alpha_term": float(alpha_term.item()),
            "gate_abs_mean": float(gate.abs().mean().item()) if gate is not None else 0.0,
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
        gate_floor: float = 0.05,
    ):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.num_heads = num_heads
        self.N = window_size * window_size
        self.rank = rank
        self.bias_max = bias_max
        self.stage_scale = stage_scale
        self.gate_floor = gate_floor
        hr = num_heads * rank

        self.visual_q = nn.Linear(dim, hr, bias=False)
        self.visual_k = nn.Linear(dim, hr, bias=False)
        self.text_q = nn.Linear(cond_dim, hr, bias=False)
        self.text_k = nn.Linear(cond_dim, hr, bias=False)
        for m in (self.visual_q, self.visual_k, self.text_q, self.text_k):
            nn.init.normal_(m.weight, std=1e-3)

        self.alpha = _scalar_param(alpha_init)
        self.state_proj: Optional[nn.Linear] = None

    def set_state_proj(self, state_proj: nn.Linear):
        self.state_proj = state_proj

    def _apply_text_cond(self, text_cond: torch.Tensor, evidence_state: Optional[torch.Tensor]) -> torch.Tensor:
        if evidence_state is None or self.state_proj is None:
            return text_cond
        return text_cond + self.state_proj(evidence_state)

    def compute_raw_bias(
        self,
        x_windows: torch.Tensor,
        text_cond: torch.Tensor,
        reliability: torch.Tensor,
        evidence_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bw, n_tokens, _ = x_windows.shape
        batch_size = text_cond.shape[0]
        H, r = self.num_heads, self.rank
        num_windows = bw // batch_size

        tc = self._apply_text_cond(text_cond, evidence_state)
        tc = tc.unsqueeze(1).expand(batch_size, num_windows, -1).reshape(bw, -1)

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
        evidence_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bw = x_windows.shape[0]
        batch_size = text_cond.shape[0]
        num_windows = bw // batch_size
        rel = reliability.view(batch_size, -1)[:, :1]
        rel = rel.unsqueeze(1).expand(batch_size, num_windows, 1).reshape(bw, 1, 1, 1)

        raw_bias = self.compute_raw_bias(x_windows, text_cond, reliability, evidence_state=evidence_state)
        gate = _alpha_gate_term(self.alpha, self.gate_floor) * self.stage_scale * rel
        attn_bias = raw_bias * gate
        return attn_bias, raw_bias, gate

    def stats(self, attn_bias, raw_bias, gate, reliability):
        head_mean = attn_bias.abs().mean(dim=(0, 2, 3)) if attn_bias is not None else None
        ent = 0.0
        if head_mean is not None and head_mean.sum() > 0:
            p = head_mean / head_mean.sum().clamp_min(1e-8)
            ent = float(-(p * (p + 1e-8).log()).sum().item())
        alpha_term = _alpha_gate_term(self.alpha, self.gate_floor)
        return {
            "alpha": float(torch.tanh(self.alpha).item()),
            "alpha_term": float(alpha_term.item()),
            "gate_abs_mean": float(gate.abs().mean().item()) if gate is not None else 0.0,
            "bias_abs_mean": float(attn_bias.abs().mean().item()) if attn_bias is not None else 0.0,
            "bias_abs_max": float(attn_bias.abs().max().item()) if attn_bias is not None else 0.0,
            "raw_bias_abs_mean": float(raw_bias.abs().mean().item()) if raw_bias is not None else 0.0,
            "reliability_mean": float(reliability.mean().item()) if reliability is not None else 0.0,
            "head_bias_entropy": ent,
        }


class StageEnhancedWTIHeadAware(nn.Module):
    """Enhanced WTI v2: MLP visual/text projectors + optional head mixer → [BW,H,N,N] bias."""

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
        projector_hidden_ratio: float = 2.0,
        projector_max_hidden: int = 512,
        use_head_mixer: bool = True,
        head_mixer_ratio: float = 2.0,
        gamma_init: float = 0.0,
        gate_mode: str = "legacy",
        bias_scale_init: float = 0.1,
        gate_floor: float = 0.05,
        fixed_stage_bias_scale: float = 0.05,
    ):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.num_heads = num_heads
        self.N = window_size * window_size
        self.rank = rank
        self.bias_max = bias_max
        self.stage_scale = stage_scale
        self.gate_floor = gate_floor
        self.fixed_stage_bias_scale = float(fixed_stage_bias_scale)
        self.use_head_mixer = use_head_mixer
        self.gate_mode = str(gate_mode).lower()
        H, r = num_heads, rank
        hr = H * r

        hidden_v = min(int(dim * projector_hidden_ratio), projector_max_hidden)
        hidden_t = min(int(cond_dim * projector_hidden_ratio), projector_max_hidden)

        self.visual_norm = nn.LayerNorm(dim)
        self.visual_proj = nn.Sequential(
            nn.Linear(dim, hidden_v),
            nn.GELU(),
            nn.Linear(hidden_v, 2 * hr),
        )
        self.text_norm = nn.LayerNorm(cond_dim)
        self.text_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_t),
            nn.GELU(),
            nn.Linear(hidden_t, 2 * hr),
        )
        for seq in (self.visual_proj, self.text_proj):
            nn.init.normal_(seq[-1].weight, std=1e-3)
            if seq[-1].bias is not None:
                nn.init.zeros_(seq[-1].bias)

        if use_head_mixer:
            mixer_hidden = max(int(num_heads * head_mixer_ratio), num_heads)
            self.head_mixer_norm = nn.LayerNorm(num_heads)
            self.head_mixer = nn.Sequential(
                nn.Linear(num_heads, mixer_hidden),
                nn.GELU(),
                nn.Linear(mixer_hidden, num_heads),
            )
            self.head_mixer_gamma = _scalar_param(gamma_init)
        else:
            self.head_mixer_norm = None
            self.head_mixer = None
            self.head_mixer_gamma = None

        self.alpha = _scalar_param(alpha_init)

        if self.gate_mode == "set_lite":
            init = torch.tensor(bias_scale_init, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
            self.bias_scale_logit = nn.Parameter(torch.logit(init))
        else:
            self.bias_scale_logit = None

    @staticmethod
    def _broadcast_text_cond(text_cond: torch.Tensor, bw: int) -> torch.Tensor:
        batch_size = text_cond.shape[0]
        num_windows = bw // batch_size
        return text_cond.unsqueeze(1).expand(batch_size, num_windows, -1).reshape(bw, -1)

    def compute_raw_bias(
        self,
        x_windows: torch.Tensor,
        text_cond: torch.Tensor,
        reliability: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bw, n_tokens, _ = x_windows.shape
        H, r = self.num_heads, self.rank

        tc = self._broadcast_text_cond(text_cond, bw)

        v = self.visual_proj(self.visual_norm(x_windows)).view(bw, n_tokens, H, 2, r)
        Q_v = v[..., 0, :].permute(0, 2, 1, 3)
        K_v = v[..., 1, :].permute(0, 2, 1, 3)

        t = self.text_proj(self.text_norm(tc)).view(bw, H, 2, r)
        Q_t, K_t = t[:, :, 0, :], t[:, :, 1, :]

        Q_prime = Q_v * Q_t.unsqueeze(2)
        K_prime = K_v * K_t.unsqueeze(2)

        B_0 = torch.einsum("bhnr,bhmr->bhnm", Q_prime, K_prime) / math.sqrt(r)

        if self.use_head_mixer and self.head_mixer is not None:
            if self.training:
                from torch.utils.checkpoint import checkpoint
                B_mix = checkpoint(self._head_mixer_mix, B_0, use_reentrant=False)
            else:
                B_mix = self._head_mixer_mix(B_0)
        else:
            B_mix = B_0

        return torch.tanh(B_mix) * self.bias_max

    def _head_mixer_mix(self, B_0: torch.Tensor) -> torch.Tensor:
        bias_h = B_0.permute(0, 2, 3, 1)
        bias_h = self.head_mixer(self.head_mixer_norm(bias_h))
        bias_delta = bias_h.permute(0, 3, 1, 2)
        gamma = torch.tanh(self.head_mixer_gamma)
        return B_0 + gamma * bias_delta

    def forward(
        self,
        x_windows: torch.Tensor,
        text_cond: torch.Tensor,
        reliability: torch.Tensor,
        set_control: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_bias = self.compute_raw_bias(x_windows, text_cond, reliability)

        if self.gate_mode == "set_lite":
            global_scale = torch.sigmoid(self.bias_scale_logit)

            if set_control is not None:
                set_mod = 0.5 + set_control
            else:
                set_mod = 1.0

            gate = global_scale * set_mod
            attn_bias = raw_bias * gate
        elif self.gate_mode == "no_alpha_fixed":
            bw = x_windows.shape[0]
            batch_size = text_cond.shape[0]
            num_windows = bw // batch_size
            rel = reliability.view(batch_size, -1)[:, :1]
            rel = rel.unsqueeze(1).expand(batch_size, num_windows, 1).reshape(bw, 1, 1, 1)

            beta_s = self.fixed_stage_bias_scale
            gate = beta_s * rel
            if set_control is not None:
                gate = gate * set_control
            attn_bias = raw_bias * gate
        else:
            bw = x_windows.shape[0]
            batch_size = text_cond.shape[0]
            num_windows = bw // batch_size
            rel = reliability.view(batch_size, -1)[:, :1]
            rel = rel.unsqueeze(1).expand(batch_size, num_windows, 1).reshape(bw, 1, 1, 1)

            alpha_term = _alpha_gate_term(self.alpha, self.gate_floor)
            gate = alpha_term * self.stage_scale * rel
            if set_control is not None:
                gate = gate * set_control
            attn_bias = raw_bias * gate
        return attn_bias, raw_bias, gate

    def stats(self, attn_bias, raw_bias, gate, reliability, set_control=None):
        head_mean = attn_bias.abs().mean(dim=(0, 2, 3)) if attn_bias is not None else None
        ent = 0.0
        if head_mean is not None and head_mean.sum() > 0:
            p = head_mean / head_mean.sum().clamp_min(1e-8)
            ent = float(-(p * (p + 1e-8).log()).sum().item())
        mixer_gamma = 0.0
        if self.head_mixer_gamma is not None:
            mixer_gamma = float(torch.tanh(self.head_mixer_gamma).item())
        st = {
            "gate_mode": self.gate_mode,
            "gate_abs_mean": float(gate.abs().mean().item()) if gate is not None else 0.0,
            "head_mixer_gamma": mixer_gamma,
            "bias_abs_mean": float(attn_bias.abs().mean().item()) if attn_bias is not None else 0.0,
            "attn_bias_abs_mean": float(attn_bias.abs().mean().item()) if attn_bias is not None else 0.0,
            "bias_abs_max": float(attn_bias.abs().max().item()) if attn_bias is not None else 0.0,
            "raw_bias_abs_mean": float(raw_bias.abs().mean().item()) if raw_bias is not None else 0.0,
            "reliability_mean": float(reliability.mean().item()) if reliability is not None else 0.0,
            "head_bias_entropy": ent,
        }
        if self.gate_mode == "no_alpha_fixed":
            st["beta_s"] = self.fixed_stage_bias_scale
        else:
            alpha_term = _alpha_gate_term(self.alpha, self.gate_floor)
            st["alpha"] = float(torch.tanh(self.alpha).item())
            st["alpha_term"] = float(alpha_term.item())
        if self.gate_mode == "set_lite" and self.bias_scale_logit is not None:
            st["bias_scale"] = float(torch.sigmoid(self.bias_scale_logit).item())
        if set_control is not None:
            st["set_control_mean"] = float(set_control.detach().mean().cpu())
            st["set_gate_mean"] = float(set_control.detach().mean().cpu())
        return st


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
        use_evidence_state: bool = False,
        evidence_state_dim: int = 64,
        use_dr_ewti: bool = False,
        evidence_dim: int = 32,
        evidence_eps: float = 1e-4,
        evidence_logit_clip: float = 4.0,
        evidence_relation_gate_init: float = 0.0,
        enhanced_wti: bool = False,
        wti_projector_ratio: float = 2.0,
        wti_projector_max_hidden: int = 512,
        wti_head_mixer: bool = True,
        wti_head_mixer_ratio: float = 2.0,
        wti_head_mixer_gamma_init: float = 0.0,
        gate_mode: str = "legacy",
        gate_floor: float = 0.05,
        fixed_stage_bias_scales: Optional[List[float]] = None,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.wti_stage_list = sorted(wti_stages if wti_stages is not None else [1, 2, 3])
        self.wti_stages = set(self.wti_stage_list)
        self.wti_start_layer = wti_start_layer
        self.window_size = window_size
        self.log_stats = log_stats
        self.head_aware = head_aware
        self.enhanced_wti = enhanced_wti
        self.gate_mode = str(gate_mode).lower()
        self.gate_floor = gate_floor
        if fixed_stage_bias_scales is None:
            fixed_stage_bias_scales = [0.05] * len(self.wti_stage_list)
        self.fixed_stage_bias_scales = list(fixed_stage_bias_scales)
        self.num_text_stages = num_text_stages
        self.use_evidence_state = use_evidence_state
        self.evidence_state_dim = evidence_state_dim
        self.use_dr_ewti = use_dr_ewti and not enhanced_wti
        self._last_stats: Dict[str, float] = {}
        self._stage_bias_abs_mean: Dict[int, float] = {}
        self._stage_dims_map: Dict[int, int] = {}

        if swin_type == "large":
            stage_dims = [192, 384, 768, 1536]
            stage_heads = [6, 12, 24, 48]
        else:
            stage_dims = self.STAGE_DIMS_BASE
            stage_heads = self.STAGE_HEADS_BASE

        if stage_scales is None:
            stage_scales = [0.25, 0.5, 0.75, 1.0]

        if use_evidence_state:
            self.state_update = nn.Sequential(
                nn.Linear(evidence_state_dim * 2, evidence_state_dim),
                nn.LayerNorm(evidence_state_dim),
                nn.GELU(),
                nn.Linear(evidence_state_dim, evidence_state_dim),
            )
            nn.init.zeros_(self.state_update[-1].weight)
            nn.init.zeros_(self.state_update[-1].bias)
            self.state_proj = nn.Linear(evidence_state_dim, cond_dim)
            nn.init.normal_(self.state_proj.weight, std=1e-3)
            # Pre-register per-stage pool projs so .cuda() / state_dict cover them
            self.visual_pool_projs = nn.ModuleDict()
            for stage_idx, stage_dim in enumerate(stage_dims):
                key = str(stage_idx)
                proj = nn.Linear(stage_dim, evidence_state_dim)
                nn.init.normal_(proj.weight, std=1e-3)
                nn.init.zeros_(proj.bias)
                self.visual_pool_projs[key] = proj
        else:
            self.state_update = None
            self.state_proj = None
            self.visual_pool_projs = None

        if enhanced_wti:
            wti_cls = StageEnhancedWTIHeadAware
        elif head_aware:
            wti_cls = StageWTIHeadAware
        else:
            wti_cls = StageWTIKeyBias
        self.wti_blocks = nn.ModuleDict()
        self.dr_wti_blocks: Optional[nn.ModuleDict] = (
            nn.ModuleDict() if self.use_dr_ewti else None
        )
        for stage_idx in self.wti_stages:
            if stage_idx >= len(stage_dims):
                continue
            self._stage_dims_map[stage_idx] = stage_dims[stage_idx]
            stage_pos = self.wti_stage_list.index(stage_idx)
            block_kwargs = dict(
                dim=stage_dims[stage_idx],
                cond_dim=cond_dim,
                num_heads=stage_heads[stage_idx],
                window_size=window_size,
                rank=wti_rank,
                bias_max=bias_max,
                alpha_init=alpha_init,
                stage_scale=stage_scales[stage_idx] if stage_idx < len(stage_scales) else 1.0,
                gate_floor=gate_floor,
            )
            if enhanced_wti:
                fixed_scale = (
                    self.fixed_stage_bias_scales[stage_pos]
                    if stage_pos < len(self.fixed_stage_bias_scales)
                    else self.fixed_stage_bias_scales[-1]
                )
                block_kwargs.update(
                    projector_hidden_ratio=wti_projector_ratio,
                    projector_max_hidden=wti_projector_max_hidden,
                    use_head_mixer=wti_head_mixer,
                    head_mixer_ratio=wti_head_mixer_ratio,
                    gamma_init=wti_head_mixer_gamma_init,
                    gate_mode=self.gate_mode,
                    fixed_stage_bias_scale=fixed_scale,
                )
            block = wti_cls(**block_kwargs)
            if use_evidence_state and self.state_proj is not None and hasattr(block, "set_state_proj"):
                block.set_state_proj(self.state_proj)
            self.wti_blocks[str(stage_idx)] = block
            if self.use_dr_ewti and isinstance(block, StageWTIHeadAware):
                from .dynamic_relational_wti import StageDynamicRelationalWTI

                assert self.dr_wti_blocks is not None
                self.dr_wti_blocks[str(stage_idx)] = StageDynamicRelationalWTI(
                    dim=stage_dims[stage_idx],
                    cond_dim=cond_dim,
                    num_heads=stage_heads[stage_idx],
                    rank=wti_rank,
                    bias_max=bias_max,
                    evidence_dim=evidence_dim,
                    eps=evidence_eps,
                    logit_clip=evidence_logit_clip,
                    evidence_relation_gate_init=evidence_relation_gate_init,
                )

    def _cond_index_for_swin_stage(self, swin_stage_idx: int, num_cond_stages: int) -> int:
        """Compact layout: cond[i] ↔ wti_stage_list[i]. Legacy: cond indexed by Swin stage id."""
        if num_cond_stages == len(self.wti_stage_list):
            return self.wti_stage_list.index(swin_stage_idx)
        return min(swin_stage_idx, num_cond_stages - 1)

    def _select_stage_cond(
        self,
        text_cond: torch.Tensor,
        reliability: torch.Tensor,
        swin_stage_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if text_cond.dim() == 3:
            s = self._cond_index_for_swin_stage(swin_stage_idx, text_cond.shape[1])
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
        evidence_state: Optional[torch.Tensor] = None,
        evidence_windows: Optional[torch.Tensor] = None,
        set_control: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if text_cond is None or reliability is None:
            return None
        if not self.enabled_for(stage_idx, layer_idx):
            return None
        key = str(stage_idx)
        if key not in self.wti_blocks:
            return None

        tc, rel = self._select_stage_cond(text_cond, reliability, stage_idx)
        sc = None
        if set_control is not None:
            sc, _ = self._select_stage_cond(set_control, set_control, stage_idx)
            bw = x_windows.shape[0]
            batch_size = tc.shape[0]
            num_windows = bw // batch_size
            sc = sc.view(batch_size, -1)[:, :1]
            sc = sc.unsqueeze(1).expand(batch_size, num_windows, 1).reshape(bw, 1, 1, 1)

        module = self.wti_blocks[key]
        state_in = evidence_state
        if self.use_evidence_state and state_in is None:
            batch_size = tc.shape[0]
            state_in = torch.zeros(
                batch_size, self.evidence_state_dim,
                device=tc.device, dtype=tc.dtype,
            )

        diag: Dict[str, float] = {}
        if (
            self.use_dr_ewti
            and self.dr_wti_blocks is not None
            and key in self.dr_wti_blocks
            and evidence_windows is not None
        ):
            dr_module = self.dr_wti_blocks[key]
            attn_bias, raw_bias, gate, dr_diag = dr_module(
                module,
                x_windows,
                tc,
                rel,
                evidence_windows=evidence_windows,
                log_stats=self.log_stats,
            )
            if dr_diag is not None:
                diag = dr_diag
        else:
            if isinstance(module, StageEnhancedWTIHeadAware):
                attn_bias, raw_bias, gate = module(x_windows, tc, rel, set_control=sc)
            else:
                attn_bias, raw_bias, gate = module(x_windows, tc, rel, evidence_state=state_in)

        if self.use_evidence_state and self.log_stats:
            self._stage_bias_abs_mean[stage_idx] = float(attn_bias.detach().abs().mean().cpu())

        if self.log_stats:
            if not diag:
                if isinstance(module, StageEnhancedWTIHeadAware):
                    st = module.stats(attn_bias, raw_bias, gate, rel, set_control=sc)
                else:
                    st = module.stats(attn_bias, raw_bias, gate, rel)
            else:
                st = dict(diag)
            st["stage_idx"] = float(stage_idx)
            st["layer_idx"] = float(layer_idx)
            st["active"] = 1.0
            if "reliability_mean" not in st:
                st["reliability_mean"] = float(rel.detach().mean().cpu())
            if sc is not None and "set_control_mean" not in st:
                st["set_control_mean"] = float(sc.detach().mean().cpu())
                st["set_gate_mean"] = float(sc.detach().mean().cpu())
            if isinstance(module, StageEnhancedWTIHeadAware) and module.gate_mode == "no_alpha_fixed":
                st["beta_s"] = module.fixed_stage_bias_scale
            if "attn_bias_abs_mean" not in st and "bias_abs_mean" in st:
                st["attn_bias_abs_mean"] = st["bias_abs_mean"]
            self._last_stats.update(st)
        return attn_bias

    def update_evidence_state(
        self,
        stage_idx: int,
        x_stage: torch.Tensor,
        prev_state: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Update cross-stage evidence state at stage boundary."""
        if not self.use_evidence_state or self.state_update is None:
            return prev_state

        batch_size = x_stage.shape[0]
        if prev_state is None:
            prev_state = torch.zeros(
                batch_size, self.evidence_state_dim,
                device=x_stage.device, dtype=x_stage.dtype,
            )

        key = str(stage_idx)
        if key not in self.visual_pool_projs:
            return prev_state

        visual_pool = x_stage.mean(dim=1)
        proj = self.visual_pool_projs[key]
        if proj.weight.device != visual_pool.device:
            proj.to(device=visual_pool.device, dtype=visual_pool.dtype)
        pool_proj = proj(visual_pool)
        state_in = torch.cat([prev_state, pool_proj], dim=-1)
        if next(self.state_update.parameters()).device != visual_pool.device:
            self.state_update.to(device=visual_pool.device, dtype=visual_pool.dtype)
        delta = self.state_update(state_in)
        return prev_state + delta

    def pop_stats(self) -> Dict[str, float]:
        stats = dict(self._last_stats)
        self._last_stats = {}
        return stats
