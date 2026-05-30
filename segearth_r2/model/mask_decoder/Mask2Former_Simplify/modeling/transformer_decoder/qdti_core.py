"""QDTI-Core: text prior -> query-specific text binding -> query-memory interaction -> P_bias_qs [B,Q,S]."""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_QDTI_CORE_DEBUG_PRINTED = False


class QDTICore(nn.Module):
    """
    Per-query decoder cross-attention bias with per-layer gate alpha_l and optional warmup.

    Injection: P = gate_eff * clamp(P_core, -max_abs, max_abs), gate_eff = alpha_l[layer] * warmup(step).
    """

    def __init__(
        self,
        text_dim: int,
        memory_dim: int = 256,
        query_dim: int = 256,
        bias_dim: int = 128,
        init_std: float = 1e-3,
        max_abs: float = 0.02,
        num_decoder_layers: int = 9,
        alpha_init: float = 0.0,
    ):
        super().__init__()
        self.bias_dim = int(bias_dim)
        self.init_std = float(init_std)
        self.max_abs = float(max_abs)
        self.num_decoder_layers = int(num_decoder_layers)
        self.text_prior_proj = nn.Linear(int(text_dim), self.bias_dim)
        self.visual_proj = nn.Linear(int(memory_dim), self.bias_dim)
        self.text_proj = nn.Linear(int(text_dim), self.bias_dim)
        self.query_proj = nn.Linear(int(query_dim), self.bias_dim)
        self.alpha_l = nn.Parameter(torch.full((self.num_decoder_layers,), float(alpha_init)))
        # Q_exp, V_exp, C_exp, prior_exp, Q*V, C*V, Q*C (query-memory + text binding)
        self._z_feat_dim = 7 * self.bias_dim
        hid = max(self.bias_dim, 32)
        self.bias_mlp = nn.Sequential(
            nn.Linear(self._z_feat_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, 1),
        )
        nn.init.normal_(self.bias_mlp[-1].weight, mean=0.0, std=float(init_std))
        nn.init.zeros_(self.bias_mlp[-1].bias)

    @staticmethod
    def warmup_factor(global_step: Optional[int], warmup_steps: int) -> float:
        if warmup_steps is None or int(warmup_steps) <= 0:
            return 1.0
        if global_step is None:
            return 0.0
        return float(min(1.0, max(0.0, float(global_step) / float(warmup_steps))))

    def _text_prior(self, text_tokens: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        mask = text_mask.unsqueeze(-1).to(dtype=text_tokens.dtype)
        denom = mask.sum(dim=1).clamp(min=1.0)
        prior_raw = (text_tokens * mask).sum(dim=1) / denom
        return self.text_prior_proj(prior_raw)

    def forward(
        self,
        memory: torch.Tensor,
        query_state: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        num_heads: int,
        layer_idx: int = 0,
        global_step: Optional[int] = None,
        warmup_steps: int = 0,
        eval_mode: str = "normal",
        force_scale: float = 1.0,
        mask_feedback: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], dict]:
        if text_tokens is None or text_mask is None:
            return None, {}
        if memory.dim() != 3 or query_state.dim() != 3 or text_tokens.dim() != 3:
            return None, {}

        S, B, _ = memory.shape
        Q, B2, _ = query_state.shape
        B3, L, _ = text_tokens.shape
        if B != B2 or B != B3 or Q < 1 or S < 1:
            return None, {}

        d = self.bias_dim
        M = memory.permute(1, 0, 2)
        V = self.visual_proj(M)
        T = self.text_proj(text_tokens)
        Qp = self.query_proj(query_state.permute(1, 0, 2))
        prior = self._text_prior(text_tokens, text_mask)

        text_logits = torch.matmul(Qp, T.transpose(-1, -2)) / math.sqrt(float(d))
        text_logits = text_logits.masked_fill(~text_mask.unsqueeze(1), -1e4)
        attn = torch.softmax(text_logits, dim=-1)
        ctxt_q = torch.matmul(attn, T)

        Q_exp = Qp.unsqueeze(2).expand(B, Q, S, d)
        prior_exp = prior.unsqueeze(1).unsqueeze(2).expand(B, Q, S, d)
        V_exp = V.unsqueeze(1).expand(B, Q, S, d)
        C_exp = ctxt_q.unsqueeze(2).expand(B, Q, S, d)
        Z = torch.cat(
            [
                Q_exp,
                V_exp,
                C_exp,
                prior_exp,
                Q_exp * V_exp,
                C_exp * V_exp,
                Q_exp * C_exp,
            ],
            dim=-1,
        )
        raw = self.bias_mlp(Z).squeeze(-1)
        raw_std = raw.float().std(dim=-1).mean() if raw.numel() > 0 else raw.new_zeros(())
        raw = raw - raw.mean(dim=-1, keepdim=True)
        P_core = torch.clamp(raw, -self.max_abs, self.max_abs)

        if mask_feedback is not None:
            if mask_feedback.shape == P_core.shape:
                mf = mask_feedback.to(dtype=P_core.dtype, device=P_core.device)
                P_core = P_core * (0.5 + 0.5 * mf)
            elif mask_feedback.dim() == 4 and mask_feedback.shape[0] == B and mask_feedback.shape[1] == Q:
                mf = F.interpolate(
                    mask_feedback.float(),
                    size=(int(math.sqrt(S)), int(math.sqrt(S))),
                    mode="bilinear",
                    align_corners=False,
                )
                if mf.shape[-2] * mf.shape[-1] == S:
                    mf = mf.flatten(2)
                    P_core = P_core * (0.5 + 0.5 * mf.to(dtype=P_core.dtype, device=P_core.device))

        li = int(layer_idx)
        if li < 0 or li >= self.num_decoder_layers:
            li = max(0, min(li, self.num_decoder_layers - 1))
        alpha = self.alpha_l[li].to(dtype=P_core.dtype, device=P_core.device)
        wu = self.warmup_factor(global_step, warmup_steps)
        gate_eff = alpha * P_core.new_tensor(wu)

        em = str(eval_mode if eval_mode is not None else "normal").strip().lower()
        if em not in ("normal", "bypass", "force_scale"):
            em = "normal"
        if em == "bypass":
            P = torch.zeros_like(P_core)
            gate_eff = gate_eff.detach() * 0.0
        elif em == "force_scale":
            fs = float(force_scale)
            P = gate_eff * float(fs) * P_core
        else:
            P = gate_eff * P_core

        ma = float(self.max_abs)
        if P.dtype == torch.float16 and ma >= 0.1:
            raise AssertionError(
                f"decoder_attn_bias_max_abs={ma} too large for fp16 hard-mask safety; use <=0.02."
            )

        bias = P.unsqueeze(1).expand(B, num_heads, Q, S).contiguous()
        attn_bias = bias.reshape(B * num_heads, Q, S)
        stats = {
            "qdti_bias_abs_mean": P.detach().abs().mean(),
            "qdti_bias_raw_std": raw_std.detach() if torch.is_tensor(raw_std) else raw_std,
            "qdti_bias_max": P.detach().max(),
            "qdti_bias_min": P.detach().min(),
            "qdti_enabled": memory.new_tensor(1.0),
            "qdti_alpha_l": alpha.detach(),
            "qdti_gate_eff": gate_eff.detach() if torch.is_tensor(gate_eff) else gate_eff,
            "qdti_warmup_factor": P.new_tensor(wu),
            "qdti_layer_idx": P.new_tensor(float(li)),
            "P_bias_qs": P,
            "P_bias_flat": P,
        }

        global _QDTI_CORE_DEBUG_PRINTED
        if not _QDTI_CORE_DEBUG_PRINTED:
            _QDTI_CORE_DEBUG_PRINTED = True
            print(
                "[DEBUG][QDTICore] "
                f"P_bias_qs={tuple(P.shape)} attn_bias={tuple(attn_bias.shape)} "
                f"layer={li} alpha_l={float(alpha.detach().item()):.6g} "
                f"warmup={wu:.6g} gate_eff={float(gate_eff.detach().item()) if gate_eff.numel()==1 else 'tensor'} "
                f"P_abs_mean={float(P.detach().abs().mean().item()):.6g}",
                flush=True,
            )
        return attn_bias, stats
