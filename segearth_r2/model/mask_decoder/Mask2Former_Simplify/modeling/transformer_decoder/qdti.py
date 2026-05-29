"""Query-specific text-memory interaction bias for Mask2Former cross-attention (QDTI)."""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class QuerySpecificTextMemoryBias(nn.Module):
    """Per-SEG-query soft attention bias (not shared [B, S] spatial bias)."""

    def __init__(
        self,
        text_dim: int,
        memory_dim: int = 256,
        query_dim: int = 256,
        bias_dim: int = 128,
        init_std: float = 1e-3,
        max_abs: float = 0.01,
        qdti_scale_init: float = 0.0,
    ):
        super().__init__()
        self.bias_dim = int(bias_dim)
        self.init_std = float(init_std)
        self.max_abs = float(max_abs)
        self.qdti_scale = nn.Parameter(torch.tensor(float(qdti_scale_init)))
        self.visual_proj = nn.Linear(int(memory_dim), self.bias_dim)
        self.text_proj = nn.Linear(int(text_dim), self.bias_dim)
        self.query_proj = nn.Linear(int(query_dim), self.bias_dim)
        hid = max(self.bias_dim, 32)
        self.bias_mlp = nn.Sequential(
            nn.Linear(3 * self.bias_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, 1),
        )
        nn.init.normal_(self.bias_mlp[-1].weight, mean=0.0, std=float(init_std))
        nn.init.zeros_(self.bias_mlp[-1].bias)

    def forward(
        self,
        memory: torch.Tensor,
        query: torch.Tensor,
        text_memory: torch.Tensor,
        text_mask: torch.Tensor,
        num_heads: int,
    ) -> Tuple[Optional[torch.Tensor], dict]:
        """
        Args:
            memory: [S, B, C]
            query: [Q, B, Cq] decoder query states (Q from actual SEG count)
            text_memory: [B, L, D_text]
            text_mask: [B, L] True = valid
            num_heads: number of attention heads

        Returns:
            extra_attn_bias: [B * num_heads, Q, S] or None
        """
        if text_memory is None or text_mask is None:
            return None, {}
        if memory.dim() != 3 or query.dim() != 3 or text_memory.dim() != 3:
            return None, {}

        S, B, _ = memory.shape
        Q, B2, _ = query.shape
        B3, L, _ = text_memory.shape
        if B != B2 or B != B3:
            return None, {}

        d = self.bias_dim
        M = memory.permute(1, 0, 2)
        V = self.visual_proj(M)
        T = self.text_proj(text_memory)
        Qp = self.query_proj(query.permute(1, 0, 2))

        text_logits = torch.matmul(Qp, T.transpose(-1, -2)) / math.sqrt(float(d))
        text_logits = text_logits.masked_fill(~text_mask.unsqueeze(1), -1e4)
        attn = torch.softmax(text_logits, dim=-1)
        ctxt = torch.matmul(attn, T)

        V_exp = V.unsqueeze(1).expand(B, Q, S, d)
        C_exp = ctxt.unsqueeze(2).expand(B, Q, S, d)
        Z = torch.cat([V_exp, C_exp, V_exp * C_exp], dim=-1)
        raw = self.bias_mlp(Z).squeeze(-1)
        raw = raw - raw.mean(dim=-1, keepdim=True)
        P = torch.clamp(raw, -self.max_abs, self.max_abs)

        scale = self.qdti_scale.to(dtype=P.dtype, device=P.device)
        P_scaled = scale * P
        bias = P_scaled.unsqueeze(1).expand(B, num_heads, Q, S).contiguous()
        extra_attn_bias = bias.reshape(B * num_heads, Q, S)
        stats = {
            "qdti_bias_abs_mean": P_scaled.detach().abs().mean(),
            "qdti_bias_max": P_scaled.detach().max(),
            "qdti_bias_min": P_scaled.detach().min(),
            "qdti_scale": scale.detach(),
        }
        return extra_attn_bias, stats
