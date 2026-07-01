"""Target-set stage-wise modulation for TG-Swin bias gate."""

from __future__ import annotations

import torch
import torch.nn as nn


class SETControlHead(nn.Module):
    """
    Image-level SET stage-wise modulation for TG-Swin bias gate.

    Input:  set_hidden [B, D] — SET projected embedding at image batch B
    Output: set_gate_b [B, S, 1] — per-image, per-text-stage scalar modulation
    Broadcast to targets in llava_phi: set_gate_t = set_gate_b[target_to_image]
    """

    def __init__(
        self,
        text_dim: int,
        num_stages: int,
        init_bias: float = 0.0,
        weight_gain: float = 0.01,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.num_stages = num_stages
        self.set_norm = nn.LayerNorm(text_dim)
        self.stage_heads = nn.ModuleList(
            [nn.Linear(text_dim, 1) for _ in range(num_stages)]
        )
        for head in self.stage_heads:
            nn.init.xavier_uniform_(head.weight, gain=weight_gain)
            nn.init.constant_(head.bias, init_bias)

    def forward(self, set_hidden: torch.Tensor) -> torch.Tensor:
        """set_hidden [B, D] or [B, 1, D] -> set_gate_b [B, S, 1]."""
        if set_hidden.dim() == 3:
            assert set_hidden.shape[1] == 1, (
                f"SETControlHead expects [B,1,D] or [B,D], got {tuple(set_hidden.shape)}"
            )
            set_hidden = set_hidden.squeeze(1)
        assert set_hidden.dim() == 2, (
            f"SETControlHead expects [B,D], got {tuple(set_hidden.shape)}"
        )
        x = self.set_norm(set_hidden)
        controls = [torch.sigmoid(head(x)) for head in self.stage_heads]
        return torch.stack(controls, dim=1)
