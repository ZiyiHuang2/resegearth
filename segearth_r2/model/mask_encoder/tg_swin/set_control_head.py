"""Target-set stage-wise modulation for TG-Swin bias gate."""

from __future__ import annotations

import torch
import torch.nn as nn


class SETControlHead(nn.Module):
    """
    Target-set stage-wise modulation for TG-Swin bias gate.

    Input:  set_hidden [T, D] — SET projected embedding (per target, after repeat)
    Output: set_control [T, S, 1] — per-target, per-text-stage scalar modulation
    """

    def __init__(
        self,
        text_dim: int,
        num_stages: int,
        init_bias: float = 4.0,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.num_stages = num_stages
        self.set_norm = nn.LayerNorm(text_dim)
        self.stage_heads = nn.ModuleList(
            [nn.Linear(text_dim, 1) for _ in range(num_stages)]
        )
        for head in self.stage_heads:
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, init_bias)

    def forward(self, set_hidden: torch.Tensor) -> torch.Tensor:
        """set_hidden [T, D] -> set_control [T, S, 1]."""
        x = self.set_norm(set_hidden)
        controls = [torch.sigmoid(head(x)) for head in self.stage_heads]
        return torch.stack(controls, dim=1)
