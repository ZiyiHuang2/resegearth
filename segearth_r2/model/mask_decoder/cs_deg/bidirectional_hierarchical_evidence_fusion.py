"""BHEF: Bidirectional Hierarchical Evidence Fusion for CS-DEG++."""
from __future__ import annotations

import torch
import torch.nn as nn


class BidirectionalHierarchicalEvidenceFusion(nn.Module):
    """
    Bidirectional interaction between evidence tokens and decoder query
    before cross-attention. Does not modify visual memory (src).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.token_cross = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=False)
        self.query_cross = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=False)
        self.bhef_gate = nn.Parameter(torch.tensor([gate_init], dtype=torch.float32))

    def forward(
        self,
        output: torch.Tensor,
        src: torch.Tensor,
        evidence_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        output: [Q, B, C]
        src: [HW, B, C]
        evidence_tokens: [P, B, C]
        """
        tokens_refined, _ = self.token_cross(
            evidence_tokens, src, src, need_weights=False
        )
        output_delta, _ = self.query_cross(
            output, tokens_refined, tokens_refined, need_weights=False
        )
        output_refined = output + self.bhef_gate * output_delta
        return output_refined, tokens_refined
