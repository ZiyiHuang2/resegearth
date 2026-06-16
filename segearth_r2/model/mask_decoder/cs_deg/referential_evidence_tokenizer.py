"""RET: Referential Evidence Tokenizer for CS-DEG++."""
from __future__ import annotations

import torch
import torch.nn as nn


class ReferentialEvidenceTokenizer(nn.Module):
    """
    Generate layer-wise evidence tokens from decoder state.

    v1 uses pooled decoder output only (no phrase hidden states).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        prompt_token_num: int = 4,
        num_layers: int = 10,
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.prompt_token_num = prompt_token_num

        self.prompt_tokens = nn.Parameter(torch.zeros(prompt_token_num, hidden_dim))
        self.layer_embed = nn.Embedding(num_layers, hidden_dim)
        self.delta_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, prompt_token_num * hidden_dim),
        )
        self.ret_gate = nn.Parameter(torch.tensor([gate_init], dtype=torch.float32))
        nn.init.zeros_(self.prompt_tokens)
        nn.init.zeros_(self.delta_proj[-1].weight)
        nn.init.zeros_(self.delta_proj[-1].bias)

    def forward(
        self,
        output: torch.Tensor,
        layer_index: int,
        seg_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        output: [Q, B, C]
        Returns evidence_tokens: [P, B, C]
        """
        del seg_embedding  # reserved for future phrase-state fusion
        q, b, c = output.shape
        pooled = output.transpose(0, 1).mean(dim=1)  # [B, C]
        delta = self.delta_proj(pooled).view(b, self.prompt_token_num, c).transpose(0, 1)
        layer_bias = self.layer_embed.weight[layer_index % self.layer_embed.num_embeddings]
        base = self.prompt_tokens.unsqueeze(1).expand(-1, b, -1)
        tokens = base + self.ret_gate * (delta + layer_bias.view(1, 1, c))
        if q > 1:
            tokens = tokens + output.mean(dim=0, keepdim=True) * 0.0
        return tokens
