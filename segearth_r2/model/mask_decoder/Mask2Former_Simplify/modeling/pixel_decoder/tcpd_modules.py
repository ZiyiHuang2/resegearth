"""SEG-conditioned TCPD-v2 modules (Target-Conditioned Pixel Decoder).

Condition source is projected [SEG] embedding only (dim 256). Gates initialize to
zero for identity forward; branch weights use standard init so gradients flow once
gates become nonzero.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_


class TargetConditionEncoder(nn.Module):
    """Encode projected [SEG] embedding into shared condition vector z_k."""

    def __init__(self, dim: int = 256):
        super().__init__()
        self.norm_in = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm_out = nn.LayerNorm(dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, seg_embedding: torch.Tensor) -> torch.Tensor:
        if seg_embedding.dim() == 3:
            seg_embedding = seg_embedding.squeeze(1)
        x = self.norm_in(seg_embedding)
        x = F.gelu(self.fc1(x))
        delta = self.fc2(x)
        return self.norm_out(seg_embedding + delta)


class TCPDFPNLevelCond(nn.Module):
    """Text-conditioned residual on one FPN top-down fusion step."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.gate_td = nn.Parameter(torch.zeros(1))
        self.gate_lat = nn.Parameter(torch.zeros(1))
        self.alpha_mlp = nn.Linear(hidden_dim, hidden_dim)
        self.td_adapter = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False)
        self.lat_adapter = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False)
        xavier_uniform_(self.td_adapter.weight)
        xavier_uniform_(self.lat_adapter.weight)
        nn.init.zeros_(self.alpha_mlp.weight)
        nn.init.zeros_(self.alpha_mlp.bias)

    def forward(self, cur: torch.Tensor, td: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Baseline y = cur + td, plus gated residual modulation."""
        y = cur + td
        alpha = torch.sigmoid(self.alpha_mlp(z)).view(-1, z.shape[-1], 1, 1)
        td_delta = self.td_adapter(td)
        lat_delta = self.lat_adapter(cur)
        y = y + self.gate_td * alpha * td_delta + self.gate_lat * (1.0 - alpha) * lat_delta
        return y


class TCPDFPNFusion(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_levels: int = 1):
        super().__init__()
        self.levels = nn.ModuleList([
            TCPDFPNLevelCond(hidden_dim) for _ in range(num_levels)
        ])

    def forward(self, cur: torch.Tensor, td: torch.Tensor, z: torch.Tensor, level_idx: int = 0) -> torch.Tensor:
        return self.levels[level_idx](cur, td, z)


class TCPDScaleFusion(nn.Module):
    """Per-target multi-scale / level fusion after pixel decoder outputs."""

    def __init__(self, hidden_dim: int = 256, num_levels: int = 3):
        super().__init__()
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        self.gate_scale = nn.Parameter(torch.zeros(1))
        self.level_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, num_levels),
        )
        nn.init.zeros_(self.level_mlp[-1].weight)
        nn.init.zeros_(self.level_mlp[-1].bias)

        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                nn.GroupNorm(32, hidden_dim),
            )
            for _ in range(num_levels)
        ])
        for adapter in self.adapters:
            xavier_uniform_(adapter[0].weight)

        self.tcpd_mask_adapter = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, hidden_dim),
        )
        xavier_uniform_(self.tcpd_mask_adapter[0].weight)
        self.tcpd_gate_mask = nn.Parameter(torch.zeros(1))

    def get_tcpd_level_weights(self, z: torch.Tensor) -> torch.Tensor:
        """Return softmax level weights [N, num_levels] for probes."""
        return F.softmax(self.level_mlp(z), dim=-1)

    def forward(
        self,
        multi_scale_features,
        mask_features: torch.Tensor,
        z: torch.Tensor,
        enabled: bool = True,
    ):
        if not enabled:
            return multi_scale_features, mask_features

        level_weights = self.get_tcpd_level_weights(z)

        modulated_ms = []
        for level_idx, feat in enumerate(multi_scale_features):
            w = level_weights[:, level_idx].view(-1, 1, 1, 1)
            adapted = self.adapters[level_idx](feat)
            modulated_ms.append(feat + self.gate_scale * w * adapted)

        mask_adapted = self.tcpd_mask_adapter(mask_features)
        mask_out = mask_features + self.tcpd_gate_mask * mask_adapted
        return modulated_ms, mask_out
