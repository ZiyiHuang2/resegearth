from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageGuidedCrossScaleBridge(nn.Module):
    """Multi-scale sample-level language guided cross-scale bridge."""

    def __init__(
        self,
        text_dim: int,
        feat_channels: Dict[str, int],
        enabled_scales: Tuple[str, ...] = ("res3", "res4", "res5"),
        cross_scale_init: float = 0.1,
        residual_init: float = 0.05,
        enable_long_skip: bool = True,
    ) -> None:
        super().__init__()

        if len(enabled_scales) < 2:
            raise ValueError(f"enabled_scales must contain at least 2 levels, got {enabled_scales}")

        self.enabled_scales = tuple(enabled_scales)
        self.feat_channels = feat_channels
        self.enable_long_skip = enable_long_skip

        for s in self.enabled_scales:
            if s not in feat_channels:
                raise KeyError(f"Scale {s} missing from feat_channels: {feat_channels.keys()}")

        # per-scale language gates
        self.text_proj = nn.ModuleDict({
            s: nn.Linear(text_dim, feat_channels[s]) for s in self.enabled_scales
        })

        # per-scale residual strength
        self.gamma = nn.ParameterDict({
            s: nn.Parameter(torch.tensor(residual_init, dtype=torch.float32))
            for s in self.enabled_scales
        })

        # lightweight per-scale fusion after aggregation
        self.fuse = nn.ModuleDict({
            s: nn.Conv2d(feat_channels[s], feat_channels[s], kernel_size=1, bias=False)
            for s in self.enabled_scales
        })

        # cross-scale projections
        self.cross_proj = nn.ModuleDict()
        self.cross_alpha = nn.ParameterDict()

        self._build_cross_scale_links(cross_scale_init)

    def _add_cross_proj(self, src: str, dst: str, init_value: float) -> None:
        name = f"{src}_to_{dst}"
        self.cross_proj[name] = nn.Conv2d(
            self.feat_channels[src],
            self.feat_channels[dst],
            kernel_size=1,
            bias=False,
        )
        self.cross_alpha[name] = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def _build_cross_scale_links(self, cross_scale_init: float) -> None:
        # adjacent bi-directional links
        for i in range(len(self.enabled_scales) - 1):
            src = self.enabled_scales[i]
            dst = self.enabled_scales[i + 1]
            self._add_cross_proj(src, dst, cross_scale_init)
            self._add_cross_proj(dst, src, cross_scale_init)

        # optional long skip: highest -> lowest
        if self.enable_long_skip and len(self.enabled_scales) >= 3:
            low = self.enabled_scales[0]
            high = self.enabled_scales[-1]
            self._add_cross_proj(high, low, cross_scale_init)

    @staticmethod
    def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(
        self,
        image_features: Dict[str, torch.Tensor],
        sample_guidance: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if sample_guidance.dim() != 2:
            raise ValueError(f"sample_guidance must be [B, H], got {tuple(sample_guidance.shape)}")

        for s in self.enabled_scales:
            if s not in image_features:
                raise KeyError(f"Required feature {s} not found in image_features: {image_features.keys()}")

        batch_size = sample_guidance.shape[0]
        output = dict(image_features)

        gates = {}
        for s in self.enabled_scales:
            feat = image_features[s]
            if feat.shape[0] != batch_size:
                raise ValueError(
                    f"Batch size mismatch on {s}: feat={feat.shape[0]}, guidance={batch_size}"
                )
            gates[s] = torch.sigmoid(self.text_proj[s](sample_guidance)).view(batch_size, -1, 1, 1)

        for dst in self.enabled_scales:
            feat_dst = image_features[dst]
            cross_sum = torch.zeros_like(feat_dst)

            for src in self.enabled_scales:
                if src == dst:
                    continue
                proj_key = f"{src}_to_{dst}"
                if proj_key not in self.cross_proj:
                    continue

                feat_src = image_features[src]
                feat_src = self._resize_like(feat_src, feat_dst)
                cross_feat = self.cross_proj[proj_key](feat_src)
                cross_sum = cross_sum + self.cross_alpha[proj_key] * cross_feat

            fused = self.fuse[dst](gates[dst] * feat_dst + cross_sum)
            delta = self.gamma[dst] * fused
            output[dst] = feat_dst + delta

        return output