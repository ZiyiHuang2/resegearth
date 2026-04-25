from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageGuidedCrossScaleBridge(nn.Module):
    """Lightweight sentence-guided dual-scale bridge for res3/res4 (lgce_variant=rebuild_sentence).

    Reconstructs the sentence_mean guidance path from current code; not asserted to match any historical checkpoint.
    """

    def __init__(
        self,
        text_dim: int,
        res3_channels: int,
        res4_channels: int,
        residual_init: float = 0.05,
        cross_scale_init: float = 0.1,
    ) -> None:
        super().__init__()

        self.text_proj_res3 = nn.Linear(text_dim, res3_channels)
        self.text_proj_res4 = nn.Linear(text_dim, res4_channels)

        self.res4_to_res3 = nn.Conv2d(res4_channels, res3_channels, kernel_size=1, bias=False)
        self.res3_to_res4 = nn.Conv2d(res3_channels, res4_channels, kernel_size=1, bias=False)

        self.gamma3 = nn.Parameter(torch.tensor(residual_init, dtype=torch.float32))
        self.gamma4 = nn.Parameter(torch.tensor(residual_init, dtype=torch.float32))

        self.alpha = nn.Parameter(torch.tensor(cross_scale_init, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(cross_scale_init, dtype=torch.float32))

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

        if "res3" not in image_features or "res4" not in image_features:
            raise KeyError(f"image_features must contain res3/res4, got {image_features.keys()}")

        res3 = image_features["res3"]
        res4 = image_features["res4"]
        batch_size = sample_guidance.shape[0]

        if res3.shape[0] != batch_size or res4.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch: guidance={batch_size}, res3={res3.shape[0]}, res4={res4.shape[0]}"
            )

        gate3 = torch.sigmoid(self.text_proj_res3(sample_guidance)).view(batch_size, -1, 1, 1)
        gate4 = torch.sigmoid(self.text_proj_res4(sample_guidance)).view(batch_size, -1, 1, 1)

        res4_to_res3 = self._resize_like(self.res4_to_res3(res4), res3)
        res3_to_res4 = self._resize_like(self.res3_to_res4(res3), res4)

        out = dict(image_features)
        out["res3"] = res3 + self.gamma3 * (gate3 * res3 + self.alpha * res4_to_res3)
        out["res4"] = res4 + self.gamma4 * (gate4 * res4 + self.beta * res3_to_res4)
        return out

class LightweightDualScaleLGCE(nn.Module):
    def __init__(self, text_dim, res3_channels, res4_channels):
        super().__init__()
        self.text_proj_res3 = nn.Linear(text_dim, res3_channels)
        self.text_proj_res4 = nn.Linear(text_dim, res4_channels)

        self.fuse_res3 = nn.Sequential(
            nn.Conv2d(res3_channels * 2, res3_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(res3_channels, res3_channels, kernel_size=1, bias=False),
        )
        self.fuse_res4 = nn.Sequential(
            nn.Conv2d(res4_channels * 2, res4_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(res4_channels, res4_channels, kernel_size=1, bias=False),
        )
        self.gamma3 = nn.Parameter(torch.zeros(1))
        self.gamma4 = nn.Parameter(torch.zeros(1))

    def forward(self, image_features, sample_guidance):
        res3 = image_features["res3"]
        res4 = image_features["res4"]
        B = sample_guidance.shape[0]

        text3 = self.text_proj_res3(sample_guidance).view(B, -1, 1, 1).expand_as(res3)
        text4 = self.text_proj_res4(sample_guidance).view(B, -1, 1, 1).expand_as(res4)

        fused_res3 = self.fuse_res3(torch.cat([res3, text3], dim=1))
        fused_res4 = self.fuse_res4(torch.cat([res4, text4], dim=1))

        out = dict(image_features)
        out["res3"] = res3 + self.gamma3 * fused_res3
        out["res4"] = res4 + self.gamma4 * fused_res4
        return out