from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageGuidedCrossScaleBridge(nn.Module):
    """Lightweight sample-level language guided cross-scale bridge for res3/res4."""

    def __init__(
        self,
        text_dim: int,
        res3_channels: int,
        res4_channels: int,
        cross_scale_init: float = 0.1,
        residual_init: float = 0.05,
    ) -> None:
        super().__init__()
        self.text_proj_res3 = nn.Linear(text_dim, res3_channels)
        self.text_proj_res4 = nn.Linear(text_dim, res4_channels)

        self.res4_to_res3 = nn.Conv2d(res4_channels, res3_channels, kernel_size=1, bias=False)
        self.res3_to_res4 = nn.Conv2d(res3_channels, res4_channels, kernel_size=1, bias=False)

        # Start from a small-but-nonzero residual so the module is observable early but still stable.
        self.gamma3 = nn.Parameter(torch.tensor(residual_init))
        self.gamma4 = nn.Parameter(torch.tensor(residual_init))
        self.alpha = nn.Parameter(torch.tensor(cross_scale_init))
        self.beta = nn.Parameter(torch.tensor(cross_scale_init))

    def forward(self, image_features: Dict[str, torch.Tensor], sample_guidance: torch.Tensor) -> Dict[str, torch.Tensor]:
        if sample_guidance.dim() != 2:
            raise ValueError(f"sample_guidance must be [B, H], got {tuple(sample_guidance.shape)}")

        if "res3" not in image_features or "res4" not in image_features:
            return image_features

        res3 = image_features["res3"]
        res4 = image_features["res4"]

        if res3.shape[0] != sample_guidance.shape[0] or res4.shape[0] != sample_guidance.shape[0]:
            raise ValueError(
                f"Batch size mismatch between features and guidance: "
                f"res3={res3.shape[0]}, res4={res4.shape[0]}, guidance={sample_guidance.shape[0]}"
            )

        gate3 = torch.sigmoid(self.text_proj_res3(sample_guidance)).view(sample_guidance.shape[0], -1, 1, 1)
        gate4 = torch.sigmoid(self.text_proj_res4(sample_guidance)).view(sample_guidance.shape[0], -1, 1, 1)

        res4_up = F.interpolate(res4, size=res3.shape[-2:], mode="bilinear", align_corners=False)
        res3_down = F.interpolate(res3, size=res4.shape[-2:], mode="bilinear", align_corners=False)

        cross_to_res3 = self.res4_to_res3(res4_up)
        cross_to_res4 = self.res3_to_res4(res3_down)

        delta3 = self.gamma3 * (gate3 * res3 + self.alpha * cross_to_res3)
        delta4 = self.gamma4 * (gate4 * res4 + self.beta * cross_to_res4)

        output = dict(image_features)
        output["res3"] = res3 + delta3
        output["res4"] = res4 + delta4
        return output
