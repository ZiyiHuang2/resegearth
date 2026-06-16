import torch
import torch.nn as nn
import fvcore.nn.weight_init as weight_init


class SingleQuerySegSpatialRefiner(nn.Module):
    """Lightweight spatial residual refiner for single active [SEG] query (Q=1)."""

    def __init__(self, seg_dim: int, mask_feature_dim: int):
        super().__init__()
        self.seg_proj = nn.Linear(int(seg_dim), int(mask_feature_dim))
        self.gamma_proj = nn.Linear(int(seg_dim), int(mask_feature_dim))
        self.beta_proj = nn.Linear(int(seg_dim), int(mask_feature_dim))
        self.residual_conv = nn.Conv2d(int(mask_feature_dim), 1, kernel_size=1)
        nn.init.zeros_(self.seg_proj.weight)
        nn.init.zeros_(self.seg_proj.bias)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)
        weight_init.c2_xavier_fill(self.residual_conv)

    def forward(self, seg_embedding: torch.Tensor, mask_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seg_embedding: [B, 1, seg_dim] projector output for active [SEG] query.
            mask_features: [B, C, H, W] pixel-decoder mask features.
        Returns:
            P_refine: [B, 1, H, W] residual mask logits aligned with mask_features spatial size.
        """
        if seg_embedding.dim() != 3 or seg_embedding.shape[1] != 1:
            raise ValueError(
                f"seg_embedding expected [B, 1, D], got {tuple(seg_embedding.shape)}"
            )
        if mask_features.dim() != 4:
            raise ValueError(
                f"mask_features expected [B, C, H, W], got {tuple(mask_features.shape)}"
            )
        batch_size = mask_features.shape[0]
        if seg_embedding.shape[0] != batch_size:
            raise ValueError(
                f"seg_embedding batch {seg_embedding.shape[0]} != mask_features batch {batch_size}"
            )

        seg_q = seg_embedding.squeeze(1)
        seg_feat = self.seg_proj(seg_q)
        gamma = self.gamma_proj(seg_q).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_proj(seg_q).unsqueeze(-1).unsqueeze(-1)
        conditioned = mask_features * (1.0 + gamma) + beta + seg_feat.unsqueeze(-1).unsqueeze(-1)
        p_refine = self.residual_conv(conditioned)
        if p_refine.shape[0] != batch_size or p_refine.shape[1] != 1:
            raise ValueError(
                f"P_refine expected [B, 1, H, W], got {tuple(p_refine.shape)}"
            )
        return p_refine
