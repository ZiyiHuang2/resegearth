import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleTextVisualAligner(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int] = (256, 512, 1024),
        text_dim: int = 2560,
        align_dim: int = 256,
        scale_weights: Sequence[float] = (0.5, 0.3, 0.2),
        max_spatial_tokens: int = 4096,
        pool_large_scale: bool = True,
    ):
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError(f"in_channels must contain 3 entries for res3/res4/res5, got {len(in_channels)}")
        self.in_channels = tuple(int(c) for c in in_channels)
        self.align_dim = int(align_dim)
        self.scale_weights = tuple(float(w) for w in scale_weights)
        self.max_spatial_tokens = int(max_spatial_tokens)
        self.pool_large_scale = bool(pool_large_scale)
        if len(self.scale_weights) != 3:
            raise ValueError(f"scale_weights must have 3 values, got {len(self.scale_weights)}")

        self.visual_proj3 = nn.Conv2d(self.in_channels[0], self.align_dim, kernel_size=1)
        self.visual_proj4 = nn.Conv2d(self.in_channels[1], self.align_dim, kernel_size=1)
        self.visual_proj5 = nn.Conv2d(self.in_channels[2], self.align_dim, kernel_size=1)

        self.q_proj3 = nn.Linear(self.align_dim, self.align_dim)
        self.q_proj4 = nn.Linear(self.align_dim, self.align_dim)
        self.q_proj5 = nn.Linear(self.align_dim, self.align_dim)

        self.text_proj = nn.Linear(text_dim, self.align_dim)
        self.k_proj = nn.Linear(self.align_dim, self.align_dim)
        self.v_proj = nn.Linear(self.align_dim, self.align_dim)

        self.rel_mlp3 = nn.Sequential(
            nn.Linear(self.align_dim * 3, self.align_dim),
            nn.GELU(),
            nn.Linear(self.align_dim, 1),
        )
        self.rel_mlp4 = nn.Sequential(
            nn.Linear(self.align_dim * 3, self.align_dim),
            nn.GELU(),
            nn.Linear(self.align_dim, 1),
        )
        self.rel_mlp5 = nn.Sequential(
            nn.Linear(self.align_dim * 3, self.align_dim),
            nn.GELU(),
            nn.Linear(self.align_dim, 1),
        )

        self.alpha3 = nn.Parameter(torch.zeros(1))
        self.alpha4 = nn.Parameter(torch.zeros(1))
        self.alpha5 = nn.Parameter(torch.zeros(1))

    def _scale_forward(
        self,
        feat: torch.Tensor,
        visual_proj: nn.Conv2d,
        q_proj: nn.Linear,
        rel_mlp: nn.Sequential,
        alpha: nn.Parameter,
        text_tokens_proj: torch.Tensor,
        text_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, _, h, w = feat.shape
        feat_dtype = feat.dtype

        v_s = visual_proj(feat)
        pooled = False
        if self.pool_large_scale and (h * w) > self.max_spatial_tokens:
            pooled = True
            target_hw = int(math.sqrt(self.max_spatial_tokens))
            target_hw = max(1, target_hw)
            stride_h = max(1, math.ceil(h / target_hw))
            stride_w = max(1, math.ceil(w / target_hw))
            kernel_h = stride_h
            kernel_w = stride_w
            v_s_attn = F.avg_pool2d(v_s, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w))
        else:
            v_s_attn = v_s
        h_low, w_low = v_s_attn.shape[-2:]
        v_s_flat = v_s_attn.flatten(2).transpose(1, 2)

        q_input = v_s_flat.to(dtype=q_proj.weight.dtype)
        q = q_proj(q_input)
        k = self.k_proj(text_tokens_proj.to(dtype=self.k_proj.weight.dtype))
        v = self.v_proj(text_tokens_proj.to(dtype=self.v_proj.weight.dtype))

        q_attn = q.float()
        k_attn = k.float()
        v_attn = v.float()
        attn = torch.matmul(q_attn, k_attn.transpose(-1, -2)) / math.sqrt(self.align_dim)
        if text_mask is not None:
            attn = attn.masked_fill(~text_mask[:, None, :], -1e4)
        a = torch.softmax(attn, dim=-1)
        c = torch.matmul(a, v_attn)

        v_s_float = v_s_flat.float()
        z = torch.cat([v_s_float, c, v_s_float * c], dim=-1)
        z = z.to(dtype=rel_mlp[0].weight.dtype)
        r_s = torch.sigmoid(rel_mlp(z))
        r_s = r_s.transpose(1, 2).reshape(bsz, 1, h_low, w_low)
        if pooled:
            r_s = F.interpolate(r_s, size=(h, w), mode="bilinear", align_corners=False)
        r_s = r_s.to(dtype=feat_dtype)

        out = feat * (1.0 + alpha.to(dtype=feat_dtype) * r_s)
        return out, r_s

    def forward(
        self,
        features,
        text_tokens: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        return_maps: bool = False,
    ):
        if not isinstance(features, (list, tuple)) or len(features) != 4:
            raise ValueError("features must be [res2, res3, res4, res5]")

        if text_tokens is None:
            if return_maps:
                return tuple(features), {}
            return tuple(features)

        if text_tokens.dim() != 3:
            raise ValueError(f"text_tokens must be [B,L,D], got shape={tuple(text_tokens.shape)}")

        bsz = features[0].shape[0]
        if text_tokens.shape[0] != bsz:
            raise ValueError(f"text_tokens batch {text_tokens.shape[0]} does not match features batch {bsz}")

        if text_mask is not None:
            if text_mask.dim() != 2:
                raise ValueError(f"text_mask must be [B,L], got shape={tuple(text_mask.shape)}")
            text_mask = text_mask.to(device=text_tokens.device, dtype=torch.bool)
            if text_mask.shape != text_tokens.shape[:2]:
                raise ValueError(f"text_mask shape {tuple(text_mask.shape)} mismatches text_tokens {tuple(text_tokens.shape[:2])}")

        text_input = text_tokens.to(dtype=self.text_proj.weight.dtype)
        text_tokens_proj = self.text_proj(text_input)

        res2, res3, res4, res5 = features
        out3, r3 = self._scale_forward(res3, self.visual_proj3, self.q_proj3, self.rel_mlp3, self.alpha3, text_tokens_proj, text_mask)
        out4, r4 = self._scale_forward(res4, self.visual_proj4, self.q_proj4, self.rel_mlp4, self.alpha4, text_tokens_proj, text_mask)
        out5, r5 = self._scale_forward(res5, self.visual_proj5, self.q_proj5, self.rel_mlp5, self.alpha5, text_tokens_proj, text_mask)

        outs = (res2, out3, out4, out5)
        if not return_maps:
            return outs

        align_info = {
            "R3": r3,
            "R4": r4,
            "R5": r5,
            "alpha3": self.alpha3.detach().clone(),
            "alpha4": self.alpha4.detach().clone(),
            "alpha5": self.alpha5.detach().clone(),
        }
        return outs, align_info
