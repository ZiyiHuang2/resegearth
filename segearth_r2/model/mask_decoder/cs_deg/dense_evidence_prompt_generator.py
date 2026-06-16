"""DEPG: Dense Evidence Prompt Generator for CS-DEG++."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseEvidencePromptGenerator(nn.Module):
    """
    Shared-trunk dense evidence prompt generator.

    Produces sparse prompt tokens and dense target/context/uncertainty maps.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        mask_dim: int = 256,
        num_levels: int = 3,
        num_heads: int = 8,
        use_mask_feature_detail: bool = True,
        use_uncertainty: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_mask_feature_detail = use_mask_feature_detail
        self.use_uncertainty = use_uncertainty

        self.level_embed = nn.Embedding(num_levels, hidden_dim // 4)
        self.token_cross = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=False)

        in_ch = 4 if use_mask_feature_detail else 3
        mid_ch = hidden_dim // 4
        self.trunk = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, mid_ch), mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, mid_ch), mid_ch),
            nn.ReLU(inplace=True),
        )

        if use_mask_feature_detail:
            self.detail_proj = nn.Conv2d(mask_dim, 1, kernel_size=1)

        self.head_target = nn.Conv2d(mid_ch, 1, kernel_size=1)
        self.head_context = nn.Conv2d(mid_ch, 1, kernel_size=1)
        self.head_uncertainty = nn.Conv2d(mid_ch, 1, kernel_size=1) if use_uncertainty else None

        for head in (self.head_target, self.head_context, self.head_uncertainty):
            if head is not None:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def _query_src_similarity(self, output: torch.Tensor, spatial_map: torch.Tensor) -> torch.Tensor:
        query = output.transpose(0, 1)
        if spatial_map.shape[1] == 1:
            q = query.unsqueeze(-1).unsqueeze(-1)
            return (q * spatial_map.unsqueeze(1)).sum(dim=2)
        if spatial_map.shape[1] != query.shape[-1]:
            spatial_map = spatial_map[:, : query.shape[-1]]
        return torch.einsum("bqc,bchw->bqhw", query, spatial_map)

    def _refine_sparse_prompts(
        self,
        evidence_tokens: torch.Tensor | None,
        src: torch.Tensor,
    ) -> torch.Tensor:
        if evidence_tokens is None:
            p = 1
            b, c = src.shape[1], src.shape[2]
            return torch.zeros(p, b, c, device=src.device, dtype=src.dtype)
        refined, _ = self.token_cross(evidence_tokens, src, src, need_weights=False)
        return refined

    def _token_visual_summary(
        self,
        sparse_prompts: torch.Tensor,
        src_map: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate token-visual attention -> [B, 1, H, W]."""
        b, _, h, w = src_map.shape
        c = src_map.shape[1]
        tokens = sparse_prompts.transpose(0, 1)  # [B, P, C]
        src_flat = src_map.flatten(2).transpose(1, 2)  # [B, HW, C]
        attn = torch.einsum("bpc,bhc->bph", tokens, src_flat) / (c ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        return attn.mean(dim=1).view(b, 1, h, w)

    def _apply_heads(self, x: torch.Tensor, q: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        x = x.to(dtype=self.trunk[0].weight.dtype)
        h = self.trunk(x)
        target = self.head_target(h)
        context = self.head_context(h)
        uncertainty = self.head_uncertainty(h) if self.head_uncertainty is not None else None

        def expand_q(t: torch.Tensor) -> torch.Tensor:
            if q == 1:
                return t
            return t.expand(-1, q, -1, -1)

        return expand_q(target), expand_q(context), expand_q(uncertainty) if uncertainty is not None else None

    def _build_feats(
        self,
        output: torch.Tensor,
        src: torch.Tensor,
        level_size: tuple[int, int],
        prev_outputs_mask: torch.Tensor,
        mask_features: torch.Tensor | None,
        sparse_prompts: torch.Tensor,
    ) -> torch.Tensor:
        q, b, c = output.shape
        hl, wl = level_size
        src_map = src.transpose(0, 1).reshape(b, c, hl, wl)
        sim = self._query_src_similarity(output, src_map)

        prev_low = F.interpolate(
            prev_outputs_mask.float(),
            size=level_size,
            mode="bilinear",
            align_corners=False,
        )
        prev_map = prev_low[:, 0] if prev_low.shape[1] == 1 else prev_low.mean(dim=1)
        sim_map = sim[:, 0] if q == 1 else sim.mean(dim=1)

        token_map = self._token_visual_summary(sparse_prompts, src_map)
        feats = [sim_map.unsqueeze(1), prev_map.unsqueeze(1), token_map]

        if self.use_mask_feature_detail and mask_features is not None:
            detail = self.detail_proj(mask_features)
            detail = F.interpolate(detail, size=level_size, mode="bilinear", align_corners=False)
            feats.append(detail)

        return torch.cat(feats, dim=1)

    def forward(
        self,
        output: torch.Tensor,
        src: torch.Tensor,
        level_size: tuple[int, int],
        prev_outputs_mask: torch.Tensor,
        mask_features: torch.Tensor | None,
        evidence_tokens: torch.Tensor | None,
        level_index: int,
        compute_high_res: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """
        Returns:
            sparse_prompts, target_lr, context_lr, uncertainty_lr,
            target_high (optional), context_high (optional)
        """
        q = output.shape[0]
        sparse_prompts = self._refine_sparse_prompts(evidence_tokens, src)
        x = self._build_feats(output, src, level_size, prev_outputs_mask, mask_features, sparse_prompts)
        target_lr, context_lr, uncertainty_lr = self._apply_heads(x, q)

        target_high = context_high = None
        if compute_high_res and mask_features is not None:
            target_high, context_high, _ = self.forward_high_res(
                output, prev_outputs_mask, mask_features, sparse_prompts
            )
        del level_index
        return sparse_prompts, target_lr, context_lr, uncertainty_lr, target_high, context_high

    def forward_high_res(
        self,
        output: torch.Tensor,
        prev_outputs_mask: torch.Tensor,
        mask_features: torch.Tensor,
        sparse_prompts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        q = output.shape[0]
        if self.use_mask_feature_detail:
            detail = self.detail_proj(mask_features)
        else:
            detail = mask_features.mean(dim=1, keepdim=True)

        sim = self._query_src_similarity(output, detail)
        if prev_outputs_mask.shape[-2:] != mask_features.shape[-2:]:
            prev_h = F.interpolate(
                prev_outputs_mask.float(),
                size=mask_features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        else:
            prev_h = prev_outputs_mask.float()
        prev_map = prev_h[:, 0] if prev_h.shape[1] == 1 else prev_h.mean(dim=1)
        sim_map = sim[:, 0] if q == 1 else sim.mean(dim=1)

        if sparse_prompts is not None:
            token_map = self._token_visual_summary(sparse_prompts, detail)
        else:
            b = mask_features.shape[0]
            token_map = torch.zeros(
                b, 1, mask_features.shape[-2], mask_features.shape[-1],
                device=mask_features.device, dtype=mask_features.dtype,
            )

        feats = [sim_map.unsqueeze(1), prev_map.unsqueeze(1), token_map]
        if self.use_mask_feature_detail:
            feats.append(self.detail_proj(mask_features))
        x = torch.cat(feats, dim=1)
        return self._apply_heads(x, q)
