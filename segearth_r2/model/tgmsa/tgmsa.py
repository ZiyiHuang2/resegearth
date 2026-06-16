from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwinOutputTargetFilter(nn.Module):
    """Text-conditioned residual gates for frozen Swin output features.

    The residual scale for every level starts at zero, so the module is an
    exact no-op until training moves the stage-specific alpha parameters.
    """

    def __init__(self, seg_dim: int, feature_dims: Dict[str, int]):
        super().__init__()
        self.levels = tuple(feature_dims.keys())
        self.text_proj = nn.ModuleDict({
            name: nn.Linear(seg_dim, dim) for name, dim in feature_dims.items()
        })
        self.spatial_proj = nn.ModuleDict({
            name: nn.Conv2d(dim, 1, kernel_size=1) for name, dim in feature_dims.items()
            if name in {"res2", "res3"}
        })
        self.channel_proj = nn.ModuleDict({
            name: nn.Linear(dim, dim) for name, dim in feature_dims.items()
            if name not in {"res2", "res3"}
        })
        self.alpha = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1)) for name in feature_dims
        })

    def _image_level_condition(
        self,
        seg_embedding: torch.Tensor,
        mask_num: Optional[Iterable[int]],
        batch_size: int,
    ) -> torch.Tensor:
        seg = seg_embedding.squeeze(1)
        if mask_num is None:
            if seg.shape[0] == batch_size:
                return seg
            return seg.mean(dim=0, keepdim=True).expand(batch_size, -1)

        counts = [int(x) for x in mask_num]
        if sum(counts) != seg.shape[0] or len(counts) != batch_size:
            raise ValueError(
                "SwinOutputTargetFilter mask_num must match seg_embedding and feature batch size: "
                f"sum(mask_num)={sum(counts)}, len(mask_num)={len(counts)}, "
                f"seg={seg.shape[0]}, batch={batch_size}"
            )
        chunks = torch.split(seg, counts, dim=0)
        return torch.stack([chunk.mean(dim=0) for chunk in chunks], dim=0)

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        seg_embedding: torch.Tensor,
        mask_num: Optional[Iterable[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        if seg_embedding is None:
            return features
        first = next(iter(features.values()))
        cond = self._image_level_condition(seg_embedding, mask_num, first.shape[0])
        out = dict(features)
        for name in self.levels:
            feat = features[name]
            level_cond = self.text_proj[name](cond).to(dtype=feat.dtype, device=feat.device)
            alpha = self.alpha[name].to(dtype=feat.dtype, device=feat.device)
            if name in self.spatial_proj:
                spatial_bias = self.spatial_proj[name](feat)
                text_bias = level_cond.mean(dim=-1, keepdim=True).view(feat.shape[0], 1, 1, 1)
                gate = torch.sigmoid(spatial_bias + text_bias)
                delta = feat * (2.0 * gate - 1.0)
            else:
                gate = torch.sigmoid(self.channel_proj[name](level_cond)).view(feat.shape[0], feat.shape[1], 1, 1)
                delta = feat * (2.0 * gate - 1.0)
            out[name] = feat + alpha * delta
        return out


class PixelTargetBackgroundCalibrator(nn.Module):
    """Target/background residual calibration after the pixel decoder."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.target_proj = nn.Linear(hidden_dim, hidden_dim)
        self.background_proj = nn.Linear(hidden_dim, hidden_dim)
        self.mask_alpha = nn.Parameter(torch.zeros(1))
        self.scale_alpha = nn.Parameter(torch.zeros(1))

    def _condition(self, seg_embedding: torch.Tensor, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        seg = seg_embedding.squeeze(1).to(device=device)
        target = torch.tanh(self.target_proj(seg)).to(dtype=dtype)
        background = torch.tanh(self.background_proj(seg)).to(dtype=dtype)
        return target.unsqueeze(-1).unsqueeze(-1), background.unsqueeze(-1).unsqueeze(-1)

    def _calibrate(self, feature: torch.Tensor, target: torch.Tensor, background: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        target_delta = feature * target
        background_delta = feature * background
        return feature + alpha.to(dtype=feature.dtype, device=feature.device) * (target_delta - background_delta)

    def forward(
        self,
        mask_features: torch.Tensor,
        multi_scale_features: List[torch.Tensor],
        seg_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if seg_embedding is None:
            return mask_features, multi_scale_features
        target, background = self._condition(seg_embedding, mask_features.dtype, mask_features.device)
        out_mask = self._calibrate(mask_features, target, background, self.mask_alpha)
        out_multi = [
            self._calibrate(feat, target.to(device=feat.device), background.to(device=feat.device), self.scale_alpha)
            for feat in multi_scale_features
        ]
        return out_mask, out_multi


class DynamicQueryBinding(nn.Module):
    """Group-aware language-conditioned query binding for [SEG] tokens.

    The module performs two jobs:
    1. choose a small dynamic query basis for every [SEG] token and expose it as
       decoder query position, so the binding affects cross-attention;
    2. regularize [SEG] tokens from the same image to avoid identical dynamic
       query choices when there are multiple targets.
    """

    def __init__(
        self,
        hidden_dim: int,
        query_bank_size: int = 4,
        alpha_init: float = 0.0,
        diversity_margin: float = 0.2,
        num_heads: int = 4,
    ):
        super().__init__()
        if query_bank_size < 1:
            raise ValueError("query_bank_size must be >= 1")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by tgmsa_query_num_heads")
        self.query_bank = nn.Parameter(torch.empty(query_bank_size, hidden_dim))
        nn.init.normal_(self.query_bank, std=0.02)
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)
        self.query_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.peer_proj = nn.Linear(hidden_dim, hidden_dim)
        self.hard_peer_proj = nn.Linear(hidden_dim, hidden_dim)
        self.contrast_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.diversity_margin = float(diversity_margin)

    def _as_counts(self, mask_num: Optional[Iterable[int]], total: int) -> Optional[List[int]]:
        if mask_num is None:
            return None
        if torch.is_tensor(mask_num):
            counts = [int(x) for x in mask_num.detach().cpu().tolist()]
        else:
            counts = [int(x) for x in mask_num]
        if sum(counts) != total:
            raise ValueError(f"sum(mask_num)={sum(counts)} must equal SEG count {total}")
        return counts

    def _peer_context(self, text: torch.Tensor, mask_num: Optional[Iterable[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        counts = self._as_counts(mask_num, text.shape[0])
        if counts is None or max(counts) <= 1:
            zeros = text.new_zeros(text.shape)
            return zeros, zeros

        peer_context = text.new_zeros(text.shape)
        hard_peer = text.new_zeros(text.shape)
        start = 0
        for count in counts:
            group = text[start:start + count]
            if count > 1:
                group_sum = group.sum(dim=0, keepdim=True)
                peer_context[start:start + count] = (group_sum - group) / float(count - 1)
                sim = torch.matmul(F.normalize(group, dim=-1), F.normalize(group, dim=-1).transpose(0, 1))
                sim = sim.masked_fill(torch.eye(count, device=sim.device, dtype=torch.bool), -1.0)
                hard_peer[start:start + count] = group[sim.argmax(dim=-1)]
            start += count
        return peer_context, hard_peer

    def _peer_contrast_loss(self, query_pos: torch.Tensor, text: torch.Tensor, hard_peer: torch.Tensor, mask_num: Optional[Iterable[int]]) -> torch.Tensor:
        counts = self._as_counts(mask_num, query_pos.shape[0])
        if counts is None or max(counts) <= 1:
            return query_pos.new_zeros(())
        valid = torch.zeros(query_pos.shape[0], device=query_pos.device, dtype=torch.bool)
        start = 0
        for count in counts:
            if count > 1:
                valid[start:start + count] = True
            start += count
        if not valid.any():
            return query_pos.new_zeros(())
        query_norm = F.normalize(query_pos[valid], dim=-1)
        text_norm = F.normalize(text[valid], dim=-1)
        peer_norm = F.normalize(hard_peer[valid], dim=-1)
        pos_sim = (query_norm * text_norm).sum(dim=-1)
        neg_sim = (query_norm * peer_norm).sum(dim=-1)
        return F.relu(self.diversity_margin - pos_sim + neg_sim).mean()

    def _diversity_loss(self, candidates: torch.Tensor) -> torch.Tensor:
        if candidates.shape[1] == 1:
            return candidates.new_zeros(())
        norm = F.normalize(candidates, dim=-1)
        sim = torch.matmul(norm, norm.transpose(-1, -2))
        eye = torch.eye(sim.shape[-1], device=sim.device, dtype=torch.bool).unsqueeze(0)
        off_diag = sim.masked_select(~eye)
        return F.relu(off_diag - self.diversity_margin).mean()

    def _segment_separation_loss(self, query_pos: torch.Tensor, mask_num: Optional[Iterable[int]]) -> torch.Tensor:
        counts = self._as_counts(mask_num, query_pos.shape[0])
        if counts is None or max(counts) <= 1:
            return query_pos.new_zeros(())
        losses = []
        start = 0
        for count in counts:
            group = query_pos[start:start + count]
            start += count
            if count <= 1:
                continue
            norm = F.normalize(group, dim=-1)
            sim = torch.matmul(norm, norm.transpose(0, 1))
            eye = torch.eye(count, device=sim.device, dtype=torch.bool)
            losses.append(F.relu(sim.masked_select(~eye) - self.diversity_margin).mean())
        if not losses:
            return query_pos.new_zeros(())
        return torch.stack(losses).mean()

    def _binding_entropy_loss(self, weights: torch.Tensor) -> torch.Tensor:
        if weights.shape[-1] == 1:
            return weights.new_zeros(())
        entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=-1)
        return entropy.mean()

    def forward(
        self,
        seg_embedding: torch.Tensor,
        mask_num: Optional[Iterable[int]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if seg_embedding is None:
            zero = self.query_bank.new_zeros(())
            return seg_embedding, {
                "query_pos": None,
                "loss_tgmsa_query_diversity": zero,
                "loss_tgmsa_segment_separation": zero,
                "loss_tgmsa_binding_entropy": zero,
            }
        seg = seg_embedding.squeeze(1)
        text = self.text_proj(seg)
        peer_context, hard_peer = self._peer_context(text, mask_num)
        peer_context = self.peer_proj(peer_context)
        hard_peer = self.hard_peer_proj(hard_peer)
        contrast = self.contrast_proj(torch.cat([text, text - peer_context, text - hard_peer], dim=-1))
        bank = self.query_bank.unsqueeze(0).expand(seg.shape[0], -1, -1).to(device=seg.device, dtype=seg.dtype)
        attended, _ = self.query_attn(contrast.unsqueeze(1), bank, bank)
        candidates = torch.tanh(attended + contrast.unsqueeze(1) + bank)
        logits = self.score(candidates).squeeze(-1)
        weights = torch.softmax(logits, dim=-1)
        selected = torch.einsum("bk,bkc->bc", weights, candidates)
        query_pos = self.output_proj(selected)
        refined = seg_embedding + self.alpha.to(device=seg.device, dtype=seg.dtype) * query_pos.unsqueeze(1)
        info = {
            "query_pos": query_pos.unsqueeze(1),
            "loss_tgmsa_query_diversity": self._diversity_loss(candidates),
            "loss_tgmsa_segment_separation": self._segment_separation_loss(query_pos, mask_num),
            "loss_tgmsa_binding_entropy": self._binding_entropy_loss(weights),
            "loss_tgmsa_peer_contrast": self._peer_contrast_loss(query_pos, text, hard_peer, mask_num),
            "tgmsa_binding_logits": logits,
        }
        return refined, info
