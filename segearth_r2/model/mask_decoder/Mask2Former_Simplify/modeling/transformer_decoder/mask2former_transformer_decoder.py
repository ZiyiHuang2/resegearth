# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import math
import logging
import fvcore.nn.weight_init as weight_init
from typing import List, Optional, Tuple
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .position_encoding import PositionEmbeddingSine

logger = logging.getLogger(__name__)


def deterministic_slot_pe(kmax: int, dim: int, device, dtype) -> Tensor:
    """Sinusoidal slot index PE [Kmax, dim], no learnable params."""
    position = torch.arange(kmax, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(kmax, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: (dim // 2)])
    return pe


def regroup_target_maps_to_bank(
    feat_t: Tensor,
    mask_num: List[int],
    slot_pe: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """feat_t [T,C,H,W] -> bank [B,Kmax,HW,C], slot_valid_mask [B,Kmax]."""
    t, c, h, w = feat_t.shape
    b = len(mask_num)
    kmax = max(int(k) for k in mask_num)
    hw = h * w
    bank = torch.zeros(b, kmax, hw, c, device=feat_t.device, dtype=feat_t.dtype)
    valid = torch.zeros(b, kmax, dtype=torch.bool, device=feat_t.device)
    split_sizes = [int(k) for k in mask_num]
    splits = list(torch.split(feat_t, split_sizes, dim=0))
    for b_idx, (seg, k) in enumerate(zip(splits, split_sizes)):
        for k_idx in range(k):
            bank[b_idx, k_idx] = seg[k_idx].flatten(1).permute(1, 0)
            if slot_pe is not None:
                bank[b_idx, k_idx] = bank[b_idx, k_idx] + slot_pe[k_idx]
        valid[b_idx, :k] = True
    return bank, valid


def regroup_mask_features_bank(mask_features: Tensor, mask_num: List[int]) -> Tuple[Tensor, Tensor]:
    """mask_features [T,C,H,W] -> [B,Kmax,C,H,W], slot_valid_mask [B,Kmax]."""
    t, c, h, w = mask_features.shape
    b = len(mask_num)
    kmax = max(int(k) for k in mask_num)
    bank = torch.zeros(b, kmax, c, h, w, device=mask_features.device, dtype=mask_features.dtype)
    valid = torch.zeros(b, kmax, dtype=torch.bool, device=mask_features.device)
    split_sizes = [int(k) for k in mask_num]
    splits = list(torch.split(mask_features, split_sizes, dim=0))
    for b_idx, (seg, k) in enumerate(zip(splits, split_sizes)):
        for k_idx in range(k):
            bank[b_idx, k_idx] = seg[k_idx]
        valid[b_idx, :k] = True
    return bank, valid


def flatten_grouped_bank(
    bank: Tensor,
    slot_valid_mask: Tensor,
) -> Tuple[Tensor, Tensor]:
    """bank [B,Kmax,HW,C] -> flat [Kmax*HW,B,C], key_pad [B,Kmax*HW]."""
    b, kmax, hw, c = bank.shape
    flat = bank.reshape(b, kmax * hw, c).permute(1, 0, 2)
    key_pad = (~slot_valid_mask).unsqueeze(-1).expand(b, kmax, hw).reshape(b, kmax * hw)
    return flat, key_pad


def flatten_grouped_seg_masks(pred_seg_grouped: Tensor, mask_num: List[int]) -> Tensor:
    """[B,Kmax,H,W] -> [T,1,H,W] in target order."""
    rows = []
    for b_idx, k in enumerate(mask_num):
        k = int(k)
        for k_idx in range(k):
            rows.append(pred_seg_grouped[b_idx, k_idx].unsqueeze(0).unsqueeze(0))
    if not rows:
        return pred_seg_grouped.new_zeros(0, 1, *pred_seg_grouped.shape[-2:])
    return torch.cat(rows, dim=0)


def flatten_grouped_seg_logits(pred_seg_logits_grouped: Tensor, mask_num: List[int]) -> Tensor:
    """[B,Kmax,1] -> [T,1,1] in target order."""
    rows = []
    for b_idx, k in enumerate(mask_num):
        k = int(k)
        for k_idx in range(k):
            rows.append(pred_seg_logits_grouped[b_idx, k_idx].unsqueeze(0).unsqueeze(0))
    if not rows:
        return pred_seg_logits_grouped.new_zeros(0, 1, 1)
    return torch.cat(rows, dim=0)


class SetUnionMaskHead(nn.Module):
    """SET query-conditioned aggregation over valid target slots -> set mask feature [B,C,H,W]."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def _score_slots(self, q_exp: Tensor, slot_desc: Tensor) -> Tensor:
        x = torch.cat([q_exp, slot_desc], dim=-1).float()
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                x = F.linear(
                    x,
                    layer.weight.float(),
                    layer.bias.float() if layer.bias is not None else None,
                )
            elif isinstance(layer, nn.ReLU):
                x = F.relu(x)
        return x.squeeze(-1)

    def forward(self, q_set: Tensor, mask_features_bank: Tensor, slot_valid_mask: Tensor) -> Tensor:
        # q_set [B,C], mask_features_bank [B,Kmax,C,H,W]
        from segearth_r2.utils.nan_debug import (
            enabled as nan_debug_enabled,
            masked_softmax_fill,
        )

        if nan_debug_enabled() and slot_valid_mask is not None:
            if not slot_valid_mask.any(dim=1).all():
                bad_rows = (~slot_valid_mask.any(dim=1)).nonzero(as_tuple=False).view(-1).tolist()
                logger.warning("SetUnionMaskHead: image rows with no valid slots: %s", bad_rows)

        slot_desc = mask_features_bank.float().mean(dim=(-2, -1)).to(q_set.dtype)
        q_exp = q_set.unsqueeze(1).expand(-1, slot_desc.shape[1], -1)
        slot_score = self._score_slots(q_exp, slot_desc)
        slot_score = slot_score.to(q_set.dtype)
        slot_score = slot_score.masked_fill(~slot_valid_mask, masked_softmax_fill())
        slot_weight = torch.softmax(slot_score.float(), dim=1)
        slot_weight = torch.nan_to_num(slot_weight, nan=0.0, posinf=0.0, neginf=0.0)
        slot_weight = slot_weight * slot_valid_mask.float()
        row_sum = slot_weight.sum(dim=1, keepdim=True)
        fallback = slot_valid_mask.float() / slot_valid_mask.float().sum(dim=1, keepdim=True).clamp(min=1e-8)
        slot_weight = torch.where(row_sum > 0, slot_weight / row_sum.clamp(min=1e-8), fallback)
        w = slot_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(mask_features_bank.dtype)
        set_feature = (w * mask_features_bank).sum(dim=1)
        self._last_diag = {
            "slot_score": slot_score.detach(),
            "slot_weight": slot_weight.detach(),
            "slot_valid_mask": slot_valid_mask.detach(),
            "set_feature": set_feature.detach(),
        }
        return set_feature


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SetGuidedQueryRefinementBlock(nn.Module):
    """
    Closed-loop Set-guided Query Refinement (CSQR).

    Feature-level SET<->SEG interaction before the grouped mask decoder.
    """

    def __init__(
            self,
            hidden_dim,
            nheads,
            dim_feedforward=2048,
            pre_norm=False,
            fusion_alpha_init=0.99,
    ):
        super().__init__()
        self.seg_self_attn = SelfAttentionLayer(
            d_model=hidden_dim, nhead=nheads, dropout=0.0,
            activation="relu", normalize_before=pre_norm,
        )
        self.set_from_seg_attn = CrossAttentionLayer(
            d_model=hidden_dim, nhead=nheads, dropout=0.0,
            activation="relu", normalize_before=pre_norm,
        )
        self.seg_from_set_attn = CrossAttentionLayer(
            d_model=hidden_dim, nhead=nheads, dropout=0.0,
            activation="relu", normalize_before=pre_norm,
        )
        self.seg_from_img_attn = CrossAttentionLayer(
            d_model=hidden_dim, nhead=nheads, dropout=0.0,
            activation="relu", normalize_before=pre_norm,
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.fusion_mlp = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        init_logit = torch.log(torch.tensor(fusion_alpha_init) / (1.0 - fusion_alpha_init))
        # Keep shape (1,) so HF weight loading can allocate with torch.empty(*param.size()).
        self.fusion_alpha_logit = nn.Parameter(init_logit.reshape(1))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        fusion_key = prefix + 'fusion_alpha_logit'
        if fusion_key in state_dict:
            alpha = state_dict[fusion_key]
            if getattr(alpha, 'ndim', None) == 0:
                state_dict[fusion_key] = alpha.reshape(1)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
        )

    def forward(
        self,
        q_set,
        q_seg,
        img_memory,
        img_pos=None,
        seg_key_padding_mask=None,
        img_key_padding_mask=None,
    ):
        q_seg_orig = q_seg

        seg_t = q_seg.permute(1, 0, 2)
        seg_t = self.seg_self_attn(seg_t, tgt_key_padding_mask=seg_key_padding_mask)
        q1 = seg_t.permute(1, 0, 2)

        q_set_t = q_set.permute(1, 0, 2)
        q1_t = q1.permute(1, 0, 2)
        q_set_t = self.set_from_seg_attn(
            q_set_t, q1_t, memory_key_padding_mask=seg_key_padding_mask,
        )
        q_set_refined = q_set_t.permute(1, 0, 2)

        q_set_mem = q_set_refined.permute(1, 0, 2)
        q2_t = self.seg_from_set_attn(q1_t, q_set_mem)
        pad_mask_t = None
        if seg_key_padding_mask is not None:
            pad_mask_t = seg_key_padding_mask.transpose(0, 1).unsqueeze(-1)
            q2_t = q2_t.masked_fill(pad_mask_t, 0.0)

        # CSQR may use grouped memory without pos (img_pos=None); main decoder uses grouped_pos_flat.
        q3_t = self.seg_from_img_attn(
            q2_t,
            img_memory,
            pos=img_pos,
            memory_key_padding_mask=img_key_padding_mask,
        )
        if pad_mask_t is not None:
            q3_t = q3_t.masked_fill(pad_mask_t, 0.0)
        q3 = q3_t.permute(1, 0, 2)

        alpha = torch.sigmoid(self.fusion_alpha_logit)
        refined = self.fusion_mlp(self.fusion_norm(q3))
        q_seg_refined = alpha * q_seg_orig + (1.0 - alpha) * refined
        if seg_key_padding_mask is not None:
            valid = (~seg_key_padding_mask).unsqueeze(-1).to(q_seg_refined.dtype)
            q_seg_refined = q_seg_refined * valid

        return q_set_refined, q_seg_refined


class MultiScaleMaskedTransformerDecoder(nn.Module):
    def __init__(
            self,
            in_channels,
            num_classes,
            mask_classification=True,
            hidden_dim=256,
            num_queries=100,
            nheads=8,
            dim_feedforward=2048,
            dec_layers=10,
            pre_norm=False,
            mask_dim=256,
            enforce_input_project=False
    ):
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(self, x, mask_features, mask=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features,
                                                                               attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features,
                                                                                   attn_mask_target_size=size_list[(
                                                                                                                           i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask.float(), size=attn_mask_target_size, mode="bilinear",
                                  align_corners=False).to(mask_embed.dtype)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                         1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]


class MultiScaleMaskedTransformerDecoderForOPTPreTrain(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_dim=256,
            num_queries=100,
            nheads=8,
            dim_feedforward=2048,
            dec_layers=10,
            pre_norm=False,
            mask_dim=256,
            enforce_input_project=False,
            seg_norm=False,
            seg_proj=True,
            seg_fuse_score=False,
            use_seg_query=False,
            use_csqr=False,
            csqr_fusion_alpha_init=0.99,
    ):
        nn.Module.__init__(self)
        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.use_seg_query = use_seg_query
        self.use_csqr = use_csqr
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.seg_norm = seg_norm
        self.seg_proj = seg_proj
        self.seg_fuse_score = seg_fuse_score
        if self.seg_norm:
            print('add seg norm for [SEG]')
            self.seg_proj_after_norm = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
            self.SEG_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.SEG_query_embed = nn.Embedding(num_queries + 1, hidden_dim)
        self.new_query_embed = nn.Embedding(1, hidden_dim) # (1, 256)
        # SET + SEG independent role embeddings (zero-init for safe training start)
        self.SET_query_embed = nn.Embedding(1, hidden_dim)
        nn.init.constant_(self.SET_query_embed.weight, 0)
        self.SEG_role_embed = nn.Embedding(1, hidden_dim)
        nn.init.constant_(self.SEG_role_embed.weight, 0)
        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.SEG_proj = MLP(hidden_dim, hidden_dim, hidden_dim, 2)

        if self.use_csqr:
            print('SET++ CSQR block enabled (Set-guided Query Refinement)')
            self.csqr_block = SetGuidedQueryRefinementBlock(
                hidden_dim=hidden_dim,
                nheads=nheads,
                dim_feedforward=dim_feedforward,
                pre_norm=pre_norm,
                fusion_alpha_init=csqr_fusion_alpha_init,
            )
        self.set_union_head = SetUnionMaskHead(hidden_dim)

    def prepare_grouped_banks(self, x, mask_features, mask_num):
        """Build slot-aware grouped memory/pos banks from per-target batch T."""
        kmax = max(int(k) for k in mask_num)
        slot_pe = deterministic_slot_pe(kmax, self.hidden_dim, x[0].device, x[0].dtype)
        memory_banks = []
        pos_banks = []
        for i, feat in enumerate(x):
            proj = self.input_proj[i](feat) + self.level_embed.weight[i].view(1, -1, 1, 1)
            pos_map = self.pe_layer(feat, None)
            mem_bank, _ = regroup_target_maps_to_bank(proj, mask_num, slot_pe=slot_pe)
            pos_bank, valid = regroup_target_maps_to_bank(pos_map, mask_num, slot_pe=None)
            memory_banks.append(mem_bank)
            pos_banks.append(pos_bank)
        mask_features_bank, slot_valid_mask = regroup_mask_features_bank(mask_features, mask_num)
        return memory_banks, pos_banks, mask_features_bank, slot_valid_mask

    def forward(
        self,
        x,
        mask_features,
        mask=None,
        seg_query=None,
        SEG_embedding=None,
        SET_embedding=None,
        mask_num=None,
        per_target_mode=False,
        grouped_setpp_mode=False,
        grouped_memory_banks=None,
        grouped_pos_banks=None,
        mask_features_bank=None,
        slot_valid_mask=None,
        target_to_image=None,
    ):
        return self.forward_woconcat(
            x,
            mask_features,
            mask,
            seg_query,
            SEG_embedding,
            SET_embedding,
            mask_num,
            per_target_mode=per_target_mode,
            grouped_setpp_mode=grouped_setpp_mode,
            grouped_memory_banks=grouped_memory_banks,
            grouped_pos_banks=grouped_pos_banks,
            mask_features_bank=mask_features_bank,
            slot_valid_mask=slot_valid_mask,
            target_to_image=target_to_image,
        )

    def forward_woconcat(
        self,
        x,
        mask_features,
        mask=None,
        seg_query=None,
        SEG_embedding=None,
        SET_embedding=None,
        mask_num=None,
        per_target_mode=False,
        grouped_setpp_mode=False,
        grouped_memory_banks=None,
        grouped_pos_banks=None,
        mask_features_bank=None,
        slot_valid_mask=None,
        target_to_image=None,
    ):
        if grouped_setpp_mode:
            assert not per_target_mode, "grouped_setpp_mode conflicts with per_target_mode"
            return self._forward_grouped_setpp(
                x=x,
                mask_num=mask_num,
                SEG_embedding=SEG_embedding,
                SET_embedding=SET_embedding,
                grouped_memory_banks=grouped_memory_banks,
                grouped_pos_banks=grouped_pos_banks,
                mask_features_bank=mask_features_bank,
                slot_valid_mask=slot_valid_mask,
                target_to_image=target_to_image,
            )

        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2).to(x[i].dtype))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten bsxCxHW to HWxbsxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        spatial_bs, _ = src[0].shape[1], src[0].shape[2]
        bs = spatial_bs

        if per_target_mode:
            # Per-target spatial batch T: one (SET, SEG) pair per row; SET already repeat_interleaved to T.
            assert SEG_embedding is not None and SET_embedding is not None
            assert SEG_embedding.shape[0] == SET_embedding.shape[0] == bs, (
                f"per_target_mode expects SEG/SET batch={bs}, got "
                f"SEG={SEG_embedding.shape[0]}, SET={SET_embedding.shape[0]}"
            )
            Kmax = 1
            SET_query = SET_embedding + self.SET_query_embed.weight.unsqueeze(0)  # [T, 1, C]
            SEG_query = SEG_embedding + self.SEG_role_embed.weight.unsqueeze(0)  # [T, 1, C]
            valid_seg_mask = torch.ones(bs, Kmax, dtype=torch.bool, device=SEG_embedding.device)
            seg_key_padding_mask = None
            if self.use_csqr:
                SET_query, SEG_query = self.csqr_block(
                    q_set=SET_query,
                    q_seg=SEG_query,
                    img_memory=src[0],
                    img_pos=pos[0],
                    seg_key_padding_mask=seg_key_padding_mask,
                )
            output = torch.cat([SET_query, SEG_query], dim=1).permute(1, 0, 2)  # [2, T, C]
            SET_emb_2d = SET_embedding.squeeze(1)
            SEG_emb_2d = SEG_embedding.squeeze(1)
            combined_emb = torch.stack([SET_emb_2d, SEG_emb_2d], dim=1)  # [T, 2, C]
            query_embed = torch.zeros(1 + Kmax, bs, self.hidden_dim, device=SEG_embedding.device, dtype=SEG_embedding.dtype)
            tgt_key_padding_mask = torch.zeros(bs, 1 + Kmax, dtype=torch.bool, device=SEG_embedding.device)
        else:
            # Image-grouped mode: spatial batch B, up to Kmax SEG slots per image.
            # SEG_embedding: [sum(K_i), 1, C] -> pad to [B, Kmax, C]
            # SET_embedding: [B, 1, C]
            Kmax = max(mask_num) if mask_num is not None else 0

            SEG_emb = SEG_embedding.squeeze(1)  # [sum(K_i), C]
            SEG_padded = torch.zeros(bs, Kmax, self.hidden_dim, device=SEG_emb.device, dtype=SEG_emb.dtype)
            split_sizes = mask_num if mask_num is not None else [SEG_emb.shape[0]]
            SEG_split = list(torch.split(SEG_emb, split_sizes, dim=0))
            for b_idx, seg in enumerate(SEG_split):
                SEG_padded[b_idx, :seg.shape[0]] = seg

            SET_query = SET_embedding + self.SET_query_embed.weight.unsqueeze(0)  # [B, 1, C]
            SEG_query = SEG_padded + self.SEG_role_embed.weight.unsqueeze(0)  # [B, Kmax, C]

            valid_seg_mask = torch.zeros(bs, Kmax, dtype=torch.bool, device=SEG_embedding.device)
            for b_idx, k in enumerate(mask_num if mask_num is not None else [Kmax]):
                valid_seg_mask[b_idx, :k] = True

            seg_key_padding_mask = None
            if Kmax > 0:
                seg_key_padding_mask = ~valid_seg_mask
            if self.use_csqr and Kmax > 0:
                SET_query, SEG_query = self.csqr_block(
                    q_set=SET_query,
                    q_seg=SEG_query,
                    img_memory=src[0],
                    img_pos=pos[0],
                    seg_key_padding_mask=seg_key_padding_mask,
                )

            output = torch.cat([SET_query, SEG_query], dim=1).permute(1, 0, 2)  # [1+Kmax, B, C]
            SET_emb_2d = SET_embedding.squeeze(1)
            combined_emb = torch.cat([SET_emb_2d.unsqueeze(1), SEG_padded], dim=1)  # [B, 1+Kmax, C]
            query_embed = torch.zeros(1 + Kmax, bs, self.hidden_dim, device=SEG_embedding.device, dtype=SEG_embedding.dtype)
            tgt_key_padding_mask = torch.ones(bs, 1 + Kmax, dtype=torch.bool, device=SEG_embedding.device)
            tgt_key_padding_mask[:, 0] = False
            for b_idx, k in enumerate(mask_num if mask_num is not None else [Kmax]):
                tgt_key_padding_mask[b_idx, 1:1 + k] = False
            
        predictions_SEG_class = []        
        predictions_mask = []


        # prediction heads on learnable query features
        if self.use_seg_query:
            SEG_class, outputs_mask, attn_mask = self.forward_prediction_heads(output,
                                                                                mask_features,
                                                                                attn_mask_target_size=
                                                                                size_list[
                                                                                            0],
                                                                                SEG_embedding=combined_emb,
                                                                                )
        else:
            SEG_class, outputs_mask, attn_mask = self.forward_prediction_heads(output,
                                                                                mask_features,
                                                                                attn_mask_target_size=
                                                                                size_list[
                                                                                            0],
                                                                                SEG_embedding=None,
                                                                                )

        predictions_SEG_class.append(SEG_class)
        
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=tgt_key_padding_mask,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            # Zero out padded SEG query positions after each decoder layer
            # SET query (index 0) stays; padded SEG queries (beyond actual K_i) get zeroed
            if mask_num is not None:
                pad_mask = tgt_key_padding_mask.T.unsqueeze(-1)  # [1+Kmax, B, 1]
                output = output.masked_fill(pad_mask, 0.0)

            if self.use_seg_query:
                SEG_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                    output, mask_features,
                    attn_mask_target_size=
                    size_list[(
                                      i + 1) % self.num_feature_levels],
                    SEG_embedding=combined_emb,
                    )
            else:
                SEG_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                    output, mask_features,
                    attn_mask_target_size=
                    size_list[(
                                      i + 1) % self.num_feature_levels],
                    SEG_embedding=None,
                    )
            predictions_SEG_class.append(SEG_class)
            
            predictions_mask.append(outputs_mask)

        assert len(predictions_SEG_class) == self.num_layers + 1

        pred_masks = predictions_mask[-1]
        pred_SEG_logits = predictions_SEG_class[-1]
        out = {
            'pred_SEG_logits': pred_SEG_logits,
            'pred_masks': pred_masks,
            'pred_set_union_mask': pred_masks[:, 0:1],
            'pred_seg_masks': pred_masks[:, 1:],
            'pred_seg_logits': pred_SEG_logits[:, 1:] if pred_SEG_logits is not None else None,
            'valid_seg_mask': valid_seg_mask,
            'per_target_mode': per_target_mode,
            'aux_outputs': self._set_aux_loss(
                predictions_SEG_class, predictions_mask,
            )
        }
        return out

    def _forward_grouped_setpp(
        self,
        x,
        mask_num,
        SEG_embedding,
        SET_embedding,
        grouped_memory_banks,
        grouped_pos_banks,
        mask_features_bank,
        slot_valid_mask,
        target_to_image,
    ):
        assert grouped_memory_banks is not None
        assert grouped_pos_banks is not None
        assert slot_valid_mask is not None
        assert mask_features_bank is not None
        assert mask_num is not None
        assert len(x) == self.num_feature_levels
        assert SET_embedding.shape[0] == len(mask_num), (
            f"SET_embedding batch={SET_embedding.shape[0]} != B={len(mask_num)}"
        )

        pred_dtype = self.transformer_cross_attention_layers[0].multihead_attn.in_proj_weight.dtype
        SEG_embedding = SEG_embedding.to(dtype=pred_dtype)
        SET_embedding = SET_embedding.to(dtype=pred_dtype)
        if self.set_union_head.mlp[0].weight.dtype != pred_dtype:
            self.set_union_head.to(dtype=pred_dtype)

        bs = len(mask_num)
        kmax = int(slot_valid_mask.shape[1])
        size_list = [x[i].shape[-2:] for i in range(self.num_feature_levels)]

        src_flat = []
        pos_flat = []
        mem_key_pads = []
        for level in range(self.num_feature_levels):
            mem_flat, key_pad = flatten_grouped_bank(grouped_memory_banks[level], slot_valid_mask)
            pos_level, _ = flatten_grouped_bank(grouped_pos_banks[level], slot_valid_mask)
            src_flat.append(mem_flat.to(dtype=pred_dtype))
            pos_flat.append(pos_level.to(dtype=pred_dtype))
            mem_key_pads.append(key_pad)

        seg_emb = SEG_embedding.squeeze(1)
        seg_padded = torch.zeros(bs, kmax, self.hidden_dim, device=seg_emb.device, dtype=seg_emb.dtype)
        split_sizes = [int(k) for k in mask_num]
        for b_idx, seg in enumerate(torch.split(seg_emb, split_sizes, dim=0)):
            seg_padded[b_idx, : seg.shape[0]] = seg

        set_query = (SET_embedding + self.SET_query_embed.weight.unsqueeze(0).to(pred_dtype))
        seg_query = (seg_padded + self.SEG_role_embed.weight.unsqueeze(0).to(pred_dtype))
        valid_seg_mask = slot_valid_mask

        seg_key_padding_mask = ~valid_seg_mask if kmax > 0 else None
        if self.use_csqr and kmax > 0:
            set_query, seg_query = self.csqr_block(
                q_set=set_query,
                q_seg=seg_query,
                img_memory=src_flat[0],
                img_pos=None,
                seg_key_padding_mask=seg_key_padding_mask,
                img_key_padding_mask=mem_key_pads[0],
            )
            set_query = set_query.to(dtype=pred_dtype)
            seg_query = seg_query.to(dtype=pred_dtype)

        output = torch.cat([set_query, seg_query], dim=1).permute(1, 0, 2).to(dtype=pred_dtype)
        set_emb_2d = SET_embedding.squeeze(1)
        combined_emb = torch.cat([set_emb_2d.unsqueeze(1), seg_padded], dim=1)
        query_embed = torch.zeros(1 + kmax, bs, self.hidden_dim, device=seg_emb.device, dtype=pred_dtype)
        tgt_key_padding_mask = torch.ones(bs, 1 + kmax, dtype=torch.bool, device=seg_emb.device)
        tgt_key_padding_mask[:, 0] = False
        for b_idx, k in enumerate(mask_num):
            tgt_key_padding_mask[b_idx, 1 : 1 + int(k)] = False

        predictions_seg_class = []
        predictions_seg_grouped = []
        predictions_set_union = []
        attn_mask = None

        seg_class, _, seg_grouped, set_union, attn_mask = self.forward_prediction_heads_grouped(
            output,
            mask_features_bank,
            size_list[0],
            combined_emb,
            slot_valid_mask,
            compute_set_union=False,
        )
        predictions_seg_class.append(seg_class)
        predictions_seg_grouped.append(seg_grouped)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            if attn_mask is not None:
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            output = self.transformer_cross_attention_layers[i](
                output,
                src_flat[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=mem_key_pads[level_index],
                pos=pos_flat[level_index],
                query_pos=query_embed,
            )
            output = self.transformer_self_attention_layers[i](
                output,
                tgt_mask=None,
                tgt_key_padding_mask=tgt_key_padding_mask,
                query_pos=query_embed,
            )
            output = self.transformer_ffn_layers[i](output)
            pad_mask = tgt_key_padding_mask.T.unsqueeze(-1)
            output = output.masked_fill(pad_mask, 0.0)

            is_final = i == self.num_layers - 1
            seg_class, _, seg_grouped, set_union, attn_mask = self.forward_prediction_heads_grouped(
                output,
                mask_features_bank,
                size_list[(i + 1) % self.num_feature_levels],
                combined_emb,
                slot_valid_mask,
                compute_set_union=is_final,
            )
            predictions_seg_class.append(seg_class)
            predictions_seg_grouped.append(seg_grouped)
            if is_final:
                predictions_set_union.append(set_union)

        pred_seg_grouped = predictions_seg_grouped[-1]
        pred_set_union = predictions_set_union[-1]
        pred_masks = torch.cat([pred_set_union, pred_seg_grouped], dim=1)
        pred_seg_flat = flatten_grouped_seg_masks(pred_seg_grouped, mask_num)
        pred_seg_logits = predictions_seg_class[-1]
        pred_seg_logits_grouped = pred_seg_logits[:, 1:] if pred_seg_logits is not None else None
        pred_seg_logits_flat = (
            flatten_grouped_seg_logits(pred_seg_logits_grouped, mask_num)
            if pred_seg_logits_grouped is not None
            else None
        )

        aux_outputs = []
        for seg_c, seg_g in zip(predictions_seg_class[:-1], predictions_seg_grouped[:-1]):
            aux_outputs.append(
                {
                    "pred_SEG_logits": seg_c[:, 1:] if seg_c is not None else None,
                    "pred_seg_masks": flatten_grouped_seg_masks(seg_g, mask_num),
                    "pred_seg_masks_grouped": seg_g,
                    "grouped_setpp_mode": True,
                    "per_target_mode": False,
                }
            )

        return {
            "pred_SEG_logits": pred_seg_logits,
            "pred_masks": pred_masks,
            "pred_set_union_mask": pred_masks[:, 0:1],
            "pred_seg_masks_grouped": pred_masks[:, 1:],
            "pred_seg_masks": pred_seg_flat,
            "pred_seg_logits": pred_seg_logits_flat,
            "pred_seg_logits_grouped": pred_seg_logits_grouped,
            "valid_seg_mask": valid_seg_mask,
            "grouped_setpp_mode": True,
            "per_target_mode": False,
            "target_to_image": target_to_image,
            "mask_num": list(mask_num),
            "aux_outputs": aux_outputs,
        }

    def forward_prediction_heads_grouped(
        self,
        output,
        mask_features_bank,
        attn_mask_target_size,
        SEG_embedding=None,
        slot_valid_mask=None,
        compute_set_union=True,
    ):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        b = decoder_output.shape[0]
        dec_fp = decoder_output.float()
        if torch.isnan(dec_fp).any() or torch.isinf(dec_fp).any():
            dec_fp = torch.nan_to_num(dec_fp, nan=0.0, posinf=50.0, neginf=-50.0)
        dec_in = torch.clamp(dec_fp, min=-50.0, max=50.0).to(decoder_output.dtype)

        if SEG_embedding is not None:
            if self.seg_proj:
                decoder_seg_output = self.SEG_proj(decoder_output)
            else:
                decoder_seg_output = decoder_output
            seg_emb = SEG_embedding
            if self.seg_norm:
                seg_emb = self.SEG_norm(seg_emb)
                seg_emb = self.seg_proj_after_norm(seg_emb)
            seg_class = torch.einsum("bqc,bqc->bq", decoder_seg_output, seg_emb).unsqueeze(-1)
        else:
            seg_class = None

        mask_embed_all = self.mask_embed(dec_in)
        pred_seg_grouped = torch.einsum(
            "bkm,bkchw->bkhw",
            mask_embed_all[:, 1:].float(),
            mask_features_bank.float(),
        ).to(decoder_output.dtype)

        if compute_set_union:
            q_set = dec_in[:, 0]
            set_feature = self.set_union_head(q_set, mask_features_bank, slot_valid_mask)
            mask_embed_set = mask_embed_all[:, 0]
            pred_set_union = torch.einsum(
                "bc,bchw->bhw",
                mask_embed_set.float(),
                set_feature.float(),
            ).unsqueeze(1).to(decoder_output.dtype)
            from segearth_r2.utils.nan_debug import audit_set_union_head, current_step, enabled as nan_debug_enabled
            if nan_debug_enabled() and hasattr(self.set_union_head, "_last_diag"):
                d = self.set_union_head._last_diag
                audit_set_union_head(
                    d["slot_score"],
                    d["slot_weight"],
                    d["slot_valid_mask"],
                    d["set_feature"],
                    mask_embed_set=mask_embed_set,
                    step=current_step(),
                )
        else:
            pred_set_union = torch.zeros(
                b,
                1,
                pred_seg_grouped.shape[-2],
                pred_seg_grouped.shape[-1],
                device=decoder_output.device,
                dtype=decoder_output.dtype,
            )

        outputs_mask = torch.cat([pred_set_union, pred_seg_grouped], dim=1)
        outputs_mask_safe = torch.nan_to_num(outputs_mask.float(), nan=0.0, posinf=0.0, neginf=0.0)
        attn_mask = F.interpolate(
            outputs_mask_safe,
            size=attn_mask_target_size,
            mode="bilinear",
            align_corners=False,
        ).to(mask_embed_all.dtype)
        attn_mask = (
            attn_mask.sigmoid()
            .flatten(2)
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
            < 0.5
        ).bool()
        attn_mask = attn_mask.detach()
        kmax = int(slot_valid_mask.shape[1]) if slot_valid_mask is not None else 1
        hw = attn_mask.shape[-1]
        if kmax > 1:
            attn_mask = (
                attn_mask.unsqueeze(-1)
                .expand(attn_mask.shape[0], attn_mask.shape[1], hw, kmax)
                .reshape(attn_mask.shape[0], attn_mask.shape[1], hw * kmax)
            )
        return seg_class, outputs_mask, pred_seg_grouped, pred_set_union, attn_mask

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size, SEG_embedding=None,
                                ):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)  # [B, Q, C] where Q = 1+Kmax
        if SEG_embedding is not None:
            if self.seg_proj:
                decoder_seg_output = self.SEG_proj(decoder_output)
            else:
                decoder_seg_output = decoder_output
            if self.seg_norm:
                SEG_embedding = self.SEG_norm(SEG_embedding)
                SEG_embedding = self.seg_proj_after_norm(SEG_embedding)
            # [B, Q, C] x [B, Q, C] -> [B, Q] dot-product, then unsqueeze to [B, Q, 1]
            SEG_class = torch.einsum('bqc,bqc->bq', decoder_seg_output, SEG_embedding).unsqueeze(-1)
        else:
            SEG_class = None

        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask.float(), size=attn_mask_target_size, mode="bilinear",
                                  align_corners=False).to(mask_embed.dtype)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                         1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return SEG_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_SEG_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # if self.mask_classification:
        #     return [
        #         {"pred_logits": a, "pred_masks": b}
        #         for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        #     ]
        # else:
        #     return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

        return [
            {"pred_SEG_logits": a, "pred_masks": c,}
            for a, c in zip(outputs_SEG_class[:-1], outputs_seg_masks[:-1])
        ]






