# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .position_encoding import PositionEmbeddingSine


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

    def forward(self, q_set, q_seg, img_memory, img_pos=None, seg_key_padding_mask=None):
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

        q3_t = self.seg_from_img_attn(q2_t, img_memory, pos=img_pos)
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


    def forward(self, x, mask_features, mask=None, seg_query=None, SEG_embedding=None, SET_embedding=None, mask_num=None):

        return self.forward_woconcat(x, mask_features, mask, seg_query, SEG_embedding, SET_embedding, mask_num)

    def forward_woconcat(self, x, mask_features, mask=None, seg_query=None, SEG_embedding=None, SET_embedding=None, mask_num=None):
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

        _, bs, _ = src[0].shape

        # QxNxC -- SET + SEG independent dual-line query layout
        # SEG_embedding: [sum(K_i), 1, C] -> pad to [B, Kmax, C]
        # SET_embedding: [B, 1, C]
        Kmax = max(mask_num) if mask_num is not None else 0

        SEG_emb = SEG_embedding.squeeze(1)  # [sum(K_i), C]
        SEG_padded = torch.zeros(bs, Kmax, self.hidden_dim, device=SEG_emb.device, dtype=SEG_emb.dtype)
        split_sizes = mask_num if mask_num is not None else [SEG_emb.shape[0]]
        SEG_split = list(torch.split(SEG_emb, split_sizes, dim=0))
        for b_idx, seg in enumerate(SEG_split):
            SEG_padded[b_idx, :seg.shape[0]] = seg

        # Build SET query: P_SET(h_SET) + e_SET (role embedding)
        SET_query = SET_embedding + self.SET_query_embed.weight.unsqueeze(0)  # [B, 1, C]

        # Build SEG queries: P_SEG(h_SEG_i) + e_SEG (role embedding)
        SEG_query = SEG_padded + self.SEG_role_embed.weight.unsqueeze(0)  # [B, Kmax, C]

        valid_seg_mask = torch.zeros(bs, Kmax, dtype=torch.bool, device=SEG_embedding.device)
        for b_idx, k in enumerate(mask_num if mask_num is not None else [Kmax]):
            valid_seg_mask[b_idx, :k] = True

        # CSQR: feature-level SET<->SEG refinement before grouped decoder
        seg_key_padding_mask = None
        if Kmax > 0:
            seg_key_padding_mask = ~valid_seg_mask  # True = padded SEG slot
        if self.use_csqr and Kmax > 0:
            SET_query, SEG_query = self.csqr_block(
                q_set=SET_query,
                q_seg=SEG_query,
                img_memory=src[0],
                img_pos=pos[0],
                seg_key_padding_mask=seg_key_padding_mask,
            )

        # Concat: [SET, SEG_1, ..., SEG_Kmax] -> [B, 1+Kmax, C] -> [1+Kmax, B, C]
        output = torch.cat([SET_query, SEG_query], dim=1)  # [B, 1+Kmax, C]
        output = output.permute(1, 0, 2)  # [1+Kmax, B, C]

        # Combined embedding for SEG_class dot-product
        SET_emb_2d = SET_embedding.squeeze(1)  # [B, C]
        combined_emb = torch.cat([SET_emb_2d.unsqueeze(1), SEG_padded], dim=1)  # [B, 1+Kmax, C]

        # query_embed (positional): zeros for now
        query_embed = torch.zeros(1 + Kmax, bs, self.hidden_dim, device=SEG_embedding.device, dtype=SEG_embedding.dtype)

        # Build tgt_key_padding_mask for self-attention (True = mask out)
        tgt_key_padding_mask = torch.ones(bs, 1 + Kmax, dtype=torch.bool, device=SEG_embedding.device)
        tgt_key_padding_mask[:, 0] = False  # SET query always valid
        for b_idx, k in enumerate(mask_num if mask_num is not None else [Kmax]):
            tgt_key_padding_mask[b_idx, 1:1 + k] = False  # valid SEG queries
            
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
            'aux_outputs': self._set_aux_loss(
                predictions_SEG_class, predictions_mask,
            )
        }
        return out

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






