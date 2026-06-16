# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .position_encoding import PositionEmbeddingSine
from segearth_r2.model.mask_decoder.cs_deg.referential_evidence_tokenizer import ReferentialEvidenceTokenizer
from segearth_r2.model.mask_decoder.cs_deg.dense_evidence_prompt_generator import DenseEvidencePromptGenerator
from segearth_r2.model.mask_decoder.cs_deg.bidirectional_hierarchical_evidence_fusion import BidirectionalHierarchicalEvidenceFusion
from segearth_r2.model.mask_decoder.cs_deg.evidence_guided_attention import fuse_evidence_attention_mask
from segearth_r2.model.mask_decoder.cs_deg.evidence_mask_head import apply_evidence_mask_head


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
            cs_deg_cfg=None,
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
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.SEG_query_embed = nn.Embedding(num_queries + 1, hidden_dim)
        self.new_query_embed = nn.Embedding(1, hidden_dim) # (1, 256)
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

        self.cs_deg_cfg = cs_deg_cfg
        self.cs_deg_enabled = bool(cs_deg_cfg and getattr(cs_deg_cfg, "ENABLED", False))
        if self.cs_deg_enabled:
            prompt_num = int(getattr(cs_deg_cfg, "PROMPT_TOKEN_NUM", 4))
            use_detail = bool(getattr(cs_deg_cfg, "USE_MASK_FEATURE_DETAIL", True))
            use_uncertainty = bool(getattr(cs_deg_cfg, "USE_UNCERTAINTY", True))
            depg_heads = int(getattr(cs_deg_cfg, "NUM_HEADS", nheads))

            self.cs_ret = bool(getattr(cs_deg_cfg, "RET", True))
            self.cs_depg = bool(getattr(cs_deg_cfg, "DEPG", True))
            self.cs_bhef = bool(getattr(cs_deg_cfg, "BHEF", True))
            self.cs_deg_soft_bias = bool(getattr(cs_deg_cfg, "SOFT_BIAS", True))
            self.cs_deg_mask_head = bool(getattr(cs_deg_cfg, "MASK_HEAD", True))
            self.evidence_start_layer = int(getattr(cs_deg_cfg, "EVIDENCE_START_LAYER", 1))
            self.context_lambda = float(getattr(cs_deg_cfg, "CONTEXT_LAMBDA", 1.0))
            self.bias_max = float(getattr(cs_deg_cfg, "BIAS_MAX", 5.0))
            self.bias_warmup_steps = int(getattr(cs_deg_cfg, "BIAS_WARMUP_STEPS", 1000))
            self.log_grad_norm = bool(getattr(cs_deg_cfg, "LOG_GRAD_NORM", True))
            self.log_attention_stats = bool(getattr(cs_deg_cfg, "LOG_ATTENTION_STATS", False))

            if self.cs_ret:
                self.ret = ReferentialEvidenceTokenizer(
                    hidden_dim=hidden_dim,
                    prompt_token_num=prompt_num,
                    num_layers=dec_layers,
                    gate_init=float(getattr(cs_deg_cfg, "RET_GATE_INIT", 0.0)),
                )
            if self.cs_depg:
                self.depg = DenseEvidencePromptGenerator(
                    hidden_dim=hidden_dim,
                    mask_dim=mask_dim,
                    num_levels=self.num_feature_levels,
                    num_heads=depg_heads,
                    use_mask_feature_detail=use_detail,
                    use_uncertainty=use_uncertainty,
                )
            if self.cs_bhef:
                self.bhef = BidirectionalHierarchicalEvidenceFusion(
                    hidden_dim=hidden_dim,
                    num_heads=depg_heads,
                    gate_init=float(getattr(cs_deg_cfg, "BHEF_GATE_INIT", 0.0)),
                )

            self.cs_alpha = nn.Parameter(torch.zeros(1))
            self.mask_gamma = nn.Parameter(
                torch.tensor([float(getattr(cs_deg_cfg, "MASK_GAMMA_INIT", 0.0))])
            )
            self.mask_eta = nn.Parameter(
                torch.tensor([float(getattr(cs_deg_cfg, "MASK_ETA_INIT", 0.0))])
            )


    def forward(self, x, mask_features, mask=None, seg_query=None, SEG_embedding=None, global_step=None):

        return self.forward_woconcat(x, mask_features, mask, seg_query, SEG_embedding, global_step=global_step)

    def forward_woconcat(self, x, mask_features, mask=None, seg_query=None, SEG_embedding=None, global_step=None):
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

        # QxNxC
        if self.use_seg_query:
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        else:
            query_embed = torch.zeros(
                self.new_query_embed.weight.shape[0], bs, self.new_query_embed.weight.shape[-1], 
                device=SEG_embedding.device, dtype=SEG_embedding.dtype
            )
        
        if seg_query is None:
            # output = self.new_query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
            output = SEG_embedding.permute(1, 0, 2)
        else:
            output = seg_query.permute(1, 0, 2) # output: [100, batch_size, mask_dim(256)]
            
        predictions_SEG_class = []        
        predictions_mask = []


        # prediction heads on learnable query features
        if self.use_seg_query:
            SEG_class, outputs_mask, attn_mask = self.forward_prediction_heads(output,
                                                                                mask_features,
                                                                                attn_mask_target_size=
                                                                                size_list[
                                                                                            0],
                                                                                SEG_embedding=SEG_embedding,
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

        cs_step = 0 if global_step is None else int(global_step)
        cs_deg_stats = {}
        final_target_h = None
        final_context_h = None
        final_uncertainty_h = None
        last_sparse_prompts = None

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            cross_attn_mask = attn_mask
            target_lr = context_lr = target_h = context_h = None
            if self.cs_deg_enabled and i >= self.evidence_start_layer:
                evidence_tokens = None
                if self.cs_ret:
                    evidence_tokens = self.ret(output, i, SEG_embedding)

                if self.cs_depg:
                    (
                        sparse_prompts,
                        target_lr,
                        context_lr,
                        _unc_lr,
                        target_h,
                        context_h,
                    ) = self.depg(
                        output,
                        src[level_index],
                        size_list[level_index],
                        outputs_mask,
                        mask_features,
                        evidence_tokens,
                        level_index,
                        compute_high_res=(i == self.num_layers - 1),
                    )
                    last_sparse_prompts = sparse_prompts
                    if target_h is not None:
                        final_target_h = target_h
                        final_context_h = context_h
                        final_uncertainty_h = _unc_lr
                    evidence_tokens = sparse_prompts

                if self.cs_bhef and evidence_tokens is not None:
                    output, evidence_tokens = self.bhef(
                        output, src[level_index], evidence_tokens
                    )

                if self.cs_deg_soft_bias and target_lr is not None:
                    cross_attn_mask, bias_scale = fuse_evidence_attention_mask(
                        attn_mask,
                        target_lr,
                        context_lr,
                        self.cs_alpha,
                        cs_step,
                        self.num_heads,
                        context_lambda=self.context_lambda,
                        bias_max=self.bias_max,
                        bias_warmup_steps=self.bias_warmup_steps,
                        mask_dtype=src[level_index].dtype,
                    )
                    cs_deg_stats["cs_bias_scale"] = bias_scale.detach()

            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=cross_attn_mask,
                memory_key_padding_mask=None,
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            output = self.transformer_ffn_layers[i](output)

            if self.use_seg_query:
                SEG_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                    output, mask_features,
                    attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
                    SEG_embedding=SEG_embedding,
                )
            else:
                SEG_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                    output, mask_features,
                    attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
                    SEG_embedding=None,
                )

            if (
                self.cs_deg_enabled
                and self.cs_deg_mask_head
                and target_lr is not None
            ):
                tgt_map = target_h if (i == self.num_layers - 1 and target_h is not None) else target_lr
                ctx_map = context_h if (i == self.num_layers - 1 and context_h is not None) else context_lr
                outputs_mask = apply_evidence_mask_head(
                    outputs_mask, tgt_map, ctx_map, self.mask_gamma, self.mask_eta
                )

            predictions_SEG_class.append(SEG_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_SEG_class) == self.num_layers + 1

        out = {
            'pred_SEG_logits': predictions_SEG_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_SEG_class, predictions_mask,
            )
        }

        if self.cs_deg_enabled and self.cs_depg:
            if final_target_h is None:
                final_target_h, final_context_h, final_uncertainty_h = self.depg.forward_high_res(
                    output, predictions_mask[-1], mask_features, last_sparse_prompts
                )
            out['pred_evidence_logits'] = final_target_h
            out['pred_context_logits'] = final_context_h
            if final_uncertainty_h is not None:
                out['pred_uncertainty_logits'] = final_uncertainty_h

            cs_deg_stats.update({
                'mask_gamma': self.mask_gamma.detach(),
                'mask_eta': self.mask_eta.detach(),
                'cs_alpha': self.cs_alpha.detach(),
            })
            if self.cs_ret:
                cs_deg_stats['ret_gate'] = self.ret.ret_gate.detach()
            if self.cs_bhef:
                cs_deg_stats['bhef_gate'] = self.bhef.bhef_gate.detach()
            if self.log_grad_norm and self.cs_depg:
                grad_sq = 0.0
                for p in self.depg.parameters():
                    if p.grad is not None:
                        grad_sq += p.grad.data.norm(2).item() ** 2
                cs_deg_stats['depg_grad_norm'] = grad_sq ** 0.5
            out['cs_deg_stats'] = cs_deg_stats

        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size, SEG_embedding=None,
                                ):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        # SEG_embedding = self.SEG_norm(SEG_embedding).expand_as(decoder_output)
        # SEG_embedding = SEG_embedding.expand_as(decoder_output)
        if SEG_embedding is not None:
            if self.seg_proj:
                decoder_seg_output = self.SEG_proj(decoder_output)
            else:
                decoder_seg_output = decoder_output
            if self.seg_norm:
                SEG_embedding = self.SEG_norm(SEG_embedding)
                SEG_embedding = self.seg_proj_after_norm(SEG_embedding)
            SEG_class = torch.einsum('bld,bcd->blc', decoder_seg_output, SEG_embedding)
        else:
            SEG_class = None
        # SEG_class = F.cosine_similarity(decoder_seg_output, SEG_embedding, dim=-1, eps=1e-6).unsqueeze(-1)

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






