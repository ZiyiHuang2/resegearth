# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import math
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
                     query_pos: Optional[Tensor] = None,
                     attn_bias: Optional[Tensor] = None):
        effective_mask = memory_mask
        if attn_bias is not None:
            if memory_mask is None:
                effective_mask = attn_bias
            elif memory_mask.dtype == torch.bool:
                if attn_bias.shape != memory_mask.shape:
                    raise ValueError(
                        f"attn_bias shape {tuple(attn_bias.shape)} != memory_mask shape {tuple(memory_mask.shape)}"
                    )
                bias = attn_bias.float()
                mask_float = torch.zeros_like(bias, dtype=torch.float32)
                mask_float = mask_float.masked_fill(memory_mask, float("-inf"))
                effective_mask = mask_float + bias
                effective_mask = effective_mask.masked_fill(memory_mask, float("-inf"))
                effective_mask = effective_mask.to(dtype=tgt.dtype)
            else:
                # Mask2Former cross-attn uses bool memory_mask; float path kept for API completeness.
                effective_mask = memory_mask.to(dtype=attn_bias.dtype) + attn_bias
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=effective_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    attn_bias: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        effective_mask = memory_mask
        if attn_bias is not None:
            if memory_mask is None:
                effective_mask = attn_bias
            elif memory_mask.dtype == torch.bool:
                if attn_bias.shape != memory_mask.shape:
                    raise ValueError(
                        f"attn_bias shape {tuple(attn_bias.shape)} != memory_mask shape {tuple(memory_mask.shape)}"
                    )
                bias = attn_bias.float()
                mask_float = torch.zeros_like(bias, dtype=torch.float32)
                mask_float = mask_float.masked_fill(memory_mask, float("-inf"))
                effective_mask = mask_float + bias
                effective_mask = effective_mask.masked_fill(memory_mask, float("-inf"))
                effective_mask = effective_mask.to(dtype=tgt2.dtype)
            else:
                # Mask2Former cross-attn uses bool memory_mask; float path kept for API completeness.
                effective_mask = memory_mask.to(dtype=attn_bias.dtype) + attn_bias
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=effective_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                attn_bias: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos, attn_bias)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos, attn_bias)


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


_DECODER_ATTN_BIAS_DEBUG_PRINTED = False


class DecoderTokenAttnBias(nn.Module):
    """Token-level text-guided soft bias for decoder cross-attention (additive on attention logits)."""

    def __init__(
        self,
        text_dim: int,
        memory_dim: int = 256,
        bias_dim: int = 128,
        init_std: float = 1e-3,
        max_abs: float = 0.01,
    ):
        super().__init__()
        self.bias_dim = int(bias_dim)
        self.init_std = float(init_std)
        self.max_abs = float(max_abs)
        self.visual_proj = nn.Linear(int(memory_dim), self.bias_dim)
        self.text_proj = nn.Linear(int(text_dim), self.bias_dim)
        hid = max(self.bias_dim, 32)
        self.bias_mlp = nn.Sequential(
            nn.Linear(3 * self.bias_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, 1),
        )
        nn.init.normal_(self.bias_mlp[-1].weight, mean=0.0, std=float(init_std))
        nn.init.zeros_(self.bias_mlp[-1].bias)

    def forward(
        self,
        memory: Tensor,
        text_tokens: Tensor,
        text_mask: Tensor,
        num_heads: int,
        num_queries: int,
        eval_mode: str = "normal",
        force_scale: float = 1.0,
    ):
        """
        memory: [S, B, C]
        text_tokens: [B, L, D]
        text_mask: [B, L] True = valid token
        returns attn_bias [B * num_heads, num_queries, S] or None
        """
        if text_tokens is None or text_mask is None:
            return None, {}
        if memory.dim() != 3 or text_tokens.dim() != 3:
            return None, {}
        S, B, Cm = memory.shape
        B2, L, Dt = text_tokens.shape
        if B != B2:
            return None, {}
        d = self.bias_dim
        M = memory.permute(1, 0, 2)
        V = self.visual_proj(M)
        T = self.text_proj(text_tokens)
        logits = torch.matmul(V, T.transpose(-1, -2)) / math.sqrt(float(d))
        logits = logits.masked_fill(~text_mask[:, None, :], -1e4)
        A = torch.softmax(logits, dim=-1)
        Ctxt = torch.matmul(A, T)
        Z = torch.cat([V, Ctxt, V * Ctxt], dim=-1)
        raw = self.bias_mlp(Z).squeeze(-1)
        raw_std = raw.float().std(dim=-1).mean() if raw.numel() > 0 else raw.new_zeros(())
        raw = raw - raw.mean(dim=-1, keepdim=True)
        P = torch.clamp(raw, -self.max_abs, self.max_abs)
        # fp16: keep max_abs small so bias cannot numerically overwhelm hard -inf masks after merge
        # (CrossAttentionLayer re-applies masked_fill(memory_mask, -inf); large max_abs is still discouraged.)
        ma = float(self.max_abs)
        if P.dtype == torch.float16 and ma >= 0.1:
            raise AssertionError(
                f"decoder_attn_bias_max_abs={ma} is too large for fp16 hard-mask safety in this PoC; "
                "use <=0.01 or run bias path in bf32/fp32."
            )
        em = str(eval_mode if eval_mode is not None else "normal").strip().lower()
        if em not in ("normal", "bypass", "force_scale"):
            em = "normal"
        if em == "bypass":
            P = torch.zeros_like(P)
        elif em == "force_scale":
            fs = float(force_scale)
            P = P * torch.tensor(fs, device=P.device, dtype=P.dtype)
        bias = P[:, None, :].expand(B, num_queries, S).contiguous()
        bias = bias[:, None, :, :].expand(B, num_heads, num_queries, S).contiguous()
        attn_bias = bias.reshape(B * num_heads, num_queries, S)
        stats = {
            "decoder_attn_bias_abs_mean": P.detach().abs().mean(),
            "decoder_attn_bias_raw_std": raw_std.detach() if torch.is_tensor(raw_std) else raw_std,
            "decoder_attn_bias_max": P.detach().max(),
            "decoder_attn_bias_min": P.detach().min(),
            "decoder_attn_bias_enabled": memory.new_tensor(1.0),
            # [B, S] 未 expand 到 head/query；供 last3 等多层 ranking loss（每层单独一条，勿只取 last）
            "P_bias_flat": P,
        }
        global _DECODER_ATTN_BIAS_DEBUG_PRINTED
        if not _DECODER_ATTN_BIAS_DEBUG_PRINTED:
            _DECODER_ATTN_BIAS_DEBUG_PRINTED = True
            p_mean = float(P.detach().abs().mean().item())
            p_max = float(P.detach().max().item())
            p_min = float(P.detach().min().item())
            ab_mean = float(attn_bias.detach().abs().mean().item())
            print(
                "[DEBUG][DecoderAttnBias] "
                f"eval_mode={eval_mode!r} em={em!r} force_scale={float(force_scale):.6g} "
                f"P_abs_mean={p_mean:.6g} P_max={p_max:.6g} P_min={p_min:.6g} "
                f"attn_bias_abs_mean={ab_mean:.6g}",
                flush=True,
            )
        return attn_bias, stats


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
                pos=pos[level_index], query_pos=query_embed,
                attn_bias=None,
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
            decoder_attn_bias_apply_layers=None,
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
        self.decoder_attn_bias_apply_layers = decoder_attn_bias_apply_layers
        self._warn_decoder_attn_bias_no_tokens = False
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

    def _decoder_bias_layer_active(self, layer_idx: int) -> bool:
        spec = self.decoder_attn_bias_apply_layers
        if spec is None:
            return False
        if spec == "last3":
            return layer_idx >= self.num_layers - 3
        return False

    def forward(
        self,
        x,
        mask_features,
        mask=None,
        seg_query=None,
        SEG_embedding=None,
        text_tokens=None,
        text_mask=None,
        dac_eval_mode: str = "normal",
        dac_force_scale: float = 1.0,
    ):

        return self.forward_woconcat(
            x,
            mask_features,
            mask,
            seg_query,
            SEG_embedding,
            text_tokens=text_tokens,
            text_mask=text_mask,
            dac_eval_mode=dac_eval_mode,
            dac_force_scale=dac_force_scale,
        )

    def forward_woconcat(
        self,
        x,
        mask_features,
        mask=None,
        seg_query=None,
        SEG_embedding=None,
        text_tokens=None,
        text_mask=None,
        dac_eval_mode: str = "normal",
        dac_force_scale: float = 1.0,
    ):
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

        dac_mod = getattr(self, "decoder_token_attn_bias", None)
        qdti_mod = getattr(self, "qdti_core", None)
        dac_stats_accum = []
        self._last_decoder_attn_bias_log = None
        self._last_qdti_log = None
        # 每次前向清空：收集所有启用 bias 的层的 P 与空间尺寸，供 ranking loss 逐层监督（勿只存最后一层）
        self._last_decoder_attn_bias_maps_for_rank = []
        self._last_qdti_maps_for_rank = []
        use_qdti_mask_feedback = bool(getattr(self, "use_qdti_mask_feedback", False))
        active_bias_mod = qdti_mod if qdti_mod is not None else dac_mod
        if active_bias_mod is not None and self.decoder_attn_bias_apply_layers:
            if text_tokens is None or text_mask is None:
                if not self._warn_decoder_attn_bias_no_tokens:
                    self._warn_decoder_attn_bias_no_tokens = True
                    print(
                        "[WARNING][DecoderAttnBias] text_tokens/text_mask unavailable; "
                        "decoder attention bias bypassed for this run."
                    )

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            layer_attn_bias = None
            if (
                active_bias_mod is not None
                and self._decoder_bias_layer_active(i)
                and text_tokens is not None
                and text_mask is not None
            ):
                # 与即将进入的 cross-attn 使用同一 memory_mask：True=禁止 attend。
                # ranking loss 只应在「至少一个 query 仍可 attend」的 memory 位置上统计 P_bias。
                nh = self.num_heads
                hl, wl = size_list[level_index]
                Ssz = hl * wl
                Qn = attn_mask.shape[1]
                spatial_allowed_flat = None
                if (
                    attn_mask.dim() == 3
                    and nh > 0
                    and attn_mask.shape[0] % nh == 0
                    and attn_mask.shape[2] == Ssz
                ):
                    B_attn = attn_mask.shape[0] // nh
                    am = attn_mask.view(B_attn, nh, Qn, Ssz)
                    # PyTorch <2.0: Tensor.any() does not accept dim as a tuple.
                    spatial_allowed_flat = (~am).any(dim=1).any(dim=1)

                mask_fb = None
                if use_qdti_mask_feedback and qdti_mod is not None and outputs_mask is not None:
                    om = outputs_mask.detach()
                    if om.dim() == 4 and om.shape[-2] * om.shape[-1] != Ssz:
                        om = F.interpolate(om.float(), size=(hl, wl), mode="bilinear", align_corners=False)
                    mask_fb = torch.sigmoid(om).flatten(2)

                if qdti_mod is not None:
                    qdti_step = getattr(self, "_qdti_global_step", None)
                    qdti_warmup = int(getattr(self, "_qdti_warmup_steps", 0) or 0)
                    attn_bias_t, st = qdti_mod(
                        src[level_index],
                        output,
                        text_tokens,
                        text_mask,
                        self.num_heads,
                        layer_idx=i,
                        global_step=qdti_step,
                        warmup_steps=qdti_warmup,
                        eval_mode=str(dac_eval_mode),
                        force_scale=float(dac_force_scale),
                        mask_feedback=mask_fb,
                    )
                else:
                    attn_bias_t, st = dac_mod(
                        src[level_index],
                        text_tokens,
                        text_mask,
                        self.num_heads,
                        output.shape[0],
                        eval_mode=str(dac_eval_mode),
                        force_scale=float(dac_force_scale),
                    )
                P_flat = st.pop("P_bias_flat", None) if isinstance(st, dict) else None
                P_qs = st.pop("P_bias_qs", None) if isinstance(st, dict) else None
                if P_qs is not None:
                    P_flat = P_qs
                if P_flat is not None:
                    per_query = P_flat.dim() == 3
                    shape_ok = (
                        (per_query and P_flat.shape[2] == Ssz)
                        or (not per_query and P_flat.dim() == 2 and P_flat.shape[1] == Ssz)
                    )
                    if not shape_ok:
                        msg = (
                            f"[QDTICore] P shape {tuple(P_flat.shape)} invalid for layer={i} "
                            f"expected [B,Q,{Ssz}] or [B,{Ssz}]."
                        )
                        if self.training:
                            raise ValueError(msg)
                        print("[WARNING]" + msg + " Skip rank map append in eval.", flush=True)
                        P_flat = None
                    if P_flat is not None:
                        Bp = P_flat.shape[0]
                        acc_spatial_shape = (Bp, Ssz)
                        if spatial_allowed_flat is None:
                            msg = (
                                f"[DecoderAttnBiasRank] cannot derive spatial_allowed from attn_mask "
                                f"shape={tuple(attn_mask.shape)} nh={nh} Ssz={Ssz} layer={i}."
                            )
                            if self.training:
                                raise ValueError(msg)
                            print("[WARNING]" + msg + " Use all-accessible fallback only in eval.", flush=True)
                            spatial_allowed_flat = torch.ones(
                                acc_spatial_shape, device=P_flat.device, dtype=torch.bool
                            )
                        elif spatial_allowed_flat.shape != acc_spatial_shape:
                            if self.training:
                                raise ValueError(
                                    f"[DecoderAttnBiasRank] spatial_allowed {tuple(spatial_allowed_flat.shape)} "
                                    f"!= expected {acc_spatial_shape} layer={i}."
                                )
                            spatial_allowed_flat = torch.ones(
                                acc_spatial_shape, device=P_flat.device, dtype=torch.bool
                            )
                        if per_query:
                            Bp, Qp, _ = P_flat.shape
                            spatial_allowed_hw = spatial_allowed_flat.view(Bp, hl, wl).detach()
                            P_store = P_flat if self.training else P_flat.detach()
                            P_map = P_store.view(Bp, Qp, hl, wl)
                        else:
                            spatial_allowed_hw = spatial_allowed_flat.view(P_flat.shape[0], hl, wl).detach()
                            P_store = P_flat if self.training else P_flat.detach()
                            P_map = P_store.view(P_flat.shape[0], hl, wl)
                        rank_entry = {
                            "P": P_map,
                            "H": hl,
                            "W": wl,
                            "layer_idx": i,
                            "spatial_allowed": spatial_allowed_hw,
                            "per_query": per_query,
                        }
                        self._last_decoder_attn_bias_maps_for_rank.append(rank_entry)
                        if qdti_mod is not None:
                            self._last_qdti_maps_for_rank.append(rank_entry)
                if attn_bias_t is not None:
                    assert attn_bias_t.shape == attn_mask.shape, (
                        f"attn_bias shape {tuple(attn_bias_t.shape)} != attn_mask shape {tuple(attn_mask.shape)}"
                    )
                    layer_attn_bias = attn_bias_t
                    dac_stats_accum.append(st)

            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed,
                attn_bias=layer_attn_bias,
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

            if self.use_seg_query:
                SEG_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                    output, mask_features,
                    attn_mask_target_size=
                    size_list[(
                                      i + 1) % self.num_feature_levels],
                    SEG_embedding=SEG_embedding,
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

        if dac_stats_accum:
            last_st = dac_stats_accum[-1]
            self._last_decoder_attn_bias_log = last_st
            if qdti_mod is not None:
                self._last_qdti_log = last_st
        elif active_bias_mod is not None and self.decoder_attn_bias_apply_layers:
            zv = predictions_mask[-1].new_zeros(())
            zo = predictions_mask[-1].new_tensor(0.0)
            self._last_decoder_attn_bias_log = {
                "decoder_attn_bias_abs_mean": zv,
                "decoder_attn_bias_raw_std": zv,
                "decoder_attn_bias_max": zv,
                "decoder_attn_bias_min": zv,
                "decoder_attn_bias_enabled": zo,
            }

        out = {
            'pred_SEG_logits': predictions_SEG_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_SEG_class, predictions_mask,
            )
        }
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






