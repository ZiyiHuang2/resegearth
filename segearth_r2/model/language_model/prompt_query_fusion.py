"""Dual-granularity prompt fusion and SEG query refinement (Stage 3 DGP)."""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def pack_seg_hidden_states_bq(
    hidden_states: torch.Tensor,
    seg_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather [SEG] hidden states into [B, Q, H] with explicit validity mask."""
    if hidden_states.dim() != 3:
        raise ValueError(f"hidden_states must be [B,L,H], got {tuple(hidden_states.shape)}")
    if seg_mask.shape[:2] != hidden_states.shape[:2]:
        raise ValueError(
            f"seg_mask {tuple(seg_mask.shape)} incompatible with hidden_states {tuple(hidden_states.shape)}"
        )
    B, _, H = hidden_states.shape
    seg_mask = seg_mask.bool()
    q_counts = seg_mask.sum(dim=1)
    Q_max = max(int(q_counts.max().item()), 1)
    out = hidden_states.new_zeros(B, Q_max, H)
    valid = torch.zeros(B, Q_max, dtype=torch.bool, device=hidden_states.device)
    for b in range(B):
        pos = seg_mask[b]
        n = int(pos.sum().item())
        if n > 0:
            out[b, :n] = hidden_states[b, pos]
            valid[b, :n] = True
    return out, valid


def expand_bq_for_mask_num(
    tensor_bq: torch.Tensor,
    mask_num,
) -> torch.Tensor:
    """Repeat each batch row by mask_num -> [B_exp, Q, D]."""
    if tensor_bq.dim() != 3:
        raise ValueError(f"tensor_bq must be [B,Q,D], got {tuple(tensor_bq.shape)}")
    if mask_num is None:
        return tensor_bq
    mn = torch.as_tensor(mask_num, device=tensor_bq.device, dtype=torch.long).flatten()
    if mn.numel() == 0:
        return tensor_bq
    if mn.numel() != tensor_bq.shape[0]:
        raise ValueError(
            f"mask_num length {mn.numel()} != batch {tensor_bq.shape[0]} for expand_bq_for_mask_num"
        )
    return torch.repeat_interleave(tensor_bq, mn, dim=0)


def expand_bp_for_mask_num(
    tensor_bp: torch.Tensor,
    mask_num,
) -> torch.Tensor:
    """Repeat each batch row by mask_num -> [B_exp, P, D]."""
    if tensor_bp.dim() != 3:
        raise ValueError(f"tensor_bp must be [B,P,D], got {tuple(tensor_bp.shape)}")
    if mask_num is None:
        return tensor_bp
    mn = torch.as_tensor(mask_num, device=tensor_bp.device, dtype=torch.long).flatten()
    if mn.numel() == 0:
        return tensor_bp
    if mn.numel() != tensor_bp.shape[0]:
        raise ValueError(
            f"mask_num length {mn.numel()} != batch {tensor_bp.shape[0]} for expand_bp_for_mask_num"
        )
    return torch.repeat_interleave(tensor_bp, mn, dim=0)


def _masked_mean_pool(
    hidden_states: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-batch masked mean -> [B, H]."""
    B, _, H = hidden_states.shape
    mask = token_mask.bool().unsqueeze(-1).to(dtype=hidden_states.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    pooled = (hidden_states * mask).sum(dim=1) / denom
    return pooled


def _gather_padded_tokens(
    hidden_states: torch.Tensor,
    token_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather masked tokens per batch, pad to Pl_max -> [B, Pl, H], [B, Pl]."""
    B, _, H = hidden_states.shape
    token_mask = token_mask.bool()
    pl_max = max(int(token_mask.sum(dim=1).max().item()), 1)
    out = hidden_states.new_zeros(B, pl_max, H)
    valid = torch.zeros(B, pl_max, dtype=torch.bool, device=hidden_states.device)
    for b in range(B):
        pos = token_mask[b]
        n = int(pos.sum().item())
        if n > 0:
            out[b, :n] = hidden_states[b, pos]
            valid[b, :n] = True
    return out, valid


def _fallback_expr_from_refer_ids(
    hidden_states: torch.Tensor,
    token_refer_id,
    embed_fn,
    pad_token_id: Optional[int],
) -> torch.Tensor:
    """Debug/fallback only: static refer embedding mean-pool -> [B, H]."""
    B, _, H = hidden_states.shape
    device = hidden_states.device
    dtype = hidden_states.dtype

    if token_refer_id is None:
        return hidden_states.new_zeros(B, H)

    if torch.is_tensor(token_refer_id):
        if token_refer_id.dim() == 1:
            refer_items = [token_refer_id]
        elif token_refer_id.dim() == 2:
            refer_items = [token_refer_id[i] for i in range(token_refer_id.shape[0])]
        else:
            raise ValueError(f"Unsupported token_refer_id dim: {token_refer_id.dim()}")
    elif isinstance(token_refer_id, (list, tuple)):
        refer_items = list(token_refer_id)
    else:
        refer_items = [None] * B

    if len(refer_items) < B:
        refer_items.extend([None] * (B - len(refer_items)))
    refer_items = refer_items[:B]

    rows = []
    for refer_ids in refer_items:
        if refer_ids is None or (torch.is_tensor(refer_ids) and refer_ids.numel() == 0):
            rows.append(torch.zeros(H, device=device, dtype=dtype))
            continue
        refer_ids = refer_ids.to(device=device, dtype=torch.long).view(-1)
        if pad_token_id is not None:
            refer_ids = refer_ids[refer_ids.ne(pad_token_id)]
        if refer_ids.numel() == 0:
            rows.append(torch.zeros(H, device=device, dtype=dtype))
            continue
        emb = embed_fn(refer_ids)
        if emb is None:
            rows.append(torch.zeros(H, device=device, dtype=dtype))
        elif emb.dim() == 1:
            rows.append(emb.to(dtype=dtype))
        else:
            rows.append(emb.mean(dim=0).to(dtype=dtype))
    return torch.stack(rows, dim=0)


class DualGranularityPromptAdapter(nn.Module):
    """Build P_g / P_l from MLLM hidden states (not static token embeddings)."""

    def __init__(self, llm_dim: int, fuse_dim: int, pg_tokens: int = 1):
        super().__init__()
        self.llm_dim = int(llm_dim)
        self.fuse_dim = int(fuse_dim)
        self.pg_tokens = max(int(pg_tokens), 1)
        self.proj = nn.Linear(self.llm_dim, self.fuse_dim)
        if self.pg_tokens > 1:
            self.pg_queries = nn.Parameter(torch.randn(self.pg_tokens, self.fuse_dim) * 0.02)
        self._warned_fallback = False

    def _build_expr_token_mask(
        self,
        attention_mask: torch.Tensor,
        seg_mask: torch.Tensor,
        image_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        expr_mask = attention_mask.bool() & ~seg_mask.bool()
        if image_mask is not None:
            expr_mask = expr_mask & ~image_mask.bool()
        return expr_mask

    def _build_p_g(
        self,
        hidden_states: torch.Tensor,
        expr_mask: torch.Tensor,
        fallback_expr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = hidden_states.shape[0]
        if self.pg_tokens == 1:
            pooled = _masked_mean_pool(hidden_states, expr_mask)
            if fallback_expr is not None:
                empty = ~expr_mask.any(dim=1)
                if empty.any():
                    if not self._warned_fallback:
                        self._warned_fallback = True
                        print(
                            "[WARNING][DGP] expression span empty; using token_refer_id fallback for those rows."
                        )
                    pooled = pooled.clone()
                    pooled[empty] = fallback_expr[empty]
            p_g = self.proj(pooled).unsqueeze(1)
            mask = torch.ones(B, 1, dtype=torch.bool, device=hidden_states.device)
            return p_g, mask

        hidden_proj = self.proj(hidden_states)
        queries = self.pg_queries.unsqueeze(0).expand(B, -1, -1)
        logits = torch.matmul(queries, hidden_proj.transpose(1, 2)) / (self.fuse_dim ** 0.5)
        logits = logits.masked_fill(~expr_mask.unsqueeze(1), -1e4)
        attn = torch.softmax(logits, dim=-1)
        p_g = torch.matmul(attn, hidden_proj)
        mask = torch.ones(B, self.pg_tokens, dtype=torch.bool, device=hidden_states.device)
        return p_g, mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        seg_mask: torch.Tensor,
        target_phrase_mask: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
        token_refer_id=None,
        embed_fn=None,
        pad_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            P_g: [B, Pg, fuse_dim]
            P_l: [B, Pl, fuse_dim]
            prompt_tokens: [B, Pg+Pl, fuse_dim]
            prompt_mask: [B, Pg+Pl]
        """
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states must be [B,L,H_lm], got {tuple(hidden_states.shape)}")

        expr_mask = self._build_expr_token_mask(attention_mask, seg_mask, image_mask)
        detail_mask = target_phrase_mask.bool() if target_phrase_mask is not None else seg_mask.bool()

        fallback_expr = None
        if token_refer_id is not None and embed_fn is not None:
            fallback_expr = _fallback_expr_from_refer_ids(
                hidden_states, token_refer_id, embed_fn, pad_token_id
            )

        p_g, pg_mask = self._build_p_g(hidden_states, expr_mask, fallback_expr=fallback_expr)

        detail_hidden, pl_mask = _gather_padded_tokens(hidden_states, detail_mask)
        p_l = self.proj(detail_hidden)

        prompt_tokens = torch.cat([p_g, p_l], dim=1)
        prompt_mask = torch.cat([pg_mask, pl_mask], dim=1)
        return p_g, p_l, prompt_tokens, prompt_mask


class PromptAwareQueryRefiner(nn.Module):
    """Refine projected SEG queries with dual-granularity prompt tokens."""

    def __init__(self, dim: int = 256, hidden_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.dim = int(dim)
        hid = max(int(hidden_dim), 32)
        self.pre_norm_q = nn.LayerNorm(self.dim)
        self.pre_norm_kv = nn.LayerNorm(self.dim)
        self.cross_attn = nn.MultiheadAttention(
            self.dim, num_heads, dropout=0.0, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, hid),
            nn.ReLU(),
            nn.Linear(hid, self.dim),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        seg_embedding: torch.Tensor,
        prompt_tokens: torch.Tensor,
        seg_query_mask: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            seg_embedding: [B, Q, 256]
            prompt_tokens: [B, P, 256]

        Returns:
            Q_ref: [B, Q, 256]
        """
        if seg_embedding.dim() != 3 or prompt_tokens.dim() != 3:
            raise ValueError(
                f"seg_embedding {tuple(seg_embedding.shape)} and prompt_tokens "
                f"{tuple(prompt_tokens.shape)} must be [B,Q,D] and [B,P,D]"
            )
        if seg_embedding.shape[0] != prompt_tokens.shape[0]:
            raise ValueError(
                f"batch mismatch: seg_embedding B={seg_embedding.shape[0]} "
                f"prompt_tokens B={prompt_tokens.shape[0]}"
            )
        if seg_embedding.shape[-1] != self.dim or prompt_tokens.shape[-1] != self.dim:
            raise ValueError(
                f"expected dim={self.dim}, got seg={seg_embedding.shape[-1]} prompt={prompt_tokens.shape[-1]}"
            )

        q = self.pre_norm_q(seg_embedding)
        kv = self.pre_norm_kv(prompt_tokens)
        key_padding_mask = None
        if prompt_mask is not None:
            key_padding_mask = ~prompt_mask.bool()

        attn_out, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask)
        delta = self.ffn(attn_out)
        q_ref = seg_embedding + self.gate * delta

        if seg_query_mask is not None:
            valid = seg_query_mask.bool().unsqueeze(-1)
            q_ref = torch.where(valid, q_ref, seg_embedding)
        return q_ref
