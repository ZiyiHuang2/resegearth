import torch
import torch.nn as nn


class SegQueryAdapterV2(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_local_queries=4,
        text_dim=None,
        use_text_cross_attn_refine=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_local_queries = int(num_local_queries)
        self.use_text_cross_attn_refine = use_text_cross_attn_refine

        self.local_offsets = nn.Parameter(torch.zeros(self.num_local_queries, hidden_dim))
        nn.init.normal_(self.local_offsets, std=0.02)

        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        ctx_dim = hidden_dim if text_dim is None else text_dim
        self.hidden_dim = hidden_dim
        self.text_dim = text_dim if text_dim is not None else hidden_dim

        if self.text_dim != hidden_dim:
            self.text_proj = nn.Linear(self.text_dim, hidden_dim)
        else:
            self.text_proj = None
        self.text_to_scale_shift = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        self.text_proj = None
        if ctx_dim != hidden_dim:
            self.text_proj = nn.Linear(ctx_dim, hidden_dim)

        self.cross_attn = None
        if self.use_text_cross_attn_refine:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=4, batch_first=True
            )

        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        seed_queries,
        text_context=None,
        text_tokens=None,
        text_key_padding_mask=None,
        visual_context=None,
    ):
        # seed_queries: [Nq, 1, C]
        q_seed = seed_queries.squeeze(1)  # [Nq, C]
        sub_queries = q_seed.unsqueeze(1) + self.local_offsets.unsqueeze(0)  # [Nq, K, C]

        if text_context is not None:
            text_ctx = text_context
            if self.text_proj is not None:
                proj0 = self.text_proj
    
                text_ctx = text_ctx.to(device=proj0.weight.device, dtype=proj0.weight.dtype)
                text_ctx = self.text_proj(text_ctx)
            if text_ctx.shape[-1] != self.hidden_dim:
                    raise ValueError(
                        f"text_ctx dim mismatch after text_proj: got {text_ctx.shape[-1]}, "
                        f"expected {self.hidden_dim}"
                    )

            ss0 = self.text_to_scale_shift[0]
            text_ctx = text_ctx.to(device=ss0.weight.device, dtype=ss0.weight.dtype)

            scale_shift = self.text_to_scale_shift(text_ctx)  # [Nq, 2C]
            scale, shift = scale_shift.chunk(2, dim=-1)

            scale = scale.to(dtype=sub_queries.dtype, device=sub_queries.device)
            shift = shift.to(dtype=sub_queries.dtype, device=sub_queries.device)

            sub_queries = sub_queries + scale.unsqueeze(1) * self.norm(sub_queries) + shift.unsqueeze(1)

        if visual_context is not None:
            sub_queries = sub_queries + visual_context.unsqueeze(1)

        if self.cross_attn is not None and text_tokens is not None:
            q = sub_queries
            k = text_tokens
            v = text_tokens
            attn_out, _ = self.cross_attn(
                q, k, v, key_padding_mask=text_key_padding_mask, need_weights=False
            )
            sub_queries = sub_queries + attn_out

        sub_queries = sub_queries + self.ffn(self.norm(sub_queries))
        query_scores = self.score_head(sub_queries)  # [Nq, K, 1]

        aux = {
            "query_feat": sub_queries,
            "seed_feat": q_seed,
            "local_offsets": self.local_offsets,
        }
        return sub_queries, query_scores, aux

