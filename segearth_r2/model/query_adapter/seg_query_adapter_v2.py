import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegQueryAdapterV2(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_local_queries=16,
        text_dim=None,
        use_text_cross_attn_refine=False,
        use_ffn=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_local_queries = int(num_local_queries)
        self.use_text_cross_attn_refine = bool(use_text_cross_attn_refine)
        self.use_ffn = bool(use_ffn)

        self.text_dim = text_dim if text_dim is not None else hidden_dim

        self.query_bank = nn.Parameter(
            torch.randn(self.num_local_queries, hidden_dim) * 0.02
        )

        if self.text_dim != hidden_dim:
            self.text_proj = nn.Linear(self.text_dim, hidden_dim)
        else:
            self.text_proj = nn.Identity()

        self.lang_k = nn.Linear(hidden_dim, hidden_dim)
        self.lang_v = nn.Linear(hidden_dim, hidden_dim)
        self.seed_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.cross_attn = None
        if self.use_text_cross_attn_refine:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                batch_first=True,
            )

        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        seed_queries,               # [Nq, 1, C]
        text_context=None,          # [Nq, C] optional
        text_tokens=None,           # [Nq, T, Ct]
        text_key_padding_mask=None, # [Nq, T], True means padding
        visual_context=None,        # [Nq, C] optional
    ):
        assert seed_queries.dim() == 3, f"expected seed_queries [Nq,1,C], got {seed_queries.shape}"
        assert seed_queries.shape[1] == 1, f"expected seed_queries second dim == 1, got {seed_queries.shape}"
        assert text_tokens is not None and text_tokens.dim() == 3, \
            f"expected text_tokens [Nq,T,Ct], got {None if text_tokens is None else text_tokens.shape}"

        q_seed = seed_queries.squeeze(1)  # [Nq, C]
        Nq, C = q_seed.shape

        text_tokens = self.text_proj(text_tokens)   # [Nq, T, C]
        text_tokens = F.gelu(text_tokens)

        assert text_tokens.shape[0] == Nq, \
            f"batch mismatch: seed_queries batch={Nq}, text_tokens batch={text_tokens.shape[0]}"
        assert text_tokens.shape[2] == C, \
            f"channel mismatch: seed dim={C}, text dim={text_tokens.shape[2]}"

        q_bank = self.query_bank.unsqueeze(0).expand(Nq, -1, -1)  # [Nq, K, C]

        lang_k = self.lang_k(text_tokens)  # [Nq, T, C]
        lang_v = self.lang_v(text_tokens)  # [Nq, T, C]

        attn_logits = torch.matmul(q_bank, lang_k.transpose(-1, -2)) / math.sqrt(C)  # [Nq, K, T]

        if text_key_padding_mask is not None:
            assert text_key_padding_mask.shape == (Nq, text_tokens.shape[1]), \
                f"expected key_padding_mask shape {(Nq, text_tokens.shape[1])}, got {text_key_padding_mask.shape}"
            attn_logits = attn_logits.masked_fill(
                text_key_padding_mask.unsqueeze(1),  # [Nq,1,T] -> broadcast to [Nq,K,T]
                float("-inf")
            )

        A_bi = torch.softmax(attn_logits, dim=-1)   # [Nq, K, T]
        q_lang = torch.matmul(A_bi, lang_v)         # [Nq, K, C]

        sub_queries = q_lang + self.seed_proj(q_seed).unsqueeze(1)

        if text_context is not None:
            assert text_context.shape == (Nq, C), \
                f"expected text_context {(Nq, C)}, got {text_context.shape}"
            sub_queries = sub_queries + text_context.unsqueeze(1)

        if visual_context is not None:
            assert visual_context.shape == (Nq, C), \
                f"expected visual_context {(Nq, C)}, got {visual_context.shape}"
            sub_queries = sub_queries + visual_context.unsqueeze(1)

        if self.cross_attn is not None:
            attn_out, _ = self.cross_attn(
                sub_queries,
                text_tokens,
                text_tokens,
                key_padding_mask=text_key_padding_mask,
                need_weights=False,
            )
            sub_queries = sub_queries + attn_out

        if self.use_ffn:
            sub_queries = sub_queries + self.ffn(self.norm(sub_queries))

        query_scores = self.score_head(sub_queries).squeeze(-1)  # [Nq, K]

        aux = {
            "query_feat": sub_queries,
            "seed_feat": q_seed,
            "query_bank": self.query_bank,
            "lqca_attn": A_bi,
        }
        return sub_queries, query_scores, aux