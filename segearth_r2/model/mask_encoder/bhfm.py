from typing import Optional, Tuple

import copy
import torch
import torch.nn as nn


class BHFMLayer(nn.Module):
    """Bi-directional Hybrid Fusion Module.

    Args:
        embed_dim: visual branch channel dim (Swin block dim).
        text_dim: text feature dim. If None, defaults to embed_dim.
        num_heads: cross-attn heads.
        mlp_ratio: MLP expansion ratio.
    """

    def __init__(
        self,
        embed_dim: int,
        text_dim: Optional[int] = None,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.text_dim = int(text_dim) if text_dim is not None else int(embed_dim)
        hidden_dim = int(self.embed_dim * mlp_ratio)

        # Text/vision dim adapter (text dim can differ from stage dim)
        self.text_in_proj = nn.Identity() if self.text_dim == self.embed_dim else nn.Linear(self.text_dim, self.embed_dim)
        self.text_out_proj_back = nn.Identity() if self.text_dim == self.embed_dim else nn.Linear(self.embed_dim, self.text_dim)

        # Vision -> Text cross-attn
        self.vis_q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.text_kv_proj = nn.Linear(self.embed_dim, self.embed_dim * 2)
        self.mhca_v = nn.MultiheadAttention(self.embed_dim, num_heads, batch_first=True)
        self.vis_out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # Frozen branch + tunable branch
        self.frozen_ln = nn.LayerNorm(self.embed_dim)
        self.frozen_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.embed_dim),
        )
        self.tunable_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.embed_dim),
        )
        self.w_v = nn.Parameter(torch.tensor(0.0))

        # Text -> Vision cross-attn
        self.text_q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.mhca_t = nn.MultiheadAttention(self.embed_dim, num_heads, batch_first=True)
        self.text_out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_t = nn.Parameter(torch.tensor(0.0))

        self.freeze_frozen_branch()

    def freeze_frozen_branch(self) -> None:
        for p in self.frozen_ln.parameters():
            p.requires_grad = False
        for p in self.frozen_mlp.parameters():
            p.requires_grad = False

    def init_frozen_from_block(self, norm2: nn.Module, mlp: nn.Module) -> None:
        """Initialize frozen branch from existing Swin block norm/mlp (avoid random frozen params)."""
        self.frozen_ln.load_state_dict(copy.deepcopy(norm2.state_dict()))
        self.frozen_mlp = copy.deepcopy(mlp)
        self.freeze_frozen_branch()

    def forward(
        self,
        f_i: torch.Tensor,
        t_i: torch.Tensor,
        f_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert f_i.ndim == 3 and f_in.ndim == 3, f"Expected [B,N,C], got f_i={tuple(f_i.shape)}, f_in={tuple(f_in.shape)}"
        assert t_i.ndim == 3, f"Expected text [B,L,Ct], got {tuple(t_i.shape)}"
        assert f_i.shape == f_in.shape, f"f_i/f_in shape mismatch: {tuple(f_i.shape)} vs {tuple(f_in.shape)}"
        assert f_i.size(-1) == self.embed_dim, f"visual dim mismatch: {f_i.size(-1)} vs {self.embed_dim}"

        t_embed = self.text_in_proj(t_i)
        assert t_embed.size(-1) == self.embed_dim, "text projection dim mismatch"

        # Vision queries Text
        q_v = self.vis_q_proj(f_i)
        k_t, v_t = self.text_kv_proj(t_embed).chunk(2, dim=-1)
        attn_v_out, _ = self.mhca_v(query=q_v, key=k_t, value=v_t)
        attn_v_out = q_v + attn_v_out

        f_mid = self.vis_out_proj(attn_v_out)
        f_added = f_i + f_mid + f_in

        frozen_out = self.frozen_mlp(self.frozen_ln(f_added))
        tune_out = self.tunable_mlp(f_added)
        f_out = frozen_out + self.w_v * tune_out

        # Text queries Vision
        q_t = self.text_q_proj(t_embed)
        attn_t_out, _ = self.mhca_t(query=q_t, key=attn_v_out, value=attn_v_out)
        attn_t_out = q_t + attn_t_out
        t_mid = self.text_out_proj(attn_t_out)
        t_embed_out = t_embed + self.w_t * t_mid
        t_out = self.text_out_proj_back(t_embed_out)
        return f_out, t_out
