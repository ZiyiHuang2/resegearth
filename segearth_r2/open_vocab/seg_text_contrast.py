import torch
import torch.nn as nn
import torch.nn.functional as F


class SegTextContrastHead(nn.Module):
    def __init__(self, dim: int = 256, proj_dim: int = 256, tau: float = 0.07):
        super().__init__()
        self.tau = tau
        self.q_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, proj_dim),
        )
        self.t_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, proj_dim),
        )

    def forward(self, seg_query_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        seg_query_embeds: [N, D]
        text_embeds:      [N, D]
        """
        q = F.normalize(self.q_proj(seg_query_embeds), dim=-1)
        t = F.normalize(self.t_proj(text_embeds), dim=-1)

        logits = torch.matmul(q, t.t()) / self.tau  # [N, N]
        labels = torch.arange(logits.size(0), device=logits.device)

        loss_q2t = F.cross_entropy(logits, labels)
        loss_t2q = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_q2t + loss_t2q)