import torch.nn as nn


class SegQueryAdapter(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, seg_queries, text_context=None, visual_context=None):
        q = seg_queries.squeeze(1)

        if text_context is not None:
            q = q + text_context
        if visual_context is not None:
            q = q + visual_context

        q_refined = q + self.ffn(self.norm(q))
        q_score = self.score_head(q_refined)

        return q_refined.unsqueeze(1), q_score, {"query_feat": q_refined}

