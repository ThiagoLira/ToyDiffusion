import torch
import torch.nn as nn
import math


class TimeConditionedMLP(nn.Module):
    """Point-wise MLP: takes (batch,2) coords + (batch,) time -> (batch,2) output.

    Time is projected through a small learned embedding MLP, then concatenated
    with spatial coords. Uses 4 hidden layers with SiLU activations.
    """

    def __init__(self, hidden_dim=256, time_emb_dim=64):
        super().__init__()
        # Learned time embedding: scalar t -> time_emb_dim vector
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(2 + time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, t):
        """x: (batch,2), t: (batch,) -> (batch,2)"""
        t_emb = self.time_mlp(t[:, None])
        inp = torch.cat([x, t_emb], dim=-1)
        return self.net(inp)
