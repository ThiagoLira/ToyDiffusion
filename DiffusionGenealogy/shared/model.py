import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for continuous time values, following the
    Transformer positional encoding pattern (Vaswani et al., 2017).
    Maps scalar t -> dense vector of dimension `dim`."""

    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """t: (batch,) tensor of time values (any range, typically [0,1] or [0,T])."""
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class TimeConditionedMLP(nn.Module):
    """Point-wise MLP: takes (batch,2) coords + (batch,) time -> (batch,2) output.
    Uses sinusoidal time embedding concatenated with coords, then 4 hidden layers."""

    def __init__(self, hidden_dim=256, time_emb_dim=64):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
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
        t_emb = self.time_emb(t)
        inp = torch.cat([x, t_emb], dim=-1)
        return self.net(inp)
