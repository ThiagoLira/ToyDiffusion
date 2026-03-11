import torch
import torch.nn as nn
import copy


class ResBlock(nn.Module):
    """Residual block: x + MLP(x)."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class TimeConditionedMLP(nn.Module):
    """Point-wise MLP with residual blocks and learned time embedding.

    Takes (batch,2) coords + (batch,) time -> (batch,2) output.
    """

    def __init__(self, hidden_dim=256, time_emb_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.proj_in = nn.Sequential(
            nn.Linear(2 + time_emb_dim, hidden_dim),
            nn.SiLU(),
        )
        self.res_blocks = nn.Sequential(
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
        )
        self.proj_out = nn.Linear(hidden_dim, 2)

    def forward(self, x, t):
        """x: (batch,2), t: (batch,) -> (batch,2)"""
        t_emb = self.time_mlp(t[:, None])
        h = self.proj_in(torch.cat([x, t_emb], dim=-1))
        h = self.res_blocks(h)
        return self.proj_out(h)


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model):
        for key, val in model.state_dict().items():
            self.shadow[key].mul_(self.decay).add_(val, alpha=1 - self.decay)

    def apply(self, model):
        """Swap model params with EMA params. Call restore() after generation."""
        self.backup = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.shadow)

    def restore(self, model):
        """Restore original model params after generation."""
        model.load_state_dict(self.backup)
        del self.backup
