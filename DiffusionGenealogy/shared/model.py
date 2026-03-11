import torch
import torch.nn as nn
import math
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
    """Point-wise MLP with sinusoidal+learned time embedding and residual blocks.

    Takes (batch,2) coords + (batch,) time -> (batch,2) output.
    """

    def __init__(self, hidden_dim=512, time_emb_dim=128):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        # Learned projection on top of sinusoidal features
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
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
            ResBlock(hidden_dim),
        )
        self.proj_out = nn.Linear(hidden_dim, 2)

    def _sinusoidal_emb(self, t):
        """Sinusoidal embedding with 1000x scaling for proper frequency coverage."""
        half = self.time_emb_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None] * 1000.0 * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, x, t):
        """x: (batch,2), t: (batch,) -> (batch,2)"""
        t_emb = self._sinusoidal_emb(t)
        t_emb = self.time_mlp(t_emb)
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
