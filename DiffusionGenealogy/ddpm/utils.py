import torch
import numpy as np


def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    """Linear variance schedule from beta_start to beta_end over T timesteps.

    Uses sqrt-linear interpolation (same as original notebook) for smoother schedule.

    Returns:
        betas: (T,) tensor
        alphas: (T,) tensor where alpha_t = 1 - beta_t
        alpha_bars: (T,) tensor of cumulative products of alphas
    """
    betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, T) ** 2
    betas = torch.tensor(betas, dtype=torch.float32)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars
