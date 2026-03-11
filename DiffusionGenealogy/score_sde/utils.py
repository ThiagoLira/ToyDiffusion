import torch
import numpy as np


def vp_sde_params(beta_min=0.1, beta_max=20.0):
    """Return VP-SDE coefficient functions.

    VP-SDE: dx = -0.5 * beta(t) * x dt + sqrt(beta(t)) dW
    where beta(t) = beta_min + t * (beta_max - beta_min)

    Marginal distribution: q(x_t | x_0) = N(alpha(t)*x_0, sigma(t)^2*I)
    where:
        alpha(t) = exp(-0.5 * integral_0^t beta(s) ds)
        sigma(t)^2 = 1 - alpha(t)^2
    """
    return beta_min, beta_max


def marginal_prob_params(t, beta_min=0.1, beta_max=20.0):
    """Compute mean coefficient alpha(t) and std sigma(t) for VP-SDE marginal.

    Args:
        t: (batch,) tensor of times in [0, 1]

    Returns:
        alpha: (batch,) mean scaling factor
        sigma: (batch,) standard deviation
    """
    # Integral of beta(s) from 0 to t: beta_min*t + 0.5*(beta_max-beta_min)*t^2
    log_mean_coeff = -0.25 * t**2 * (beta_max - beta_min) - 0.5 * t * beta_min
    alpha = torch.exp(log_mean_coeff)
    sigma = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
    return alpha, sigma


def sde_drift_diffusion(t, beta_min=0.1, beta_max=20.0):
    """Compute drift coefficient f(t) and diffusion coefficient g(t) for VP-SDE.

    dx = f(t)*x dt + g(t) dW
    f(t) = -0.5 * beta(t)
    g(t) = sqrt(beta(t))
    """
    beta_t = beta_min + t * (beta_max - beta_min)
    f = -0.5 * beta_t
    g = torch.sqrt(beta_t)
    return f, g
