import torch
import numpy as np


def karras_sigma_schedule(n_steps, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    """Karras et al. noise schedule for sampling.

    sigma_i = (sigma_max^(1/rho) + i/(n-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho

    Args:
        n_steps: number of sampling steps
        sigma_min: minimum noise level
        sigma_max: maximum noise level
        rho: schedule curvature parameter

    Returns:
        sigmas: (n_steps+1,) tensor including sigma=0 at the end
    """
    ramp = torch.linspace(0, 1, n_steps)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    sigmas = torch.cat([sigmas, torch.zeros(1)])  # append sigma=0
    return sigmas


def edm_precond(sigma, sigma_data=0.5):
    """EDM preconditioning functions (Karras et al. Table 1).

    Args:
        sigma: noise level (scalar or tensor)
        sigma_data: data standard deviation

    Returns:
        c_skip: skip connection scaling
        c_out: output scaling
        c_in: input scaling
        c_noise: noise conditioning value
    """
    c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out = sigma * sigma_data / torch.sqrt(sigma**2 + sigma_data**2)
    c_in = 1.0 / torch.sqrt(sigma**2 + sigma_data**2)
    c_noise = 0.25 * torch.log(sigma)
    return c_skip, c_out, c_in, c_noise


def edm_loss_weight(sigma, sigma_data=0.5):
    """EDM loss weighting: lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2"""
    return (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2
