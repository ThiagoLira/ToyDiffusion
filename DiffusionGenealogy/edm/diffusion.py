import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from ..shared.model import TimeConditionedMLP
from .utils import karras_sigma_schedule, edm_precond, edm_loss_weight


class EDMDiffusion:
    """EDM: Preconditioned denoiser D_theta with Heun 2nd-order solver.

    The raw network F_theta takes (c_in*x, c_noise) and the preconditioned
    denoiser is: D_theta(x;sigma) = c_skip*x + c_out*F_theta(c_in*x, c_noise)

    Train: sigma~LogNormal(P_mean, P_std), x_noisy = x_0 + sigma*eps,
           loss = weight(sigma) * ||D_theta(x_noisy;sigma) - x_0||^2
    Sample: Heun 2nd-order on Karras sigma schedule, 40 steps.
    """

    def __init__(self, model=None, device="cpu", hidden_dim=256, time_emb_dim=64,
                 sigma_data=None, P_mean=-1.2, P_std=1.2):
        self.device = device
        self.sigma_data = sigma_data  # computed from data if None
        self.P_mean = P_mean
        self.P_std = P_std
        if model is None:
            self.model = TimeConditionedMLP(hidden_dim, time_emb_dim).to(device)
        else:
            self.model = model.to(device)

    def _denoiser(self, x, sigma):
        """Preconditioned denoiser D_theta(x; sigma)."""
        c_skip, c_out, c_in, c_noise = edm_precond(sigma, self.sigma_data)
        c_skip = c_skip[:, None]
        c_out = c_out[:, None]
        c_in = c_in[:, None]

        F_out = self.model(c_in * x, c_noise)
        return c_skip * x + c_out * F_out

    def train(self, data, epochs=100, batch_size=512, lr=1e-3):
        data = data.to(self.device)
        N = data.shape[0]

        if self.sigma_data is None:
            self.sigma_data = float(data.std())

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
        losses = []

        for epoch in tqdm(range(epochs), desc="EDM"):
            epoch_loss = 0.0
            n_batches = 0

            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, batch_size):
                idx = perm[start : start + batch_size]
                x_0 = data[idx]
                bs = x_0.shape[0]

                ln_sigma = torch.randn(bs, device=self.device) * self.P_std + self.P_mean
                sigma = torch.exp(ln_sigma)

                eps = torch.randn_like(x_0)
                x_noisy = x_0 + sigma[:, None] * eps

                D_pred = self._denoiser(x_noisy, sigma)

                weight = edm_loss_weight(sigma, self.sigma_data)
                loss = (weight[:, None] * (D_pred - x_0) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            losses.append(epoch_loss / n_batches)

        return losses

    @torch.no_grad()
    def generate(self, n_samples, n_steps=40):
        """Heun's 2nd-order method on Karras sigma schedule."""
        self.model.eval()
        sigmas = karras_sigma_schedule(n_steps).to(self.device)

        x = torch.randn(n_samples, 2, device=self.device) * sigmas[0]
        trajectory = [x.cpu().numpy()]

        for i in range(n_steps):
            sigma_cur = sigmas[i]
            sigma_next = sigmas[i + 1]

            if sigma_cur == 0:
                break

            sigma_batch = torch.full((n_samples,), sigma_cur, device=self.device)
            D = self._denoiser(x, sigma_batch)
            d = (x - D) / sigma_cur

            x_next = x + d * (sigma_next - sigma_cur)

            if sigma_next > 0:
                sigma_next_batch = torch.full((n_samples,), sigma_next, device=self.device)
                D_next = self._denoiser(x_next, sigma_next_batch)
                d_next = (x_next - D_next) / sigma_next
                x_next = x + 0.5 * (d + d_next) * (sigma_next - sigma_cur)

            x = x_next
            trajectory.append(x.cpu().numpy())

        self.model.train()
        return trajectory
