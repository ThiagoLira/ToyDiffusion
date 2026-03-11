import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from ..shared.model import TimeConditionedMLP
from ..ddpm.utils import linear_beta_schedule
from .utils import make_ddim_timesteps


class DDIMDiffusion:
    """DDIM: Same training as DDPM, deterministic sampling with eta=0.

    Train: Identical to DDPM (epsilon-prediction).
    Sample: DDIM update rule (Eq.12 from Song et al. 2020), 50 steps.
    """

    def __init__(self, model=None, device="cpu", hidden_dim=256, time_emb_dim=64,
                 T=300, beta_start=1e-4, beta_end=0.02, eta=0.0):
        self.device = device
        self.T = T
        self.eta = eta
        if model is None:
            self.model = TimeConditionedMLP(hidden_dim, time_emb_dim).to(device)
        else:
            self.model = model.to(device)

        betas, alphas, alpha_bars = linear_beta_schedule(T, beta_start, beta_end)
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alpha_bars = alpha_bars.to(device)

    def train(self, data, epochs=100, batch_size=512, lr=1e-3):
        """Train identical to DDPM."""
        data = data.to(self.device)
        N = data.shape[0]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
        criterion = nn.MSELoss()
        losses = []

        for epoch in tqdm(range(epochs), desc="DDIM"):
            epoch_loss = 0.0
            n_batches = 0

            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, batch_size):
                idx = perm[start : start + batch_size]
                x_0 = data[idx]
                bs = x_0.shape[0]

                t_int = torch.randint(0, self.T, (bs,), device=self.device)
                eps = torch.randn_like(x_0)

                abar = self.alpha_bars[t_int][:, None]
                x_t = torch.sqrt(abar) * x_0 + torch.sqrt(1.0 - abar) * eps

                t_norm = t_int.float() / self.T

                optimizer.zero_grad()
                eps_pred = self.model(x_t, t_norm)
                loss = criterion(eps_pred, eps)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            losses.append(epoch_loss / n_batches)

        return losses

    @torch.no_grad()
    def generate(self, n_samples, n_steps=50):
        """DDIM deterministic sampling (eta=0)."""
        self.model.eval()
        timesteps = make_ddim_timesteps(self.T, n_steps)
        x = torch.randn(n_samples, 2, device=self.device)
        trajectory = [x.cpu().numpy()]

        for i in range(len(timesteps) - 1, -1, -1):
            t = timesteps[i]
            t_prev = timesteps[i - 1] if i > 0 else 0

            t_batch = torch.full((n_samples,), t, device=self.device)
            t_norm = t_batch.float() / self.T

            eps_pred = self.model(x, t_norm)

            abar_t = self.alpha_bars[t]
            abar_prev = self.alpha_bars[t_prev] if i > 0 else torch.tensor(1.0, device=self.device)

            # Predict x_0 from eps
            x0_pred = (x - torch.sqrt(1.0 - abar_t) * eps_pred) / torch.sqrt(abar_t)

            # DDIM deterministic update (eta=0)
            sigma = self.eta * torch.sqrt(
                (1.0 - abar_prev) / (1.0 - abar_t) * (1.0 - abar_t / abar_prev)
            )

            dir_xt = torch.sqrt(1.0 - abar_prev - sigma**2) * eps_pred
            x = torch.sqrt(abar_prev) * x0_pred + dir_xt

            if self.eta > 0 and i > 0:
                x = x + sigma * torch.randn_like(x)

            trajectory.append(x.cpu().numpy())

        self.model.train()
        return trajectory
