import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from ..shared.model import TimeConditionedMLP, EMA
from .utils import cosine_beta_schedule


class DDPMDiffusion:
    """DDPM: Epsilon-prediction with stochastic reverse sampling.

    Uses cosine beta schedule (Improved DDPM) for better fine detail.
    Train: t~U{0,T-1}, eps~N(0,I), x_t = sqrt(abar_t)*x_0 + sqrt(1-abar_t)*eps,
           loss = MSE(eps, model(x_t, t/T))
    Sample: Reverse t=T-1..0, predict eps, compute mu, add noise (except t=0).
    """

    def __init__(self, model=None, device="cpu", hidden_dim=512, time_emb_dim=128,
                 T=1000):
        self.device = device
        self.T = T
        if model is None:
            self.model = TimeConditionedMLP(hidden_dim, time_emb_dim).to(device)
        else:
            self.model = model.to(device)

        betas, alphas, alpha_bars = cosine_beta_schedule(T)
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alpha_bars = alpha_bars.to(device)

    def train(self, data, epochs=100, batch_size=512, lr=1e-3):
        data = data.to(self.device)
        N = data.shape[0]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
        criterion = nn.MSELoss()
        self.ema = EMA(self.model)
        losses = []

        for epoch in tqdm(range(epochs), desc="DDPM"):
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                self.ema.update(self.model)

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            losses.append(epoch_loss / n_batches)

        return losses

    @torch.no_grad()
    def generate(self, n_samples, n_steps=None):
        """Reverse diffusion sampling from t=T-1 down to 0."""
        self.model.eval()
        if hasattr(self, 'ema'):
            self.ema.apply(self.model)

        x = torch.randn(n_samples, 2, device=self.device)
        trajectory = [x.cpu().numpy()]

        for t in range(self.T - 1, -1, -1):
            t_batch = torch.full((n_samples,), t, device=self.device)
            t_norm = t_batch.float() / self.T

            eps_pred = self.model(x, t_norm)

            alpha_t = self.alphas[t]
            abar_t = self.alpha_bars[t]
            beta_t = self.betas[t]

            mu = (1.0 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1.0 - abar_t)) * eps_pred
            )

            if t > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta_t)
                x = mu + sigma * noise
            else:
                x = mu

            trajectory.append(x.cpu().numpy())

        if hasattr(self, 'ema'):
            self.ema.restore(self.model)
        self.model.train()
        return trajectory
