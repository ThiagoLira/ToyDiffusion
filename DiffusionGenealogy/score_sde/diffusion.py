import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from ..shared.model import TimeConditionedMLP, EMA
from .utils import marginal_prob_params, sde_drift_diffusion


class ScoreSDEDiffusion:
    """VP-SDE Score Matching with PF-ODE sampling.

    Train: t~U[eps,1], score matching with marginal perturbation kernel.
           Model predicts score = -eps/sigma (equivalent to noise prediction scaled by -1/sigma).
    Sample: PF-ODE: dx = [f(t)*x - 0.5*g(t)^2 * score] dt, Euler from t=1 -> eps.
    """

    def __init__(self, model=None, device="cpu", hidden_dim=512, time_emb_dim=128,
                 beta_min=0.1, beta_max=20.0):
        self.device = device
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.eps = 1e-3
        if model is None:
            self.model = TimeConditionedMLP(hidden_dim, time_emb_dim).to(device)
        else:
            self.model = model.to(device)

    def train(self, data, epochs=100, batch_size=512, lr=1e-3):
        data = data.to(self.device)
        N = data.shape[0]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
        self.ema = EMA(self.model)
        losses = []

        for epoch in tqdm(range(epochs), desc="Score-SDE"):
            epoch_loss = 0.0
            n_batches = 0

            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, batch_size):
                idx = perm[start : start + batch_size]
                x_0 = data[idx]
                bs = x_0.shape[0]

                t = torch.rand(bs, device=self.device) * (1.0 - self.eps) + self.eps
                eps = torch.randn_like(x_0)

                alpha, sigma = marginal_prob_params(t, self.beta_min, self.beta_max)
                x_t = alpha[:, None] * x_0 + sigma[:, None] * eps

                target_score = -eps / sigma[:, None]

                optimizer.zero_grad()
                score_pred = self.model(x_t, t)

                weight = sigma[:, None] ** 2
                loss = (weight * (score_pred - target_score) ** 2).mean()
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
    def generate(self, n_samples, n_steps=1000):
        """PF-ODE sampling: dx = [f(t)*x - 0.5*g(t)^2 * score] dt from t=1 to eps."""
        self.model.eval()
        if hasattr(self, 'ema'):
            self.ema.apply(self.model)

        dt = -(1.0 - self.eps) / n_steps
        x = torch.randn(n_samples, 2, device=self.device)
        trajectory = [x.cpu().numpy()]

        t_current = 1.0
        for step in range(n_steps):
            t_batch = torch.full((n_samples,), t_current, device=self.device)

            f, g = sde_drift_diffusion(t_batch, self.beta_min, self.beta_max)
            score = self.model(x, t_batch)

            drift = f[:, None] * x - 0.5 * (g[:, None] ** 2) * score
            x = x + drift * dt

            t_current += dt
            trajectory.append(x.cpu().numpy())

        if hasattr(self, 'ema'):
            self.ema.restore(self.model)
        self.model.train()
        return trajectory
