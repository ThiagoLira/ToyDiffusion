import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from ..shared.model import TimeConditionedMLP, EMA


class RectifiedFlowDiffusion:
    """Rectified Flow: linear interpolation + velocity prediction + Euler ODE.

    Convention (Liu et al.): t=0 is noise (source), t=1 is data (target).
    Train: t~U[0,1], x_t = (1-t)*z + t*x_1, v_target = x_1 - z
    Sample: Midpoint ODE dx = v*dt from t=0 -> 1, 200 steps
    """

    def __init__(self, model=None, device="cpu", hidden_dim=256, time_emb_dim=64):
        self.device = device
        if model is None:
            self.model = TimeConditionedMLP(hidden_dim, time_emb_dim).to(device)
        else:
            self.model = model.to(device)

    def train(self, data, epochs=100, batch_size=512, lr=1e-3):
        data = data.to(self.device)
        N = data.shape[0]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
        criterion = nn.MSELoss()
        self.ema = EMA(self.model)
        losses = []

        for epoch in tqdm(range(epochs), desc="Rectified Flow"):
            epoch_loss = 0.0
            n_batches = 0

            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, batch_size):
                idx = perm[start : start + batch_size]
                x_1 = data[idx]
                bs = x_1.shape[0]

                z = torch.randn_like(x_1)
                t = torch.rand(bs, device=self.device)
                t_expand = t[:, None]
                x_t = (1 - t_expand) * z + t_expand * x_1
                v_target = x_1 - z

                optimizer.zero_grad()
                v_pred = self.model(x_t, t)
                loss = criterion(v_pred, v_target)
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
    def generate(self, n_samples, n_steps=200):
        """Midpoint method ODE integration from t=0 to t=1."""
        self.model.eval()
        if hasattr(self, 'ema'):
            self.ema.apply(self.model)

        dt = 1.0 / n_steps
        x = torch.randn(n_samples, 2, device=self.device)
        trajectory = [x.cpu().numpy()]

        for step in range(n_steps):
            t = step * dt
            t_batch = torch.full((n_samples,), t, device=self.device)
            v = self.model(x, t_batch)

            # Midpoint method (2nd-order)
            t_mid = torch.full((n_samples,), t + 0.5 * dt, device=self.device)
            x_mid = x + 0.5 * dt * v
            v_mid = self.model(x_mid, t_mid)
            x = x + v_mid * dt

            trajectory.append(x.cpu().numpy())

        if hasattr(self, 'ema'):
            self.ema.restore(self.model)
        self.model.train()
        return trajectory
