import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from ..shared.model import TimeConditionedMLP


class RectifiedFlowDiffusion:
    """Rectified Flow: linear interpolation + velocity prediction + Euler ODE.

    Train: t~U[0,1], x_t = (1-t)*x_0 + t*eps, v_target = eps - x_0
    Sample: Euler ODE dx = v*dt from t=0 -> 1, 100 steps
    """

    def __init__(self, model=None, device="cpu", hidden_dim=256, time_emb_dim=64):
        self.device = device
        if model is None:
            self.model = TimeConditionedMLP(hidden_dim, time_emb_dim).to(device)
        else:
            self.model = model.to(device)

    def train(self, data, epochs=100, batch_size=512, lr=1e-3):
        """Train the velocity field.

        Args:
            data: (N, 2) tensor of target data points
            epochs: number of training epochs
            batch_size: training batch size
            lr: learning rate

        Returns:
            list of per-epoch average losses
        """
        data = data.to(self.device)
        N = data.shape[0]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        losses = []

        for epoch in tqdm(range(epochs), desc="Rectified Flow"):
            epoch_loss = 0.0
            n_batches = 0

            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, batch_size):
                idx = perm[start : start + batch_size]
                x_0 = data[idx]  # target data
                bs = x_0.shape[0]

                # Sample noise and time
                x_1 = torch.randn_like(x_0)
                t = torch.rand(bs, device=self.device)

                # Linear interpolation: x_t = (1-t)*x_0 + t*x_1
                t_expand = t[:, None]
                x_t = (1 - t_expand) * x_0 + t_expand * x_1

                # Target velocity: v = x_1 - x_0
                v_target = x_1 - x_0

                optimizer.zero_grad()
                v_pred = self.model(x_t, t)
                loss = criterion(v_pred, v_target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            losses.append(epoch_loss / n_batches)

        return losses

    @torch.no_grad()
    def generate(self, n_samples, n_steps=100):
        """Generate samples via Euler ODE integration from t=0 to t=1.

        Args:
            n_samples: number of points to generate
            n_steps: number of Euler steps

        Returns:
            list of (n_samples, 2) numpy arrays (trajectory snapshots)
        """
        self.model.eval()
        dt = 1.0 / n_steps
        x = torch.randn(n_samples, 2, device=self.device)
        trajectory = [x.cpu().numpy()]

        for step in range(n_steps):
            t = torch.full((n_samples,), step * dt, device=self.device)
            v = self.model(x, t)
            x = x + v * dt
            trajectory.append(x.cpu().numpy())

        self.model.train()
        return trajectory
