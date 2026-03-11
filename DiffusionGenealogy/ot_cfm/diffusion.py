import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from ..shared.model import TimeConditionedMLP
from .utils import compute_ot_plan


class OTCFMDiffusion:
    """Optimal Transport Conditional Flow Matching.

    Same as Rectified Flow but with minibatch OT coupling:
    before interpolation, solve linear_sum_assignment to reorder noise.

    Train: t~U[0,1], OT-couple (x_0, x_1), x_t = (1-t)*x_0 + t*x_1, v = x_1 - x_0
    Sample: Euler ODE dx = v*dt from t=0 -> 1, 50 steps
    """

    def __init__(self, model=None, device="cpu", hidden_dim=256, time_emb_dim=64):
        self.device = device
        if model is None:
            self.model = TimeConditionedMLP(hidden_dim, time_emb_dim).to(device)
        else:
            self.model = model.to(device)

    def train(self, data, epochs=100, batch_size=256, lr=1e-3):
        """Train with OT-coupled pairs.

        Args:
            data: (N, 2) tensor of target data points
            epochs: number of training epochs
            batch_size: batch size (kept small since OT is O(n^3))
            lr: learning rate

        Returns:
            list of per-epoch average losses
        """
        data = data.to(self.device)
        N = data.shape[0]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        losses = []

        for epoch in tqdm(range(epochs), desc="OT-CFM"):
            epoch_loss = 0.0
            n_batches = 0

            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, batch_size):
                idx = perm[start : start + batch_size]
                x_0 = data[idx]  # target data
                bs = x_0.shape[0]

                # Sample noise
                x_1 = torch.randn(bs, 2, device=self.device)

                # Compute OT coupling on CPU (scipy)
                x_0_np = x_0.cpu().numpy()
                x_1_np = x_1.cpu().numpy()
                ot_perm = compute_ot_plan(x_0_np, x_1_np)
                x_1 = x_1[torch.from_numpy(ot_perm).to(self.device)]

                # Sample time
                t = torch.rand(bs, device=self.device)
                t_expand = t[:, None]

                # Linear interpolation with OT-coupled pairs
                x_t = (1 - t_expand) * x_0 + t_expand * x_1

                # Target velocity
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
    def generate(self, n_samples, n_steps=50):
        """Generate samples via Euler ODE integration from t=0 to t=1.

        Fewer steps needed thanks to straighter OT paths.

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
