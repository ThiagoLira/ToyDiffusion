import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from ..shared.model import TimeConditionedMLP, EMA
from .utils import compute_ot_plan_gpu


class OTCFMDiffusion:
    """Optimal Transport Conditional Flow Matching.

    Same as Rectified Flow but with minibatch OT coupling via GPU Sinkhorn:
    before interpolation, compute approximate OT plan to reorder noise.

    Convention: t=0 is noise (source), t=1 is data (target).
    Train: t~U[0,1], OT-couple (z, x_1), x_t = (1-t)*z + t*x_1, v = x_1 - z
    Sample: Midpoint ODE dx = v*dt from t=0 -> 1, 200 steps
    """

    def __init__(self, model=None, device="cpu", hidden_dim=512, time_emb_dim=128):
        self.device = device
        if model is None:
            self.model = TimeConditionedMLP(hidden_dim, time_emb_dim).to(device)
        else:
            self.model = model.to(device)

    def train(self, data, epochs=100, batch_size=256, lr=1e-3,
              patience=200, min_delta=1e-4):
        """Train with early stopping when loss plateaus.

        Args:
            patience: stop after this many epochs with no improvement > min_delta
            min_delta: minimum loss decrease to count as improvement
        """
        data = data.to(self.device)
        N = data.shape[0]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
        criterion = nn.MSELoss()
        self.ema = EMA(self.model)
        losses = []
        best_loss = float('inf')
        wait = 0

        pbar = tqdm(range(epochs), desc="OT-CFM")
        for epoch in pbar:
            epoch_loss = 0.0
            n_batches = 0

            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, batch_size):
                idx = perm[start : start + batch_size]
                x_1 = data[idx]
                bs = x_1.shape[0]

                z = torch.randn(bs, 2, device=self.device)

                # GPU Sinkhorn OT coupling — no CPU roundtrip
                ot_perm = compute_ot_plan_gpu(x_1, z)
                z = z[ot_perm]

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
            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)

            # Early stopping check (use smoothed loss over last 50 epochs)
            if len(losses) >= 50:
                recent_avg = sum(losses[-50:]) / 50
                if recent_avg < best_loss - min_delta:
                    best_loss = recent_avg
                    wait = 0
                else:
                    wait += 1
                if wait >= patience:
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", status="converged")
                    print(f"\n  Early stop at epoch {epoch+1} (no improvement for {patience} epochs)")
                    break

            if epoch % 100 == 0:
                pbar.set_postfix(loss=f"{avg_loss:.4f}")

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

            t_mid = torch.full((n_samples,), t + 0.5 * dt, device=self.device)
            x_mid = x + 0.5 * dt * v
            v_mid = self.model(x_mid, t_mid)
            x = x + v_mid * dt

            trajectory.append(x.cpu().numpy())

        if hasattr(self, 'ema'):
            self.ema.restore(self.model)
        self.model.train()
        return trajectory
