# Upgrading ToyDiffusion to Flow Matching

## A practical guide for adapting your Homer Simpson scatterplot project

---

## 1. Background: What Changed Since DDPM?

Your original ToyDiffusion project was based on two papers:

- **Sohl-Dickstein et al. (2015)** — "Deep Unsupervised Learning using Nonequilibrium Thermodynamics"
- **Ho et al. (2020)** — "Denoising Diffusion Probabilistic Models" (DDPM)

The core idea: define a forward process that adds Gaussian noise over T steps, then train a neural network to reverse the process step by step. The model predicts the noise (epsilon) added at each step.

**Flow Matching** (2022) provides a simpler, more elegant alternative. Instead of a discrete Markov chain with a noise schedule, we define a *continuous* path from noise to data and train a network to predict the *velocity field* along that path.

---

## 2. The Two Seminal Papers You Should Read

### Paper 1: "Flow Matching for Generative Modeling"
- **Authors:** Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le (Meta FAIR)
- **Published:** October 2022 (ICLR 2023)
- **arXiv:** https://arxiv.org/abs/2210.02747

**Key contributions:**
- Introduces *Flow Matching* (FM): a simulation-free way to train Continuous Normalizing Flows (CNFs)
- Instead of learning to reverse a diffusion process, you directly regress a vector field that transports a simple distribution (noise) to your data distribution
- Shows that diffusion paths are a *special case* of the more general FM framework
- Proposes using Optimal Transport (OT) displacement interpolation, which gives straighter paths and faster sampling
- Achieves state-of-the-art on ImageNet in both likelihood and sample quality

### Paper 2: "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
- **Authors:** Xingchao Liu, Chengyue Gong, Qiang Liu (UT Austin)
- **Published:** September 2022 (ICLR 2023 Spotlight)
- **arXiv:** https://arxiv.org/abs/2209.03003
- **Code:** https://github.com/gnobitab/RectifiedFlow

**Key contributions:**
- Introduces *Rectified Flow*: learn an ODE that follows straight-line paths between noise and data
- Training is a simple least-squares regression problem
- The "reflow" procedure iteratively straightens paths, eventually enabling single-step generation
- Demonstrates that straight paths = no discretization error = fast inference

### Bonus: "Flow Matching Guide and Code"
- **Authors:** Lipman et al. (Meta FAIR, December 2024)
- **arXiv:** https://arxiv.org/abs/2412.06264
- Comprehensive tutorial with PyTorch code examples — ideal companion for implementation.

---

## 3. Conceptual Comparison: DDPM vs Flow Matching

### Your Original DDPM Approach

```
Forward process:  x_0 → x_1 → x_2 → ... → x_T  (add noise, T discrete steps)
Reverse process:  x_T → x_{T-1} → ... → x_0    (denoise, predict epsilon)

Training target: predict the noise epsilon added at step t
Loss: ||epsilon - epsilon_predicted||^2
Sampling: iterate T steps backwards (typically T=1000)
```

### Flow Matching Approach

```
Interpolation:    x_t = (1-t) * x_noise + t * x_data,  t in [0, 1]
Velocity field:   v(x_t, t) = x_data - x_noise  (the target velocity)

Training target: predict the velocity v at position x_t and time t
Loss: ||v_target - v_predicted(x_t, t)||^2
Sampling: solve ODE from t=0 to t=1 (can use few Euler steps!)
```

**The key simplification:** There is no noise schedule, no beta parameters, no
alpha_bar calculations. You just linearly interpolate between noise and data,
and train the network to predict the direction (velocity) from noise toward data.

---

## 4. How to Modify Your ToyDiffusion Code

### 4.1 The Forward Process (Replace `q_sample`)

**Old (DDPM):**
```python
def q_sample(x_0, t, list_bar_alphas):
    """Sample from q(x_t | x_0) — add noise according to schedule"""
    bar_alpha = list_bar_alphas[t]
    noise = np.random.randn(*x_0.shape)
    x_t = np.sqrt(bar_alpha) * x_0 + np.sqrt(1 - bar_alpha) * noise
    return x_t, noise
```

**New (Flow Matching):**
```python
def interpolate(x_1, t):
    """Create noisy sample at time t by linear interpolation.
    x_1 is the data, x_0 is noise. t in [0, 1] where t=0 is noise, t=1 is data.
    """
    x_0 = np.random.randn(*x_1.shape)  # pure noise
    x_t = (1 - t) * x_0 + t * x_1
    # The target velocity is simply the direction from noise to data
    v_target = x_1 - x_0
    return x_t, v_target, x_0
```

That's it. No noise schedule needed.

### 4.2 The Training Loop

**Old (DDPM):**
```python
for epoch in range(n_epochs):
    # Sample random timestep
    t = random.randint(0, T)
    # Get noisy data and the noise that was added
    x_t, noise = q_sample(x_0, t, list_bar_alphas)
    # Model predicts the noise
    noise_pred = model(x_t, t)
    loss = MSE(noise, noise_pred)
    loss.backward()
    optimizer.step()
```

**New (Flow Matching):**
```python
for epoch in range(n_epochs):
    # Sample random time in [0, 1]
    t = random.uniform(0, 1)
    # Get interpolated sample and target velocity
    x_t, v_target, x_0 = interpolate(x_data, t)
    # Model predicts the velocity
    v_pred = model(x_t, t)
    loss = MSE(v_target, v_pred)
    loss.backward()
    optimizer.step()
```

### 4.3 The Reverse Process (Sampling / Generation)

**Old (DDPM):**
```python
def reverse_diffusion(model, N_steps):
    """Start from noise, denoise step by step"""
    x = np.random.randn(n_points, 2)  # pure noise
    for t in reversed(range(N_steps)):
        noise_pred = model(x, t)
        # Complex formula involving alpha, beta, posterior mean...
        x = compute_posterior_mean(x, noise_pred, t) + sigma_t * z
    return x
```

**New (Flow Matching) — simple Euler ODE solver:**
```python
def generate(model, n_steps=100):
    """Start from noise, follow the velocity field to data"""
    x = np.random.randn(n_points, 2)  # pure noise at t=0
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = i * dt
        v = model(x, t)
        x = x + v * dt  # Euler step
    return x  # at t=1, should look like data
```

**Note how much simpler this is!** No posterior distribution calculation,
no variance formulas, no alpha/beta schedules. Just follow the velocity.

### 4.4 The Model Architecture

Your model architecture (the neural network) can stay essentially the same.
The inputs are the same: a point x_t and a time t.
The output is the same shape: a 2D vector (for your scatterplot case).
The only difference is the interpretation: instead of predicting noise epsilon,
it predicts velocity v.

A simple MLP works fine for 2D data:

```python
import torch
import torch.nn as nn

class VelocityMLP(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),   # 2D point + time
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),   # 2D velocity output
        )

    def forward(self, x, t):
        # x: (batch, 2), t: (batch, 1)
        inp = torch.cat([x, t], dim=-1)
        return self.net(inp)
```

---

## 5. Complete Minimal Example (PyTorch)

Here's a complete, self-contained script you can adapt for your Homer Simpson
scatterplot:

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# =====================
# 1. Load your 2D data
# =====================
# Replace this with your Homer Simpson scatterplot data
# data shape: (N, 2)
data = np.load("homer_simpson.npy")  # your data
data_tensor = torch.tensor(data, dtype=torch.float32)

# =====================
# 2. Define the model
# =====================
class VelocityField(nn.Module):
    def __init__(self, dim=2, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))

model = VelocityField()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# =====================
# 3. Training loop
# =====================
n_epochs = 5000
batch_size = 256

for epoch in range(n_epochs):
    # Sample a batch of data points
    idx = torch.randint(0, len(data_tensor), (batch_size,))
    x_1 = data_tensor[idx]  # data (target)

    # Sample noise
    x_0 = torch.randn_like(x_1)

    # Sample random time
    t = torch.rand(batch_size, 1)

    # Linear interpolation: x_t = (1-t)*x_0 + t*x_1
    x_t = (1 - t) * x_0 + t * x_1

    # Target velocity: direction from noise to data
    v_target = x_1 - x_0

    # Predict velocity
    v_pred = model(x_t, t)

    # Loss
    loss = ((v_pred - v_target) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# =====================
# 4. Generate samples
# =====================
@torch.no_grad()
def generate_samples(model, n_samples=1000, n_steps=100):
    x = torch.randn(n_samples, 2)  # start from noise
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((n_samples, 1), i * dt)
        v = model(x, t)
        x = x + v * dt
    return x.numpy()

samples = generate_samples(model)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(data[:, 0], data[:, 1], s=1, alpha=0.5)
axes[0].set_title("Original Data")
axes[1].scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5)
axes[1].set_title("Generated (Flow Matching)")
plt.savefig("flow_matching_result.png", dpi=150)
plt.show()
```

---

## 6. Generating an Animation (Like Your Original GIF)

To recreate the cool reverse-process GIF from your original project:

```python
@torch.no_grad()
def generate_with_trajectory(model, n_samples=1000, n_steps=100):
    """Generate samples and save intermediate states for animation"""
    x = torch.randn(n_samples, 2)
    trajectory = [x.numpy().copy()]
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((n_samples, 1), i * dt)
        v = model(x, t)
        x = x + v * dt
        trajectory.append(x.numpy().copy())
    return trajectory

trajectory = generate_with_trajectory(model)

# Create animation
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(6, 6))

def update(frame):
    ax.clear()
    ax.scatter(trajectory[frame][:, 0], trajectory[frame][:, 1], s=1, alpha=0.5)
    ax.set_title(f"t = {frame / len(trajectory):.2f}")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

anim = FuncAnimation(fig, update, frames=len(trajectory), interval=50)
anim.save("flow_matching_generation.gif", writer="pillow")
```

---

## 7. Key Differences to Keep in Mind

| Aspect | DDPM (Your Original) | Flow Matching |
|--------|---------------------|---------------|
| Forward process | Markov chain, T discrete steps | Linear interpolation, continuous t in [0,1] |
| Noise schedule | Required (betas, alphas) | Not needed |
| Training target | Predict noise (epsilon) | Predict velocity (v = x_data - x_noise) |
| Loss function | MSE on epsilon | MSE on velocity |
| Sampling | ~1000 steps typical | ~10-100 steps (straighter paths!) |
| Math complexity | KL divergence, posterior | Simple ODE integration |
| Posterior formula | Complex closed-form | Not needed |

---

## 8. Why Flow Matching is Better for a Toy Project

1. **Simpler code**: No noise schedule, no posterior calculations, no alpha/beta bookkeeping
2. **Faster sampling**: Straight paths mean fewer steps needed at generation time
3. **Same model architecture**: Your MLP can stay the same, just change what it predicts
4. **Cleaner math**: The training objective is literally "predict which direction to go"
5. **Modern**: This is what Stable Diffusion 3, Flux, and other state-of-the-art models use

---

## 9. Further Reading

- **Flow Matching Guide and Code** (Lipman et al., 2024): https://arxiv.org/abs/2412.06264
  - Has a complete PyTorch package with examples
- **MIT Course "Flow Matching and Diffusion Models"** (2026): https://diffusion.csail.mit.edu/
  - Full lecture notes available as PDF
- **Cambridge MLG Blog Post**: https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html
  - Excellent intuitive explanation
- **"Diffusion Meets Flow Matching"** blog: https://diffusionflow.github.io/
  - Shows the mathematical equivalence between the two frameworks

---

## 10. TL;DR — The 3 Changes to Your Project

1. **Replace `q_sample`** with linear interpolation: `x_t = (1-t)*noise + t*data`
2. **Change the training target** from noise prediction to velocity prediction: `v = data - noise`
3. **Replace the reverse process** with a simple Euler ODE solver: `x += v * dt`

That's it. Everything else (data loading, model architecture, visualization) stays the same.

Happy reading on your trip to Antarctica!
