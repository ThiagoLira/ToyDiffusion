# Flow Matching: Complete Reading Guide

## For upgrading ToyDiffusion from DDPM to Flow Matching

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Paper 1: Flow Matching (Lipman et al.)](#2-paper-1-flow-matching-for-generative-modeling)
3. [Paper 2: Rectified Flow (Liu et al.)](#3-paper-2-rectified-flow)
4. [DDPM vs Flow Matching: Side-by-Side](#4-ddpm-vs-flow-matching-side-by-side)
5. [The Math You Need](#5-the-math-you-need)
6. [The 3 Changes to Your Code](#6-the-3-changes-to-your-code)
7. [Advanced: Reflow for 1-Step Generation](#7-advanced-reflow-for-1-step-generation)
8. [Reading Order & Resources](#8-reading-order--resources)

---

## 1. The Big Picture

Your original ToyDiffusion implements **DDPM** (Ho et al., 2020):
- Define a **Markov chain** that adds Gaussian noise over T discrete steps
- Train a neural network to **reverse** the process step by step
- The model predicts the noise (epsilon) or the posterior mean (mu_tilde)

**Flow Matching** (2022) provides a simpler alternative:
- Define a **continuous path** from noise to data via linear interpolation
- Train a neural network to predict the **velocity field** along that path
- Generate samples by integrating a simple ODE

The key insight: **diffusion models are a special case of flow matching** with a
particular (suboptimal) choice of interpolation path. Using straight-line
(Optimal Transport) paths instead gives straighter trajectories, which means
fewer sampling steps and simpler math.

**Who uses flow matching now?** Stable Diffusion 3, Flux, and most modern
production generative models have switched from DDPM to flow matching.

---

## 2. Paper 1: Flow Matching for Generative Modeling

> **Authors:** Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le (Meta FAIR)
> **Published:** October 2022 (ICLR 2023)
> **arXiv:** https://arxiv.org/abs/2210.02747

### Core Idea

Learn a velocity field `v_theta(x, t)` that defines an ODE:

```
dx/dt = v_theta(x, t)
```

Starting from noise `x_0 ~ N(0, I)` at `t=0`, integrating this ODE to `t=1`
produces a sample from the data distribution.

### The Flow Matching Loss (Intractable)

The ideal objective would be:

```
L_FM = E_{t ~ U(0,1), x ~ p_t} || v_theta(x, t) - u_t(x) ||^2
```

where `u_t(x)` is the true velocity field that generates the probability path
`p_t`. **Problem:** we don't know `p_t` or `u_t`.

### Conditional Flow Matching (The Key Contribution)

Instead of working with intractable marginals, define **conditional** paths.
For each data point `x_1`, the conditional path is:

```
x_t = (1 - t) * x_0 + t * x_1       where x_0 ~ N(0, I)
```

The conditional velocity is simply:

```
u_t(x | x_1) = x_1 - x_0
```

The **Conditional Flow Matching** loss is:

```
L_CFM = E_{t, x_0, x_1} || v_theta(x_t, t) - (x_1 - x_0) ||^2
```

**Key theorem:** The gradients of L_CFM and L_FM are identical. Training with
the tractable conditional loss is equivalent to training with the intractable
marginal loss.

### Why Optimal Transport Paths?

The linear interpolation `x_t = (1-t)*x_0 + t*x_1` is the Wasserstein-2
optimal transport geodesic between the conditional distributions. This makes
the conditional trajectories **straight lines**, which has two consequences:

1. The marginal velocity field is smoother and closer to straight
2. Fewer ODE solver steps are needed at inference (even 10-20 Euler steps work)

Compare: DDPM-style paths are curved (they first destroy signal heavily, then
denoise), requiring ~1000 steps or specialized samplers like DDIM.

### Connection to Diffusion Models

Diffusion models are flow matching with a **specific non-linear path**:

```
x_t = sqrt(alpha_bar_t) * x_1 + sqrt(1 - alpha_bar_t) * epsilon
```

This choice corresponds to the VP-SDE schedule. The score function (what
diffusion models learn) and the velocity field (what flow matching learns)
are related by:

```
v(x, t) = f(x, t) - (1/2) * g(t)^2 * grad_x log p_t(x)
```

Predicting noise `epsilon`, predicting score `grad log p_t`, and predicting
velocity `v` are all equivalent parameterizations related by linear transforms.

---

## 3. Paper 2: Rectified Flow

> **Authors:** Xingchao Liu, Chengyue Gong, Qiang Liu (UT Austin)
> **Published:** September 2022 (ICLR 2023 Spotlight)
> **arXiv:** https://arxiv.org/abs/2209.03003
> **Code:** https://github.com/gnobitab/RectifiedFlow

### Core Formulation

Same ODE setup: learn `v_theta` such that `dZ_t/dt = v_theta(Z_t, t)`.

The training objective is identical to Conditional Flow Matching:

```
L = E_{t ~ U(0,1)} E_{X_0 ~ N(0,I), X_1 ~ p_data} || v_theta(X_t, t) - (X_1 - X_0) ||^2
```

where `X_t = (1 - t) * X_0 + t * X_1`.

### The Reflow Procedure (Unique Contribution)

The key idea: **iteratively straighten the flow trajectories**.

**Round 1 (Initial Training):**
- Sample independent pairs `(X_0, X_1)` where `X_0 ~ N(0,I)`, `X_1 ~ p_data`
- Train `v_1` with the loss above

**Round 2 (Reflow):**
- For many `X_0 ~ N(0,I)`, solve the ODE of `v_1` forward to get `Z_1`
- Now `(X_0, Z_1)` are **coupled** pairs (they lie on the same trajectory)
- Retrain `v_2` on these coupled pairs

**Why it works:** In round 1, independent pairs produce crossing linear
interpolation paths, forcing the velocity field to average at intersections
(creating curvature). After reflow, the pairs are correlated -- their
interpolations cross much less, so the retrained flow is straighter.

### Few-Step and One-Step Generation

If all trajectories are perfectly straight, a single Euler step suffices:

```
Z_1 = Z_0 + v(Z_0, 0)      # one-step generation!
```

In practice:
- After 1 reflow: good results with 2-4 Euler steps
- After 2 reflows: competitive 1-step generation

### Straightness Metric

```
S(v) = integral_0^1 E[ || v(Z_t, t) - (Z_1 - Z_0) ||^2 ] dt
```

`S(v) = 0` iff all trajectories are perfectly straight lines.

---

## 4. DDPM vs Flow Matching: Side-by-Side

### Notation Comparison

| Concept | DDPM (Your Code) | Flow Matching |
|---------|------------------|---------------|
| Time | `t in {0, 1, ..., T}` discrete | `t in [0, 1]` continuous |
| Data | `x_0` (original) | `x_1` (target at t=1) |
| Noise | `x_T` (fully noised) | `x_0` (noise at t=0) |
| Direction | noise -> data is t: T -> 0 | noise -> data is t: 0 -> 1 |

**Watch out:** The time direction is flipped! In DDPM, `t=0` is clean data.
In flow matching, `t=0` is noise and `t=1` is data.

### Process Comparison

| Aspect | DDPM (Your Original) | Flow Matching |
|--------|---------------------|---------------|
| Forward process | Markov chain, T discrete steps | Linear interpolation, continuous |
| Noise schedule | Required (betas, alphas, alpha_bars) | **Not needed** |
| Training target | Predict noise epsilon or mu_tilde | Predict velocity `v = x_data - x_noise` |
| Loss function | `\|\|mu_tilde - mu_theta\|\|^2` | `\|\|v_target - v_theta\|\|^2` |
| Sampling | ~1000 steps, complex posterior formula | ~10-100 Euler steps: `x += v * dt` |
| Math complexity | KL divergence, posterior distribution | Simple ODE integration |

### Equation Comparison

**Interpolation (how to create a noisy sample):**

```
DDPM:           x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
Flow Matching:  x_t = (1 - t) * x_noise + t * x_data
```

**Training target (what the network learns):**

```
DDPM (mu):      target = mu_tilde = f(x_0, x_t, alpha_t, beta_t, ...)  [complex formula]
DDPM (eps):     target = epsilon                                        [the added noise]
Flow Matching:  target = x_data - x_noise                               [the direction]
```

**Sampling (how to generate):**

```
DDPM:           x_{t-1} = mu_theta(x_t, t) + sigma_t * z              [reverse Markov]
Flow Matching:  x_{t+dt} = x_t + v_theta(x_t, t) * dt                 [Euler ODE step]
```

---

## 5. The Math You Need

### 5.1 The Continuity Equation

The probability path `p_t` and velocity field `u_t` satisfy:

```
dp_t/dt + div(p_t * u_t) = 0
```

This is the law of conservation of probability. If particles at positions
sampled from `p_t` all move according to velocity `u_t`, the resulting
density at the next instant is described by this PDE.

### 5.2 Gaussian Conditional Probability Paths

The general Gaussian conditional path is:

```
p_t(x | x_1) = N(x; mu_t(x_1), sigma_t(x_1)^2 * I)
```

With boundary conditions:
- At t=0: `mu_0 = 0, sigma_0 = 1`  (standard Gaussian prior)
- At t=1: `mu_1 = x_1, sigma_1 ~ 0`  (concentrated at data)

**OT/Linear path** (recommended):  `mu_t = t * x_1`,  `sigma_t = 1 - t`
**VP/Diffusion path** (DDPM-like):  `mu_t = sqrt(alpha_bar_t) * x_1`,  `sigma_t = sqrt(1 - alpha_bar_t)`

### 5.3 The Conditional Velocity Field

For the Gaussian path, the conditional velocity is:

```
u_t(x | x_1) = (sigma_t' / sigma_t) * (x - mu_t) + mu_t'
```

For the OT/linear path (`mu_t = t*x_1`, `sigma_t = 1-t`), this simplifies to:

```
u_t(x | x_1) = x_1 - x_0
```

A constant! The velocity doesn't depend on x or t -- it's just the straight
line from noise to data.

### 5.4 Score-Velocity-Epsilon Conversion

All three parameterizations are equivalent:

```
velocity:  v = alpha_t' * x_1 + sigma_t' * epsilon
score:     grad log p_t(x) = -(x - alpha_t * x_1) / sigma_t^2
epsilon:   eps = (x_t - alpha_t * x_1) / sigma_t
data:      x_1 = (x_t - sigma_t * epsilon) / alpha_t
```

You can convert between any two given the schedule (alpha_t, sigma_t).

---

## 6. The 3 Changes to Your Code

### Change 1: Replace `q_sample` with linear interpolation

```python
# OLD (DDPM) -- diffusion.py:q_sample
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

# NEW (Flow Matching)
x_t = (1 - t) * x_noise + t * x_data
v_target = x_data - x_noise
```

### Change 2: Change the training target to velocity

```python
# OLD (DDPM) -- trains on mu_tilde or epsilon
mu_theta = model(x_t, t)
loss = MSE(mu_tilde, mu_theta)

# NEW (Flow Matching) -- trains on velocity
v_pred = model(x_t, t)
loss = MSE(v_target, v_pred)
```

### Change 3: Replace reverse process with Euler ODE solver

```python
# OLD (DDPM) -- complex posterior sampling
x_{t-1} ~ N(mu_theta(x_t, t), beta_t * I)

# NEW (Flow Matching) -- simple Euler step
x = x + v_theta(x, t) * dt
```

### Complete Training Pseudocode

```python
for each batch:
    x_1 = sample_data()               # from dataset
    x_0 = torch.randn_like(x_1)       # from N(0, I)
    t = torch.rand(batch_size, 1)      # from U(0, 1)
    x_t = (1 - t) * x_0 + t * x_1     # linear interpolation
    v_target = x_1 - x_0              # target velocity
    v_pred = model(x_t, t)            # network prediction
    loss = MSE(v_pred, v_target)
    loss.backward()
    optimizer.step()
```

### Complete Sampling Pseudocode

```python
x = torch.randn(n_samples, dim)       # start from noise
dt = 1.0 / n_steps
for i in range(n_steps):
    t = i * dt
    v = model(x, t)
    x = x + v * dt                    # Euler step
# x is now a generated sample
```

---

## 7. Advanced: Reflow for 1-Step Generation

After training a basic flow matching model (v_1), you can optionally
straighten it further:

```python
# Step 1: Generate coupled pairs using v_1
x_0 = torch.randn(N, dim)                    # noise
z_1 = ode_solve(v_1, x_0, t=0 -> t=1)       # run ODE forward

# Step 2: Retrain v_2 on coupled pairs
for each batch:
    # Use the COUPLED pairs (x_0, z_1) instead of independent ones
    t = torch.rand(batch_size, 1)
    x_t = (1 - t) * x_0 + t * z_1
    v_target = z_1 - x_0
    v_pred = model_v2(x_t, t)
    loss = MSE(v_pred, v_target)
    ...

# After reflow, can generate with very few steps (even 1!):
sample = x_0 + model_v2(x_0, 0)              # one-step generation
```

---

## 8. Reading Order & Resources

### Recommended Reading Order

1. **Start here** -- The guide in this repo:
   `THEORY/extracted/GUIDE_Flow_Matching_for_ToyDiffusion.md`
   Practical, code-focused, maps directly to your ToyDiffusion project.

2. **Intuition** -- Cambridge MLG Blog:
   https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html
   Best intuitive explanation. Builds up from CNFs -> FM -> CFM -> OT paths.

3. **Interactive visuals** -- Diffusion Meets Flow Matching:
   https://diffusionflow.github.io/
   Shows the mathematical equivalence visually. Straight vs curved paths.

4. **Paper 1** -- Flow Matching (Lipman et al., 2022):
   https://arxiv.org/abs/2210.02747
   The theoretical foundation. Read Sections 1-3 for the math.

5. **Paper 2** -- Rectified Flow (Liu et al., 2022):
   https://arxiv.org/abs/2209.03003
   The reflow procedure for straightening. Read Sections 1-3.

6. **Implementation companion** -- Flow Matching Guide and Code (Lipman et al., 2024):
   https://arxiv.org/abs/2412.06264
   Comprehensive tutorial with PyTorch code. The best reference for coding.

7. **Code comparison** -- Medium post by Harsh Maheshwari:
   https://harshm121.medium.com/flow-matching-vs-diffusion-79578a16c510
   Side-by-side code for diffusion vs flow matching.

### Code Repos

| Repo | What | Link |
|------|------|------|
| RectifiedFlow | Official PyTorch implementation | https://github.com/gnobitab/RectifiedFlow |
| ToyDiffusion | Your original DDPM project | https://github.com/ThiagoLira/ToyDiffusion |

### MIT Course

- **Flow Matching and Diffusion Models** (2026 lecture notes):
  https://diffusion.csail.mit.edu/docs/lecture-notes.pdf
  Full academic treatment. Download the PDF for offline reading.

---

## Quick Reference Card

```
===================== FLOW MATCHING CHEAT SHEET =====================

TRAINING:
  x_1 ~ p_data                        # sample data
  x_0 ~ N(0, I)                       # sample noise
  t   ~ U(0, 1)                       # sample time
  x_t = (1-t)*x_0 + t*x_1             # interpolate
  v   = x_1 - x_0                     # target velocity
  loss = || v_theta(x_t, t) - v ||^2  # MSE loss

SAMPLING (Euler, N steps):
  x = sample from N(0, I)
  dt = 1/N
  for i in 0..N-1:
      t = i * dt
      x = x + v_theta(x, t) * dt
  return x

WHAT THE NETWORK PREDICTS:
  Input:  (x_t, t)   -- noisy point + time
  Output: v           -- velocity vector (same dim as x_t)

THAT'S IT. No betas. No alphas. No posterior. Just velocity.
=================================================================
```
