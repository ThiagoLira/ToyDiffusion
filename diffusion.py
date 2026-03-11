import torch
import numpy as np


def q_sample(x_start, t, list_bar_alphas, device):
    """
    Forward diffusion process: sample x_t given x_0 in closed form.
    (t == 0 means diffused for 1 step)

    This implements Eq. (4) from the DDPM paper:

        q(x_t | x_0) = N(x_t;  sqrt(alpha_bar_t) * x_0,  (1 - alpha_bar_t) * I)

    Instead of applying noise step-by-step through the Markov chain
    q(x_t | x_{t-1}) for each step from 0 to t, we can jump directly to any
    timestep t thanks to the reparameterization trick:

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

    where epsilon ~ N(0, I). This is equivalent to sampling from the
    multivariate normal below. alpha_bar_t = prod(alpha_s, s=1..t) is the
    cumulative product of alphas, so as t grows, alpha_bar_t shrinks toward 0,
    meaning the signal (x_0) fades and the noise dominates.
    """
    alpha_bar_t = list_bar_alphas[t]

    # Mean of the distribution: sqrt(alpha_bar_t) * x_0
    # This scales down the original data -- the signal gets weaker over time
    mean = alpha_bar_t * x_start

    # Covariance: (1 - alpha_bar_t) * I  (isotropic Gaussian noise)
    # As alpha_bar_t -> 0 (many steps), covariance -> I, i.e. pure noise
    cov = torch.eye(x_start.shape[0]).to(device)
    cov = cov * (1 - alpha_bar_t)

    # Sample x_t from N(sqrt(alpha_bar_t)*x_0, (1-alpha_bar_t)*I)
    return torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=cov).sample().to(device)


def denoise_with_mu(denoise_model, x_t, t, list_alpha, list_alpha_bar, DATA_SIZE, device):
    """
    Reverse diffusion step: sample x_{t-1} given x_t using the learned model.

    This implements sampling from pθ(x_{t-1} | x_t) as defined in Eq. (1):

        pθ(x_{t-1} | x_t) = N(x_{t-1};  mu_theta(x_t, t),  sigma_t^2 * I)

    In this implementation the model directly predicts the posterior mean
    mu_theta (the "mu-prediction" variant from Section 3.2 / Table 2 of the
    paper). The paper also explores an epsilon-prediction variant (Eq. 11)
    which performed better on images, but for this toy example mu-prediction
    is used.

    The variance sigma_t^2 is set to beta_t (= 1 - alpha_t), which is one of
    the two fixed variance choices discussed in Section 3.2. The paper notes
    that sigma_t^2 = beta_t is optimal when x_0 ~ N(0, I).
    """
    alpha_t = list_alpha[t]
    # beta_t = 1 - alpha_t is the noise variance added at step t in the forward process
    beta_t = 1 - alpha_t
    alpha_bar_t = list_alpha_bar[t]

    # The neural network predicts mu_theta: the mean of pθ(x_{t-1} | x_t)
    mu_theta = denoise_model(x_t, t)

    # Sample x_{t-1} ~ N(mu_theta, beta_t * I)
    # This adds a small amount of stochasticity during the reverse process,
    # controlled by beta_t. At the final step (t=0) the paper sets z=0
    # (no noise), but here we always sample for simplicity.
    x_t_before = torch.distributions.MultivariateNormal(
        loc=mu_theta,
        covariance_matrix=torch.diag(beta_t.repeat(DATA_SIZE))
    ).sample().to(device)

    return x_t_before


def posterior_q(x_start, x_t, t, list_alpha, list_alpha_bar, device):
    """
    Compute the parameters of the forward process posterior q(x_{t-1} | x_t, x_0).

    This implements Eq. (6) and (7) from the DDPM paper:

        q(x_{t-1} | x_t, x_0) = N(x_{t-1};  mu_tilde_t,  beta_tilde_t * I)

    This posterior is TRACTABLE because we condition on x_0 (the original data).
    It tells us: "given where we started (x_0) and where we are now (x_t),
    what was the most likely previous step x_{t-1}?"

    The training objective (Eq. 8) is to make the neural network's predicted
    mean mu_theta match this tractable posterior mean mu_tilde:

        L_{t-1} = (1 / 2*sigma_t^2) * || mu_tilde_t - mu_theta(x_t, t) ||^2

    This is why we compute mu_tilde: it serves as the training target.

    Eq. (7) - Posterior mean:
        mu_tilde_t = [sqrt(alpha_bar_{t-1}) * beta_t / (1 - alpha_bar_t)] * x_0
                   + [sqrt(alpha_t) * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)] * x_t

    Eq. (7) - Posterior variance:
        beta_tilde_t = [(1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)] * beta_t
    """
    beta_t = 1 - list_alpha[t]
    alpha_t = list_alpha[t]
    alpha_bar_t = list_alpha_bar[t]
    # alpha_bar_{t-1}: cumulative product up to step t-1
    alpha_bar_t_before = list_alpha_bar[t - 1]

    # ---- Compute mu_tilde (Eq. 7, first formula) ----
    # First term:  x_0 * sqrt(alpha_bar_{t-1}) * beta_t / (1 - alpha_bar_t)
    # This "pulls" the mean toward x_0, weighted by how much noise was added at step t
    first_term = x_start * torch.sqrt(alpha_bar_t_before) * beta_t / (1 - alpha_bar_t)

    # Second term: x_t * sqrt(alpha_t) * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
    # This "pulls" the mean toward x_t, weighted by cumulative noise up to t-1
    second_term = x_t * torch.sqrt(alpha_t) * (1 - alpha_bar_t_before) / (1 - alpha_bar_t)

    # The posterior mean is a weighted combination of x_0 and x_t
    mu_tilde = first_term + second_term

    # ---- Compute beta_tilde (Eq. 7, second formula) ----
    # beta_tilde_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
    # Note: this is NOT used as the covariance below (see note)
    beta_t_tilde = beta_t * (1 - alpha_bar_t_before) / (1 - alpha_bar_t)

    # NOTE: The covariance returned here uses (1 - alpha_bar_t) * I instead of
    # beta_tilde_t * I. This is a simplification for this toy implementation.
    cov = torch.eye(x_start.shape[0]).to(device) * (1 - alpha_bar_t)

    return mu_tilde, cov


def position_encoding_init(n_position, d_pos_vec):
    """
    Sinusoidal position encoding table, from "Attention Is All You Need"
    (Vaswani et al., 2017).

    In the DDPM paper (Section 4, Appendix B), the diffusion timestep t is
    embedded into the network using this same Transformer-style sinusoidal
    encoding. This gives the network a smooth, continuous representation of
    "how noisy is the input" so it can adapt its denoising behavior per step.

    The encoding for position `pos` and dimension `i` is:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    where d = d_pos_vec is the embedding dimension.

    Args:
        n_position: number of timesteps T (each timestep gets its own embedding)
        d_pos_vec:  embedding dimension (set equal to x_dim so it can be added
                    directly to the input data)
    """
    # Build the argument matrix: pos / 10000^(2i/d)
    # Position 0 is reserved as a zero vector (padding token convention)
    position_enc = np.array([
        [pos / np.power(10000, 2 * i / d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    # Apply sin to even indices (2i) and cos to odd indices (2i+1)
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).to(torch.float32)


class Denoising(torch.nn.Module):
    """
    The denoising neural network mu_theta(x_t, t).

    In the full DDPM paper, this would be a U-Net with self-attention
    (Appendix B). Here, since we're working with a 1D vector of (x,y)
    coordinates instead of images, a simple 3-layer MLP suffices.

    The network takes noisy data x_t and the current timestep t, and
    predicts the posterior mean mu_tilde_t -- i.e., it estimates what
    x_{t-1} should look like (the "mu-prediction" parameterization from
    Table 2 of the paper).

    The timestep t is incorporated via a sinusoidal positional embedding
    (same as the Transformer) that is ADDED to the input before the first
    layer. This is how the network knows "how much noise" is in the data,
    so it can calibrate its denoising strength accordingly.
    """

    def __init__(self, x_dim, num_diffusion_timesteps):
        super(Denoising, self).__init__()

        # Three linear layers: input_dim -> input_dim (keeps dimensionality)
        self.linear1 = torch.nn.Linear(x_dim, x_dim)
        # Precompute sinusoidal embeddings for all timesteps [0, T)
        # Shape: (num_diffusion_timesteps, x_dim)
        self.emb = position_encoding_init(num_diffusion_timesteps, x_dim)
        self.linear2 = torch.nn.Linear(x_dim, x_dim)
        self.linear3 = torch.nn.Linear(x_dim, x_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x_input, t):
        # Look up the sinusoidal embedding for timestep t
        emb_t = self.emb[t]

        # Add the timestep embedding directly to the input (element-wise).
        # This informs the network about the noise level at this step.
        # The full paper uses addition into residual blocks; same idea here.
        x = self.linear1(x_input + emb_t)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)

        # Output: predicted posterior mean mu_theta(x_t, t)
        # Has same dimension as x_input (the noisy data)
        return x