import numpy as np


def make_ddim_timesteps(T, n_steps):
    """Create a subsequence of timesteps for DDIM sampling.

    Evenly spaces n_steps timesteps across [0, T-1].

    Args:
        T: total training timesteps
        n_steps: number of DDIM sampling steps

    Returns:
        list of int timesteps in increasing order
    """
    step_size = T // n_steps
    timesteps = list(range(0, T, step_size))[:n_steps]
    return timesteps
