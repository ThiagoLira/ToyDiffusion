import torch
import numpy as np


def compute_straightness(trajectories):
    """Compute straightness metric: ratio of direct distance to path length.

    Straightness = ||x_1 - x_0|| / sum(||x_{t+1} - x_t||)
    A value of 1.0 means perfectly straight paths.

    Args:
        trajectories: list of (N, 2) numpy arrays

    Returns:
        float: mean straightness across all points (0 to 1)
    """
    x_start = trajectories[0]
    x_end = trajectories[-1]

    direct_dist = np.linalg.norm(x_end - x_start, axis=1)

    path_length = np.zeros(len(x_start))
    for i in range(1, len(trajectories)):
        step_dist = np.linalg.norm(trajectories[i] - trajectories[i - 1], axis=1)
        path_length += step_dist

    # Avoid division by zero
    mask = path_length > 1e-8
    straightness = np.zeros_like(direct_dist)
    straightness[mask] = direct_dist[mask] / path_length[mask]

    return float(straightness.mean())
