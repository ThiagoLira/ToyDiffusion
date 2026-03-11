import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_ot_plan(x_0, x_1):
    """Compute optimal transport assignment between x_0 (data) and x_1 (noise).

    Solves the linear assignment problem to minimize total squared Euclidean distance.

    Args:
        x_0: (batch, 2) numpy array — data points
        x_1: (batch, 2) numpy array — noise points

    Returns:
        perm: index permutation to reorder x_1 so that x_1[perm] is optimally coupled to x_0
    """
    # Cost matrix: squared Euclidean distances
    diff = x_0[:, None, :] - x_1[None, :, :]
    cost = (diff ** 2).sum(axis=-1)

    _, col_ind = linear_sum_assignment(cost)
    return col_ind
