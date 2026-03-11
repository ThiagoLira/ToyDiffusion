import torch


@torch.no_grad()
def sinkhorn_assignment(cost, n_iters=100, reg=None):
    """GPU Sinkhorn algorithm to approximate OT assignment.

    Args:
        cost: (batch, batch) cost matrix (torch tensor, on GPU)
        n_iters: number of Sinkhorn iterations
        reg: entropic regularization (smaller = closer to exact OT).
             If None, auto-scale to 0.01 * median(cost).

    Returns:
        perm: (batch,) index permutation tensor (on same device as cost)
    """
    n = cost.shape[0]

    # Auto-scale regularization relative to cost magnitude
    if reg is None:
        reg = 0.01 * cost.median().clamp(min=1e-6)

    # Log-domain Sinkhorn for numerical stability
    log_K = -cost / reg

    log_u = torch.zeros(n, device=cost.device)
    log_v = torch.zeros(n, device=cost.device)

    for _ in range(n_iters):
        log_u = -torch.logsumexp(log_K + log_v[None, :], dim=1)
        log_v = -torch.logsumexp(log_K + log_u[:, None], dim=0)

    # Transport plan
    log_P = log_u[:, None] + log_K + log_v[None, :]
    P = log_P.exp()

    # Extract permutation: argmax + fully vectorized collision fix
    perm = P.argmax(dim=1)
    confidence = P[torch.arange(n, device=cost.device), perm]

    # For each column, keep only the row with highest confidence (winner)
    # scatter_ with ascending order so highest confidence overwrites
    winner_per_col = torch.full((n,), -1, dtype=torch.long, device=cost.device)
    order = confidence.argsort()  # ascending — highest last wins
    winner_per_col.scatter_(0, perm[order], order)

    # Rows that aren't winners for their chosen column have collisions
    rows = torch.arange(n, device=cost.device)
    is_winner = winner_per_col[perm] == rows
    conflict = ~is_winner

    # Reassign conflicting rows to unused columns
    if conflict.any():
        used_mask = torch.zeros(n, dtype=torch.bool, device=cost.device)
        used_mask[perm[is_winner]] = True
        unused_cols = rows[~used_mask]
        conflict_rows = rows[conflict]
        # Random assignment among unused columns
        rand_idx = torch.randperm(unused_cols.shape[0], device=cost.device)[
            : conflict_rows.shape[0]
        ]
        perm[conflict_rows] = unused_cols[rand_idx]

    return perm


def compute_ot_plan_gpu(x_0, x_1):
    """Compute OT assignment entirely on GPU using Sinkhorn.

    Args:
        x_0: (batch, 2) tensor — data points (on GPU)
        x_1: (batch, 2) tensor — noise points (on GPU)

    Returns:
        perm: (batch,) index permutation tensor (on GPU)
    """
    # Squared Euclidean cost matrix
    cost = torch.cdist(x_0, x_1, p=2).pow(2)
    return sinkhorn_assignment(cost)
