

import math
import time
import torch


def sample_z(size, device):
    return math.sqrt(0.1) * torch.randn(size, 2, device=device)


def generate_data(n=2000, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.randn(n, 2, device=device)
    logits = X[:, 0] + X[:, 1] + X[:, 0].square() - X[:, 1].square()
    Y = 2 * torch.bernoulli(torch.sigmoid(logits)) - 1
    return X, Y


def risk(X, Y, Z, theta, smooth=False, tau=0.3):
    x1 = X[:, 0, None] + Z[None, :, 0]
    x2 = X[:, 1, None] + Z[None, :, 1]
    margin = Y[:, None] * (
        x1 + theta[0] * x2 + theta[1] * x1.square() + theta[2] * x2.square()
    )
    loss = torch.sigmoid(-margin / tau) if smooth else (margin < 0).float()
    return loss.mean()


def smooth_gradient(X, Y, Z, theta, tau=0.3):
    x1 = X[:, 0, None] + Z[None, :, 0]
    x2 = X[:, 1, None] + Z[None, :, 1]
    f = x1 + theta[0] * x2 + theta[1] * x1.square() + theta[2] * x2.square()
    s = torch.sigmoid(-Y[:, None] * f / tau)
    c = -Y[:, None] * s * (1 - s) / (tau * f.numel())
    return torch.stack(((c * x2).sum(), (c * x1.square()).sum(),
                        (c * x2.square()).sum()))


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def _mh(start, log_target, variance, steps, burn_in, refresh=None):
    """MH samplnhg """
    if not 0 <= burn_in < steps:
        raise ValueError("Require 0 <= burn_in < steps")
    theta = start.clone()
    batch = refresh() if refresh is not None else None
    current = log_target(theta, batch)
    chain = torch.empty(steps, 3, device=theta.device)
    for k in range(steps):
        proposal = theta + math.sqrt(variance) * torch.randn_like(theta)
        proposed_batch = batch
        if refresh is not None and torch.rand((), device=theta.device) < 0.5:
            proposed_batch = refresh()
        proposed = log_target(proposal, proposed_batch)
        if torch.log(torch.rand((), device=theta.device)) < proposed - current:
            theta, batch, current = proposal, proposed_batch, proposed
        chain[k] = theta
    return chain[burn_in:].cpu().numpy()


@torch.no_grad()
def sample_full(X, Y, Z_full, steps=10000, burn_in=1000, smooth=False):
    """Full MH, or Smooth MH when smooth=True, starting at (5,5,5)."""
    synchronize(X.device)
    started = time.perf_counter()
    n = len(X)

    def log_target(theta, batch):
        return -n * risk(X, Y, Z_full, theta, smooth) - 0.5 * (theta @ theta)

    start = X.new_full((3,), 5.0)
    samples = _mh(start, log_target, 20 / n, steps, burn_in)
    synchronize(X.device)
    return samples, time.perf_counter() - started


@torch.no_grad()
def sample_smh(X, Y, Z_full, m, steps=10000, burn_in=1000,
               pilot_steps=2000, pilot_burn_in=1000):
    """Return SMH samples, reference point, and pilot + SMH total time."""
    if m <= 0:
        raise ValueError("m must be positive")
    n = len(X)
    synchronize(X.device)
    started = time.perf_counter()

    # Stage 1: naive subsampling MH and the post-burn-in reference mean.
    alpha = 50 * m ** (2 / 5)

    def pilot_target(theta, Z):
        return -alpha * risk(X, Y, Z, theta) - 0.5 * (theta @ theta)

    pilot = _mh(X.new_full((3,), 5.0), pilot_target, 20 / alpha,
                pilot_steps, pilot_burn_in, lambda: sample_z(m, X.device))
    reference = pilot.mean(axis=0)
    center = torch.as_tensor(reference, device=X.device, dtype=X.dtype)
    synchronize(X.device)
    pilot_time = time.perf_counter() - started

    # Stage 2 Sub1 chain.
    started = time.perf_counter()
    full_gradient = smooth_gradient(X, Y, Z_full, center)

    def refresh():
        Z = sample_z(m, X.device)
        # These reference quantities stay fixed while this batch is retained.
        return (Z, risk(X, Y, Z, center),
                smooth_gradient(X, Y, Z, center) - full_gradient)

    def log_target(theta, batch):
        Z, reference_risk, gradient_difference = batch
        loss = (risk(X, Y, Z, theta) - reference_risk
                - gradient_difference @ (theta - center))
        return -n * loss - 0.5 * (theta @ theta)

    samples = _mh(center, log_target, 20 / n, steps, burn_in, refresh)
    synchronize(X.device)
    smh_time = time.perf_counter() - started
    return samples, reference, pilot_time + smh_time
