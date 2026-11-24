import time
import ssl
from pathlib import Path
from urllib.request import urlopen
import numpy as np
import torch
from scipy.integrate import trapezoid
from scipy.stats import gaussian_kde


PARAMETER_NAMES = ("pi", "A1", "B1", "g1", "k1", "A2", "B2", "g2", "k2")
LOWER_BOUNDS = (0.0, -2.0, 0.0, -1.0, -0.04, -2.0, 0.0, -1.0, -0.04)
UPPER_BOUNDS = (0.5, 2.0, 3.0, 1.0, 2.0, 2.0, 3.0, 1.0, 2.0)
INITIAL_THETA = (0.25, 0.25, 0.60, 0.0, 0.25, -0.25, 0.60, 0.0, 0.25)
DEFAULT_H_VALUES = (0.020, 0.030, 0.020, 0.020, 0.015, 0.030, 0.020, 0.020, 0.015)
_W2_COUPLING_CACHE = {}

DATA_URL = "https://gammaray.nsstc.nasa.gov/batse/grb/catalog/4b/tables/4br_grossc.duration"


def load_data(data_file=None):
    data_file = Path(data_file) if data_file is not None else Path(__file__).with_name("4br_grossc.duration")
    if not data_file.exists():
        data_file.parent.mkdir(parents=True, exist_ok=True)
        context = ssl.create_default_context()
        if Path("/etc/ssl/cert.pem").exists():
            context.load_verify_locations("/etc/ssl/cert.pem")
        with urlopen(DATA_URL, context=context, timeout=30) as response:
            data_file.write_bytes(response.read())
    # The fifth column contains T90 durations in seconds.
    durations = np.loadtxt(data_file, usecols=4, dtype=np.float64)
    values = np.log10(durations)
    location, scale = float(values.mean()), float(values.std(ddof=1))
    y_real = torch.tensor(((values - location) / scale)[:, None], dtype=torch.float32)
    return y_real, values.astype(np.float32), location, scale


def sample_latents(size):
    return (torch.rand(size, dtype=torch.float32),
            torch.randn(size, dtype=torch.float32),
            torch.randn(size, dtype=torch.float32))


def g_and_k(z, A, B, g, k):
    skew = 1.0 + 0.8 * torch.tanh(0.5 * g * z)
    tails = torch.pow(1.0 + z.square(), k)
    return A + B * skew * tails * z


def generate(bank, theta):
    pi, A1, B1, g1, k1, A2, B2, g2, k2 = theta
    u, x, z = bank
    first = g_and_k(x, A1, B1, g1, k1)
    second = g_and_k(z, A2, B2, g2, k2)
    choose_second = (u < pi).to(first.dtype)
    return ((1.0 - choose_second) * first + choose_second * second)[:, None]


def valid(theta):
    lower, upper = theta.new_tensor(LOWER_BOUNDS), theta.new_tensor(UPPER_BOUNDS)
    return bool(((theta > lower) & (theta < upper)).all())


def _coupling(n_x, n_y):
    key = (n_x, n_y)
    if key not in _W2_COUPLING_CACHE:
        ix, iy, weights = [], [], []
        i = j = 0
        left = 0.0
        while i < n_x and j < n_y:
            next_x, next_y = (i + 1) / n_x, (j + 1) / n_y
            right = min(next_x, next_y)
            ix.append(i)
            iy.append(j)
            weights.append(right - left)
            left = right
            if next_x <= right + 1e-14:
                i += 1
            if next_y <= right + 1e-14:
                j += 1
        _W2_COUPLING_CACHE[key] = (torch.tensor(ix, dtype=torch.long),
                                  torch.tensor(iy, dtype=torch.long),
                                  torch.tensor(weights, dtype=torch.float32))
    return _W2_COUPLING_CACHE[key]


def w2_squared(x, y):
    """Wasserstein-2 distance"""
    x_sorted = torch.sort(x.reshape(-1)).values
    y_sorted = torch.sort(y.reshape(-1)).values
    ix, iy, weights = _coupling(len(x_sorted), len(y_sorted))
    return torch.sum(weights * (x_sorted[ix] - y_sorted[iy]).square())


def _sample(y_real, bank, start, alpha, steps, burn_in, h_values,
            m=None, reference=None, lazy_prob=0.2):
    """MH sampling """
    h = torch.as_tensor(DEFAULT_H_VALUES if h_values is None else h_values,
                        dtype=torch.float32)
    theta = torch.as_tensor(start, dtype=torch.float32).detach().clone()
    alpha = torch.tensor(alpha, dtype=torch.float32)
    pi_sd = torch.sqrt(2.0 * h[0] / alpha)
    eta_sd = torch.sqrt(2.0 * h[1:] / alpha)
    eta_var = 2.0 * h[1:] / alpha
    anchor = None if reference is None else torch.as_tensor(reference, dtype=torch.float32).detach().clone()
    if m is not None:
        bank = sample_latents(m)

    def target_and_gradient(state, latent_bank):
        """risk and gradient"""
        pi = state[0].detach()
        eta = state[1:].detach().clone().requires_grad_(True)
        state_for_grad = torch.cat((pi.reshape(1), eta))
        loss = w2_squared(y_real, generate(latent_bank, state_for_grad))
        if anchor is not None:
            loss = loss - w2_squared(y_real, generate(latent_bank, anchor))
        log_target = -alpha * loss
        gradient = torch.autograd.grad(log_target, eta)[0]
        return log_target.detach(), gradient.detach()

    current, gradient = target_and_gradient(theta, bank)
    chain = np.empty((steps, 9), dtype=np.float32)
    for k in range(steps):
        proposed_bank = bank
        if m is not None and np.random.uniform() >= lazy_prob:
            proposed_bank = sample_latents(m)
        pi_proposal = theta[0] + pi_sd * torch.randn((), dtype=theta.dtype)
        if 0.0 < float(pi_proposal) < 0.5:
            eta = theta[1:]
            mean_forward = eta + (h[1:] / alpha) * gradient
            eta_proposal = mean_forward + eta_sd * torch.randn_like(eta)
            proposal = torch.cat((pi_proposal.reshape(1), eta_proposal))
            if valid(proposal):
                proposed, proposed_gradient = target_and_gradient(proposal, proposed_bank)
                mean_reverse = eta_proposal + (h[1:] / alpha) * proposed_gradient
                log_q_forward = -0.5 * torch.sum((eta_proposal - mean_forward).square() / eta_var)
                log_q_reverse = -0.5 * torch.sum((eta - mean_reverse).square() / eta_var)
                log_ratio = float(proposed - current + log_q_reverse - log_q_forward)
                if np.isfinite(log_ratio) and np.log(np.random.uniform()) < min(0.0, log_ratio):
                    theta, bank = proposal.detach(), proposed_bank
                    current, gradient = proposed, proposed_gradient
        chain[k] = theta.numpy()
    return chain


def sample_full(y_real, bank, steps=200000, burn_in=2000, h_values=None):
    """Full MCMC"""
    _W2_COUPLING_CACHE.clear()
    started = time.perf_counter()
    chain = _sample(y_real, bank, INITIAL_THETA, len(bank[0]), steps, burn_in, h_values)
    return chain, time.perf_counter() - started


def sample_sub(y_real, m=1000, steps=200000, burn_in=2000, alpha=100,
               reference=None, lazy_prob=0.2, h_values=None):
    """Naive when reference=None; otherwise Sub0. """
    started = time.perf_counter()
    start = INITIAL_THETA if reference is None else reference
    chain = _sample(y_real, None, start, alpha, steps, burn_in, h_values,
                    m=m, reference=reference, lazy_prob=lazy_prob)
    return chain, time.perf_counter() - started


def density_grid(log10_values, n_grid=300):
    low, high = float(np.min(log10_values)), float(np.max(log10_values))
    padding = 0.25 * (high - low)
    return np.linspace(low - padding, high + padding, n_grid)


def kde_density(values, grid):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2 or np.var(values) <= 1e-12:
        raise ValueError("KDE requires at least two nonconstant finite values")
    if values.size > 20000:
        values = values[np.linspace(0, values.size - 1, 20000, dtype=int)]
    density = np.maximum(gaussian_kde(values)(grid), 0.0)
    area = float(trapezoid(density, grid))
    if not np.isfinite(area) or area <= 0:
        raise ValueError("Invalid KDE mass on the supplied grid")
    return density / area


def predictive_curves(selected_theta, bank, location, scale, grid):
    curves = np.empty((len(selected_theta), len(grid)), dtype=float)
    with torch.no_grad():
        for i, theta in enumerate(selected_theta):
            state = torch.as_tensor(theta, dtype=torch.float32)
            values = generate(bank, state).squeeze(1).numpy()
            curves[i] = kde_density(location + scale * values, grid)
    return curves
