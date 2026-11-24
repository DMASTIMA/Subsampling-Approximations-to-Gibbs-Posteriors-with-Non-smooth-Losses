import time
import numpy as np

DIMENSION = 3


def generate_data(n=1000):
    # Generate covariates and antisymmetric pairwise comparisons.
    random = np.random.default_rng()
    X = np.ascontiguousarray(random.normal(size=(n, DIMENSION)))
    latent = X @ np.ones(DIMENSION) + random.normal(size=n)
    C = np.zeros((n, n), dtype=np.int8)
    for i in range(n - 1):
        labels = np.where(latent[i] - latent[i + 1:] + 0.5 * random.normal(size=n - i - 1) > 0, 1, -1)
        C[i, i + 1:] = labels
        C[i + 1:, i] = -labels
    return X, C


def initialize(X, C):
    # pairwise least-squares estimate
    n = len(X)
    gram = n * (X.T @ X) - np.outer(X.sum(axis=0), X.sum(axis=0))
    return np.ascontiguousarray(np.linalg.solve(gram, X.T @ C.sum(axis=1)))


class Model:
    #
    def __init__(self, X, C):
        self.X, self.C, self.n = X, C, len(X)
        self.i, self.j = np.triu_indices(self.n, 1)
        self.labels = C[self.i, self.j]
        self.scale = 6.0 / (self.n * (self.n - 1))

    def risk(self, theta):
        scores = self.X @ theta
        margin = self.labels * (scores[self.i] - scores[self.j])
        return self.scale * np.maximum(0.0, 1.0 - margin).sum()

    def gradient(self, theta):
        scores = self.X @ theta
        active = self.labels * (scores[self.i] - scores[self.j]) < 1.0
        weights = self.labels * active
        node_weights = (np.bincount(self.j, weights=weights, minlength=self.n)
                        - np.bincount(self.i, weights=weights, minlength=self.n))
        gradient = self.scale * (node_weights @ self.X)
        directions = np.full((self.n, self.n), -1, dtype=np.int8)
        directions[self.i[active], self.j[active]] = 1
        directions[self.j[active], self.i[active]] = 1
        return gradient, directions


def sample_full(X, C, KK=10000, burn=2000, omega=1.0, sigma=1.5):
    #Full MCMC
    start = time.perf_counter()
    theta = initialize(X, C)
    model = Model(X, C)
    random = np.random.default_rng()
    alpha = omega * len(X)
    scale = sigma / np.sqrt(alpha)
    current = model.risk(theta)
    Theta = np.empty((KK - burn, DIMENSION))
    for step in range(KK):
        proposal = theta + scale * random.normal(size=DIMENSION)
        if np.all(np.abs(proposal) <= 10.0):
            proposed = model.risk(proposal)
            if np.log(random.random()) < min(0.0, -alpha * (proposed - current)):
                theta, current = proposal, proposed
        if step >= burn:
            Theta[step - burn] = theta
    return Theta, time.perf_counter() - start


def sample_stage(model, initial, m, alpha, sigma, steps, burn, reference=None,
                 replace=False):
    # Minibatch MH sampling with optional first-order control variates.
    random = np.random.default_rng()
    subset_random = np.random.default_rng()
    theta = np.asarray(initial, dtype=float).copy()
    i, j = np.triu_indices(m, 1)
    gradient, directions = (None, None) if reference is None else model.gradient(reference)
    scale = 6.0 / (m * (m - 1))
    if replace:
        scale *= model.n / (model.n - 1)

    def draw_batch():
        indices = np.sort(subset_random.choice(model.n, m, replace=replace))
        first, second = indices[i], indices[j]
        direction = np.full(len(i), -1, dtype=np.int8) if directions is None else directions[first, second]
        # Repeated copies of the same observation contribute zero pair loss.
        direction = np.where(first == second, 0, direction)
        coefficient = direction * model.C[first, second]
        return model.X[indices], coefficient, direction

    def log_target(state, batch):
        inputs, coefficient, direction = batch
        scores = inputs @ state
        value = scale * np.maximum(0.0, coefficient * (scores[i] - scores[j]) - direction).sum()
        if reference is not None:
            value += gradient @ (state - reference)
        return -alpha * value

    batch = draw_batch()
    current = log_target(theta, batch)
    proposal_scale = sigma / np.sqrt(alpha)
    Theta = np.empty((steps - burn, DIMENSION))
    for step in range(steps):
        proposal = theta + proposal_scale * random.normal(size=DIMENSION)
        proposed_batch = draw_batch() if random.random() >= 0.5 else batch
        if np.all(np.abs(proposal) <= 10.0):
            proposed = log_target(proposal, proposed_batch)
            if np.log(random.random()) < min(0.0, proposed - current):
                theta, batch, current = proposal, proposed_batch, proposed
        if step >= burn:
            Theta[step - burn] = theta
    return Theta


def sample_sub(X, C, m, KK=10000, burn=2000, pilot_steps=2000,
               pilot_burn=1000, omega=1.0, sigma=1.5, replace=False):
    #SMH
    start = time.perf_counter()
    initial = initialize(X, C)
    model = Model(X, C)
    # Use the same replacement mode in all three stages.
    first = sample_stage(model, initial, m, omega * 10 * m**0.4,
                         sigma, pilot_steps, pilot_burn, replace=replace)
    reference1 = first.mean(axis=0)
    second = sample_stage(model, reference1, m, omega * 10 * m**0.8,
                          sigma, pilot_steps, pilot_burn, reference1,
                          replace=replace)
    reference2 = second.mean(axis=0)
    Theta = sample_stage(model, reference2, m, omega * len(X),
                         sigma, KK, burn, reference2, replace=replace)
    return Theta, Theta.mean(axis=0), time.perf_counter() - start
