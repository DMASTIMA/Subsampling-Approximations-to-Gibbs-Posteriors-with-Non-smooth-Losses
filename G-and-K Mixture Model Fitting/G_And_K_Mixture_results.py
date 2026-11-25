import time
import numpy as np
import torch
from scipy.integrate import trapezoid
from G_And_K_Mixture_sampler import (
    load_data, sample_latents, sample_full, sample_sub,
    density_grid, kde_density, predictive_curves,
)


def main(
    m=1000,                # Latent batch size for Naive and Sub0
    Total=30,              # Independent replications
    N=20000,               # Fixed latent sample size  for Full MH
    KK=200000,             # Total iterations
    burn_in=2000,          # Burn-in length
    thin=10,               # Select every thin-th state for predictive density estimation
    predictive_size=2000,  # Predictive sample size for each selected parameter state
    grid_size=300,         # Grid points for density evaluation and L1/L2 integration
    alpha_naive=100.0,     # Naive inverse temperature
    alpha_sub0=250.0,      # Sub0 inverse temperature
    lazy_prob=0.2,         # Laziness
    h_values=None,         #  step-size coefficients
    include_burn_in=True,  # Include thinned burn-in states in predictive density estimation
    threads=1             # Number of PyTorch CPU threads
):
    torch.set_num_threads(threads)
    # thin=10 selects iterations 10, 20, 30, ...
    indices = np.arange(thin - 1, KK, thin)
    if not include_burn_in:
        indices = indices[indices >= burn_in]

    y_real, log10_values, location, scale = load_data()
    grid = density_grid(log10_values, grid_size)
    observed_density = kde_density(log10_values, grid)
    methods = ['Full MH', 'Naive', 'Sub0']
    L1 = np.zeros(3)
    L2 = np.zeros(3)
    timetable = np.zeros(3)

    print(f"\n{'=' * 60}")
    print(f"BATSE G-and-K: n={len(y_real)}, m={m}, repetitions={Total}")
    print(f"{'=' * 60}\n")

    for ite in range(Total):
        # Each repetition uses the same observed data and a new latent bank.
        bank = sample_latents(N)

        # ========== Full MH ==========
        Theta_full, time_full = sample_full(
            y_real, bank, steps=KK, burn_in=burn_in, h_values=h_values)

        # ========== Naive ==========
        Theta_naive, time_naive = sample_sub(
            y_real, m=m, steps=KK, burn_in=burn_in, alpha=alpha_naive,
            lazy_prob=lazy_prob, h_values=h_values)

        # ========== Sub0 ==========
        started = time.perf_counter()
        reference = Theta_naive[burn_in:].mean(axis=0)
        reference_time = time.perf_counter() - started
        Theta_sub0, time_sub0 = sample_sub(
            y_real, m=m, steps=KK, burn_in=burn_in, alpha=alpha_sub0,
            reference=reference, lazy_prob=lazy_prob, h_values=h_values)

        timetable[0] += time_full / Total
        timetable[1] += time_naive / Total
        timetable[2] += time_sub0 / Total

        # One predictive bank is shared by all states and all three methods.
        predictive_bank = sample_latents(predictive_size)
        for i, Theta in enumerate([Theta_full, Theta_naive, Theta_sub0]):
            curves = predictive_curves(Theta[indices], predictive_bank, location, scale, grid)
            difference = curves.mean(axis=0) - observed_density
            L1[i] += trapezoid(np.abs(difference), x=grid) / Total
            L2[i] += np.sqrt(trapezoid(difference ** 2, x=grid)) / Total

    print("L1 Density Distance:")
    print(f"{'Method':<18}{'L1':>12}")
    for i, method in enumerate(methods):
        print(f"{method:<18}{L1[i]:12.6f}")

    print("\nL2 Density Distance:")
    print(f"{'Method':<18}{'L2':>12}")
    for i, method in enumerate(methods):
        print(f"{method:<18}{L2[i]:12.6f}")

    print("\nComputation Time (seconds):")
    print(f"{'Method':<18}{'Time':>12}")
    for i, method in enumerate(methods):
        print(f"{method:<18}{timetable[i]:12.2f}")

    return {'L1': L1, 'L2': L2, 'timetable': timetable}


if __name__ == "__main__":
    main()
