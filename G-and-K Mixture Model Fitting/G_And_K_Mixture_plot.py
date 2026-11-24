from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from G_And_K_Mixture_sampler import (
    DEFAULT_H_VALUES, load_data, sample_latents, sample_full, sample_sub,
    density_grid, kde_density, predictive_curves,
)


def plot(data, output_dir="gk_plots"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = [("Original", "Full MH"), ("NCV", "Naive"), ("ZCV", "Sub0")]
    grid = data["grid"]

    def draw(ax, method, label):
        ax.plot(grid, data["observed_density"], color="#1f77b4", linewidth=2,
                label="Observed density")
        ax.plot(grid, data[f"{method}_mean"], color="#ff7f0e", linewidth=2,
                label="Mean predictive density")
        ax.plot(grid, data[f"{method}_q25"], color="#2ca02c", linestyle="--",
                label="25th percentile")
        ax.plot(grid, data[f"{method}_q75"], color="#d62728", linestyle=":",
                label="75th percentile")
        ax.fill_between(grid, data[f"{method}_q25"], data[f"{method}_q75"],
                        color="#cfe9fb", alpha=0.65, label="IQR (25–75%)", zorder=0)
        ax.set(title=label, xlabel=r"$\log_{10}(T_{90})$", ylabel="Density",
               xlim=(grid[0], grid[-1]), ylim=(0, None))
        ax.grid(axis="y", color="#e8e8e8")

    def save(fig, name):
        for extension in ("png", "pdf"):
            fig.savefig(output_dir / f"{name}.{extension}", dpi=220,
                        bbox_inches="tight", facecolor="white")
        plt.close(fig)

    for method, label in methods:
        fig, ax = plt.subplots(figsize=(7.4, 5.4))
        draw(ax, method, label)
        ax.legend(fontsize=9)
        fig.tight_layout()
        save(fig, {"Original": "full_mh_density", "NCV": "naive_density",
                   "ZCV": "sub0_density"}[method])

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.5), sharey=True)
    ymax = max(np.max(data["observed_density"]),
               *(np.max(data[f"{method}_{key}"])
                 for method, _ in methods for key in ("mean", "q25", "q75")))
    for ax, (method, label) in zip(axes, methods):
        draw(ax, method, label)
        ax.set_ylim(0, 1.08 * ymax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "figure2")


def main(
    m=1000,                # Latent batch size for Naive and Sub0
    N=20000,               # Fixed latent sample size  for Full MH
    KK=200000,             # Total iterations
    burn_in=2000,          # Burn-in length
    thin=10,               # Select every thin-th state for predictive density estimation
    predictive_size=2000,  # Predictive sample size for each selected parameter state
    grid_size=300,         # Number of grid points for density curves
    alpha_naive=100,       # Naive inverse temperature
    alpha_sub0=250,        # Sub0 inverse temperature; also controls proposal scaling
    lazy_prob=0.2,         # Laziness
    h_values=None,         # Proposal step-size coefficients
    include_burn_in=True,  # Include thinned burn-in states in predictive density estimation
    threads=1,            # Number of PyTorch CPU threads
    output_dir="gk_plots"  # Directory for figures and saved plotting data
):
    # thin=10 selects iterations 10, 20, 30, ...
    indices = np.arange(thin - 1, KK, thin)
    if not include_burn_in:
        indices = indices[indices >= burn_in]
    torch.set_num_threads(threads)
    y_real, log10_values, location, scale = load_data()
    bank = sample_latents(N)
    print("Full MH", flush=True)
    full, _ = sample_full(y_real, bank, KK, burn_in, h_values)
    print("Naive", flush=True)
    naive, _ = sample_sub(y_real, m, KK, burn_in, alpha_naive,
                          lazy_prob=lazy_prob, h_values=h_values)
    reference = naive[burn_in:].mean(axis=0)
    print("Sub0", flush=True)
    sub0, _ = sample_sub(y_real, m, KK, burn_in, alpha_sub0, reference,
                         lazy_prob=lazy_prob, h_values=h_values)

    selected = {"Original": full[indices], "NCV": naive[indices], "ZCV": sub0[indices]}
    predictive_bank = sample_latents(predictive_size)
    grid = density_grid(log10_values, grid_size)
    data = {"grid": grid, "observed_density": kde_density(log10_values, grid)}
    for method, theta in selected.items():
        curves = predictive_curves(theta, predictive_bank, location, scale, grid)
        q25, q75 = np.quantile(curves, [0.25, 0.75], axis=0)
        data.update({f"{method}_mean": curves.mean(axis=0),
                     f"{method}_q25": q25, f"{method}_q75": q75})

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "density_summary.npz", **data, m=m, N=N, KK=KK,
                        burn_in=burn_in, thin=thin, predictive_size=predictive_size,
                        grid_size=grid_size, alpha_naive=alpha_naive, alpha_sub0=alpha_sub0,
                        lazy_prob=lazy_prob, include_burn_in=include_burn_in,
                        h_values=DEFAULT_H_VALUES if h_values is None else h_values)
    np.savez_compressed(output_dir / "selected_theta_and_latents.npz", **selected,
                        iterations=indices + 1, in_burn_in=indices < burn_in,
                        sub0_anchor=reference, predictive_U=predictive_bank[0].numpy(),
                        predictive_Z1=predictive_bank[1].numpy(),
                        predictive_Z2=predictive_bank[2].numpy(),
                        standardization_mean=location, standardization_std=scale)
    plot(data, output_dir)
    return data


if __name__ == "__main__":
    main()
