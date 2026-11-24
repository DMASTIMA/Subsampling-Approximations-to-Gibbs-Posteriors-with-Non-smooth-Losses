

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde

from Robust_classification_sampler import generate_data, sample_z, sample_full, sample_smh


def plot(samples, reference, output_dir="classification_plots"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"Full MH": "#ff7f0e", "Smooth MH": "#2ca02c", "SMH": "#1f77b4"}
    curves = []
    for j in range(3):
        estimators = {name: gaussian_kde(chain[:, j]) for name, chain in samples.items()}
        padding = 4 * max(float(np.sqrt(kde.covariance[0, 0])) for kde in estimators.values())
        low = min(reference[j], min(chain[:, j].min() for chain in samples.values()))
        high = max(reference[j], max(chain[:, j].max() for chain in samples.values()))
        grid = np.linspace(low - padding, high + padding, 1000)
        curves.append((grid, {name: kde(grid) for name, kde in estimators.items()}))

    def draw(ax, j):
        grid, densities = curves[j]
        for name, density in densities.items():
            ax.plot(grid, density, color=colors[name], label=name, linewidth=1.6)
            ax.fill_between(grid, density, color=colors[name], alpha=0.15)
        ax.axvline(reference[j], color="gray", linestyle="--", label="Reference point")
        ax.set(xlabel=rf"$\theta_{{{j + 1}}}$", ylabel="Density", title="Posterior KDE",
               xlim=(grid[0], grid[-1]), ylim=(0, None))
        ax.legend(fontsize=9)

    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(output_dir / f"{name}.{extension}", dpi=200, facecolor="white")
        plt.close(fig)

    for j in range(3):
        fig, ax = plt.subplots(figsize=(7, 5))
        draw(ax, j)
        save(fig, f"theta_{j + 1}")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for j, ax in enumerate(axes):
        draw(ax, j)
    save(fig, "kde_overview")


def main(n=2000, N_Z=20000, m=1000, KK=10000, burn_in=1000,
         pilot_steps=2000, pilot_burn_in=1000, device=None, seed=2026,
         output_dir="classification_plots"):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    X, Y = generate_data(n, device)
    Z_full = sample_z(N_Z, X.device)
    print("Full MH", flush=True)
    full, _ = sample_full(X, Y, Z_full, KK, burn_in)
    print("Smooth MH", flush=True)
    smooth, _ = sample_full(X, Y, Z_full, KK, burn_in, smooth=True)
    print(f"SMH (m={m})", flush=True)
    smh, reference, _ = sample_smh(X, Y, Z_full, m, KK, burn_in,
                                  pilot_steps, pilot_burn_in)
    samples = {"Full MH": full, "Smooth MH": smooth, "SMH": smh}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "chains.npz", full=full, smooth=smooth, smh=smh,
                        reference=reference, m=m, seed=seed)
    plot(samples, reference, output_dir)
    print(f"Saved chains and figures to {output_dir.resolve()}")
    return samples, reference


if __name__ == "__main__":
    main()
