import numpy as np
import arviz as az
import warnings
from Robust_classification_sampler import generate_data, sample_z, sample_full, sample_smh

warnings.filterwarnings('ignore')


def main(
    m=(200, 400, 600, 800, 1000),
    Total=100,
    n=2000,
    KK=10000,
    N_Z=20000,
    burn_in=1000,
    pilot_steps=2000,
    pilot_burn_in=1000
):
    m_values = [m] if np.isscalar(m) else list(m)
    dd = 3
    all_results = {}

    print(f"\n{'=' * 60}")
    print(f"Processing dd = {dd}")
    print(f"{'=' * 60}\n")

    CI95 = np.zeros((3, len(m_values), dd))  # CI length
    diff = np.zeros((2, len(m_values)))  # Quantile difference
    rdev = np.zeros((2, len(m_values)))  # Relative deviation
    ESS_table = np.zeros((3, len(m_values)))  # ESS
    timetable = np.zeros((3, len(m_values)))  # Time

    for ite in range(Total):
        # Generate data
        X, Y = generate_data(n)
        Z_full = sample_z(N_Z, X.device)

        # ========== Full MH ==========
        Theta_full, time_taken = sample_full(X, Y, Z_full, steps=KK, burn_in=burn_in)

        ess_full = np.mean([az.ess(Theta_full[:, i][np.newaxis, :], method="mean")
                            for i in range(dd)])
        for d in range(dd):
            q1_full = np.percentile(Theta_full[:, d], 97.5)
            q2_full = np.percentile(Theta_full[:, d], 2.5)
            CI95[0, :, d] += (q1_full - q2_full) / Total
        ESS_table[0, :] += ess_full / Total
        timetable[0, :] += time_taken / Total

        # ========== Smooth MH ==========
        Theta, time_taken = sample_full(
            X, Y, Z_full, steps=KK, burn_in=burn_in, smooth=True)

        ess = np.mean([az.ess(Theta[:, i][np.newaxis, :], method="mean")
                       for i in range(dd)])
        rdev[0, :] += np.mean(
            np.abs((np.mean(Theta, axis=0) - np.mean(Theta_full, axis=0)) /
                   np.std(Theta_full, axis=0))
        ) / Total

        q1_list = []
        q2_list = []
        for d in range(dd):
            q1 = np.percentile(Theta[:, d], 97.5)
            q2 = np.percentile(Theta[:, d], 2.5)
            q1_list.append(q1)
            q2_list.append(q2)
            CI95[1, :, d] += (q1 - q2) / Total
        diff[0, :] += np.mean([
            (np.abs(q1_list[d] - np.percentile(Theta_full[:, d], 97.5))
             + np.abs(q2_list[d] - np.percentile(Theta_full[:, d], 2.5))) /
            (np.percentile(Theta_full[:, d], 97.5)
             - np.percentile(Theta_full[:, d], 2.5)) for d in range(dd)
        ]) / Total
        ESS_table[1, :] += ess / Total
        timetable[1, :] += time_taken / Total

        # ========== SMH ==========
        for m_idx, m in enumerate(m_values):
            Theta, thetahat1, time_taken = sample_smh(
                X, Y, Z_full, m, steps=KK, burn_in=burn_in,
                pilot_steps=pilot_steps, pilot_burn_in=pilot_burn_in)

            ess = np.mean([az.ess(Theta[:, i][np.newaxis, :], method="mean")
                           for i in range(dd)])
            rdev[1, m_idx] += np.mean(
                np.abs((np.mean(Theta, axis=0) - np.mean(Theta_full, axis=0)) /
                       np.std(Theta_full, axis=0))
            ) / Total

            q1_list = []
            q2_list = []
            for d in range(dd):
                q1 = np.percentile(Theta[:, d], 97.5)
                q2 = np.percentile(Theta[:, d], 2.5)
                q1_list.append(q1)
                q2_list.append(q2)
                CI95[2, m_idx, d] += (q1 - q2) / Total
            diff[1, m_idx] += np.mean([
                (np.abs(q1_list[d] - np.percentile(Theta_full[:, d], 97.5))
                 + np.abs(q2_list[d] - np.percentile(Theta_full[:, d], 2.5))) /
                (np.percentile(Theta_full[:, d], 97.5)
                 - np.percentile(Theta_full[:, d], 2.5)) for d in range(dd)
            ]) / Total
            ESS_table[2, m_idx] += ess / Total
            timetable[2, m_idx] += time_taken / Total

    all_results[dd] = {
        'CI95': CI95,
        'diff': diff,
        'rdev': rdev,
        'ESS_table': ESS_table,
        'timetable': timetable,
    }

    print(f"\n{'=' * 60}")
    print(f"Results for dd = {dd}:")
    print(f"{'=' * 60}\n")

    print("95% Credible Interval Length:")
    print(f"{'Method':<18}", end='')
    for m in m_values:
        print(f"m={m:<6}", end='')
    print()
    for i, method in enumerate(['Full MH', 'Smooth MH', 'SMH']):
        for d in range(dd):
            print(f"{method + '_dim' + str(d):<18}", end='')
            for j in range(len(m_values)):
                print(f"{CI95[i, j, d]:.4f}  ", end='')
            print()
        print()

    print("Average CI Length (across all components):")
    print(f"{'Method':<18}", end='')
    for m in m_values:
        print(f"m={m:<6}", end='')
    print()
    for i, method in enumerate(['Full MH', 'Smooth MH', 'SMH']):
        print(f"{method:<18}", end='')
        for j in range(len(m_values)):
            print(f"{np.mean(CI95[i, j, :]):.4f}  ", end='')
        print()
    print()

    print("\nQuantile Differences:")
    print(f"{'Method':<18}", end='')
    for m in m_values:
        print(f"m={m:<6}", end='')
    print()
    for i, method in enumerate(['Smooth MH', 'SMH']):
        print(f"{method:<18}", end='')
        for j in range(len(m_values)):
            print(f"{diff[i, j]:.4f}  ", end='')
        print()

    print("\nRelative Deviations:")
    print(f"{'Method':<18}", end='')
    for m in m_values:
        print(f"m={m:<6}", end='')
    print()
    for i, method in enumerate(['Smooth MH', 'SMH']):
        print(f"{method:<18}", end='')
        for j in range(len(m_values)):
            print(f"{rdev[i, j]:.4f}  ", end='')
        print()

    print("\nEffective Sample Size:")
    print(f"{'Method':<18}", end='')
    for m in m_values:
        print(f"m={m:<6}", end='')
    print()
    for i, method in enumerate(['Full MH', 'Smooth MH', 'SMH']):
        print(f"{method:<18}", end='')
        for j in range(len(m_values)):
            print(f"{ESS_table[i, j]:.2f}  ", end='')
        print()

    print("\nComputation Time (seconds):")
    print(f"{'Method':<18}", end='')
    for m in m_values:
        print(f"m={m:<6}", end='')
    print()
    for i, method in enumerate(['Full MH', 'Smooth MH', 'SMH']):
        print(f"{method:<18}", end='')
        for j in range(len(m_values)):
            print(f"{timetable[i, j]:.2f}  ", end='')
        print()

    return all_results


if __name__ == "__main__":
    main()
