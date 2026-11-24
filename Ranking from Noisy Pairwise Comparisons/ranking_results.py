from ranking_sampler import generate_data, sample_full, sample_sub
import numpy as np
import arviz as az


def main(
    m=(100, 200, 250, 300),
    Total=1000,
    n=1000,
    KK=10000,
    burn=2000,
    pilot_steps=2000,
    pilot_burn=1000,
    replace=False  # True: with replacement; False: without replacement.
):
    m_values = [m] if np.isscalar(m) else list(m)
    dd = 3
    truth = np.repeat(0.8610463640618999, dd)
    methods = ['Full', 'SMH']

    covpp95 = np.zeros((2, len(m_values), dd))
    CI95 = np.zeros((2, len(m_values), dd))
    diff = np.zeros((1, len(m_values)))
    rdev = np.zeros((1, len(m_values)))
    ESS_table = np.zeros((2, len(m_values)))
    timetable = np.zeros((2, len(m_values)))

    print(f"\n{'=' * 60}")
    print(f"Processing dd = {dd}: Full, SMH")
    print('Minibatch: with replacement' if replace else 'Minibatch: without replacement')
    print(f"{'=' * 60}\n")

    for ite in range(Total):
        # Data generation
        X, C = generate_data(n)

        # Full MCMC
        Theta_full, time_taken = sample_full(X, C, KK=KK, burn=burn)
        q_full = np.percentile(Theta_full, [2.5, 97.5], axis=0)
        CI95[0] += (q_full[1] - q_full[0]) / Total
        covpp95[0] += ((q_full[0] <= truth) & (truth <= q_full[1])) / Total
        ESS_table[0] += np.mean([az.ess(Theta_full[:, d][np.newaxis, :], method="mean")
                                for d in range(dd)]) / Total
        timetable[0] += time_taken / Total

        # Use Full as the reference.
        full_mean = Theta_full.mean(axis=0)
        full_sd = Theta_full.std(axis=0, ddof=1)
        full_length = q_full[1] - q_full[0]

        # Subsampling
        for m_idx, batch in enumerate(m_values):
            Theta, thetahat1, time_taken = sample_sub(
                X, C, batch, KK=KK, burn=burn,
                pilot_steps=pilot_steps, pilot_burn=pilot_burn, replace=replace)
            q = np.percentile(Theta, [2.5, 97.5], axis=0)
            CI95[1, m_idx] += (q[1] - q[0]) / Total
            covpp95[1, m_idx] += ((q[0] <= truth) & (truth <= q[1])) / Total
            diff[0, m_idx] += np.mean(
                (np.abs(q[0] - q_full[0]) + np.abs(q[1] - q_full[1]))
                / full_length) / Total
            rdev[0, m_idx] += np.mean(np.abs((thetahat1 - full_mean) / full_sd)) / Total
            ESS_table[1, m_idx] += np.mean([az.ess(Theta[:, d][np.newaxis, :], method="mean")
                                          for d in range(dd)]) / Total
            timetable[1, m_idx] += time_taken / Total

    print(f"\n{'=' * 60}")
    print(f"Results for dd = {dd}:")
    print(f"{'=' * 60}\n")

    print("Coverage Probability (95% CI):")
    print(f"{'Method':<22}", end='')
    for batch in m_values:
        print(f"{'m=' + str(batch):<10}", end='')
    print()
    for i, method in enumerate(methods):
        for d in range(dd):
            print(f"{method + '_dim' + str(d):<22}", end='')
            for j in range(len(m_values)):
                print(f"{covpp95[i, j, d]:<10.4f}", end='')
            print()
        print()

    print("Average Coverage (across all components):")
    print(f"{'Method':<22}", end='')
    for batch in m_values:
        print(f"{'m=' + str(batch):<10}", end='')
    print()
    for i, method in enumerate(methods):
        print(f"{method:<22}", end='')
        for j in range(len(m_values)):
            print(f"{np.mean(covpp95[i, j, :]):<10.4f}", end='')
        print()
    print()

    print("95% Credible Interval Length:")
    print(f"{'Method':<22}", end='')
    for batch in m_values:
        print(f"{'m=' + str(batch):<10}", end='')
    print()
    for i, method in enumerate(methods):
        for d in range(dd):
            print(f"{method + '_dim' + str(d):<22}", end='')
            for j in range(len(m_values)):
                print(f"{CI95[i, j, d]:<10.4f}", end='')
            print()
        print()

    print("Average CI Length (across all components):")
    print(f"{'Method':<22}", end='')
    for batch in m_values:
        print(f"{'m=' + str(batch):<10}", end='')
    print()
    for i, method in enumerate(methods):
        print(f"{method:<22}", end='')
        for j in range(len(m_values)):
            print(f"{np.mean(CI95[i, j, :]):<10.4f}", end='')
        print()
    print()

    print("Quantile Differences :")
    print(f"{'Method':<22}", end='')
    for batch in m_values:
        print(f"{'m=' + str(batch):<10}", end='')
    print()
    print(f"{'SMH':<22}", end='')
    for j in range(len(m_values)):
        print(f"{diff[0, j]:<10.4f}", end='')
    print('\n')

    print("Relative Deviations :")
    print(f"{'Method':<22}", end='')
    for batch in m_values:
        print(f"{'m=' + str(batch):<10}", end='')
    print()
    print(f"{'SMH':<22}", end='')
    for j in range(len(m_values)):
        print(f"{rdev[0, j]:<10.4f}", end='')
    print('\n')

    print("Effective Sample Size:")
    print(f"{'Method':<22}", end='')
    for batch in m_values:
        print(f"{'m=' + str(batch):<10}", end='')
    print()
    for i, method in enumerate(methods):
        print(f"{method:<22}", end='')
        for j in range(len(m_values)):
            print(f"{ESS_table[i, j]:<10.2f}", end='')
        print()
    print()

    print("Computation Time (seconds):")
    print(f"{'Method':<22}", end='')
    for batch in m_values:
        print(f"{'m=' + str(batch):<10}", end='')
    print()
    for i, method in enumerate(methods):
        print(f"{method:<22}", end='')
        for j in range(len(m_values)):
            print(f"{timetable[i, j]:<10.2f}", end='')
        print()

    return {
        'covpp95': covpp95,
        'CI95': CI95,
        'diff': diff,
        'rdev': rdev,
        'ESS_table': ESS_table,
        'timetable': timetable,
    }


if __name__ == '__main__':
    main()
