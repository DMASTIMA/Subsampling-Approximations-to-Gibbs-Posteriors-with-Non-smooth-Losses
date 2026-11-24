import numpy as np
import arviz as az
import warnings
from Huber_Regression_sampler import generate_data, sample_full, sample_sub

warnings.filterwarnings('ignore')


def main(
    d=(2, 5, 10),
    m=(200, 400, 800, 1200, 1600, 2000),
    Total=100,
    n=20000,
    KK=20000,
    replace=False  # True: with replacement; False: without replacement.
):
    """Main function"""
    m_values = [m] if np.isscalar(m) else list(m)
    dd_values = [d] if np.isscalar(d) else list(d)


    all_results = {}

    for dd in dd_values:
        print(f"\n{'=' * 60}")
        print(f"Processing dd = {dd}")
        print('Minibatch: with replacement' if replace else 'Minibatch: without replacement')
        print(f"{'=' * 60}\n")


        covpp95 = np.zeros((3, len(m_values), dd))  #Coverage
        CI95 = np.zeros((3, len(m_values), dd))  # CI length
        diff = np.zeros((2, len(m_values)))  # Quantile difference
        rdev = np.zeros((2, len(m_values)))  # Relative Deviation
        ESS_table = np.zeros((3, len(m_values)))  # Quantile
        timetable = np.zeros((3, len(m_values)))  # Time


        for ite in range(Total):

            # Generate data
            X, Y = generate_data(dd, n)

            # ========== Full Batch MCMC ==========
            Theta_full, time_taken = sample_full(X, Y, dd, n, KK)


            ess_full = np.mean([az.ess(Theta_full[:, i][np.newaxis, :], method="mean")
                                for i in range(dd)])

            for d in range(dd):
                q1_full = np.percentile(Theta_full[:, d], 97.5)
                q2_full = np.percentile(Theta_full[:, d], 2.5)
                CI95[0, :, d] += (q1_full - q2_full) / Total
                if 1 <= q1_full and 1 >= q2_full:
                    covpp95[0, :, d] += 1 / Total
            ESS_table[0, :] += ess_full / Total
            timetable[0, :] += time_taken / Total



            # ========== Control variate methods ==========
            for m_idx, m in enumerate(m_values):
                for order in [0, 1]:
                    Theta, thetahat1, time_taken = sample_sub(X, Y, dd, n, m, KK, order, replace=replace)


                    ess = np.mean([az.ess(Theta[:, i][np.newaxis, :], method="mean")
                                   for i in range(dd)])
                    rdev[order, m_idx] += np.mean(
                        np.abs((thetahat1 - np.mean(Theta_full, axis=0)) /
                               np.std(Theta_full, axis=0))
                    ) / Total

                    method_idx = order + 1
                    q1_list = []
                    q2_list = []
                    for d in range(dd):
                        q1 = np.percentile(Theta[:, d], 97.5)
                        q2 = np.percentile(Theta[:, d], 2.5)
                        q1_list.append(q1)
                        q2_list.append(q2)
                        CI95[method_idx, m_idx, d] += (q1 - q2) / Total
                        if 1 <= q1 and 1 >= q2:
                            covpp95[method_idx, m_idx, d] += 1 / Total
                    diff[order, m_idx] += np.mean([(np.abs(q1_list[d] - np.percentile(Theta_full[:, d], 97.5))
                                         + np.abs(q2_list[d] - np.percentile(Theta_full[:, d], 2.5))) / (
                                                               np.percentile(Theta_full[:, d], 97.5) - np.percentile(
                                                           Theta_full[:, d], 2.5)) for d in range(dd)]) / Total
                    ESS_table[method_idx, m_idx] += ess / Total
                    timetable[method_idx, m_idx] += time_taken / Total




        all_results[dd] = {
            'covpp95': covpp95,
            'CI95': CI95,
            'diff': diff,
            'rdev': rdev,
            'ESS_table': ESS_table,
            'timetable': timetable,
        }


        print(f"\n{'=' * 60}")
        print(f"Results for dd = {dd}:")
        print(f"{'=' * 60}\n")

        print("Coverage Probability (95% CI):")
        print(f"{'Method':<10}", end='')
        for m in m_values:
            print(f"m={m:<6}", end='')
        print()

        for i, method in enumerate(['Full', 'Sub0', 'Sub1']):
            for d in range(dd):
                print(f"{method}_dim{d:<9}", end='')
                for j in range(len(m_values)):
                    print(f"{covpp95[i, j, d]:.4f}  ", end='')
                print()
            print()

        print("Average Coverage (across all components):")
        print(f"{'Method':<10}", end='')
        for m in m_values:
            print(f"m={m:<6}", end='')
        print()
        for i, method in enumerate(['Full', 'Sub0', 'Sub1']):
            print(f"{method:<10}", end='')
            for j in range(len(m_values)):
                print(f"{np.mean(covpp95[i, j, :]):.4f}  ", end='')
            print()
        print()

        print("95% Confidence Interval Length:")
        print(f"{'Method':<10}", end='')
        for m in m_values:
            print(f"m={m:<6}", end='')
        print()

        for i, method in enumerate(['Full', 'Sub0', 'Sub1']):
            for d in range(dd):
                print(f"{method}_dim{d:<9}", end='')
                for j in range(len(m_values)):
                    print(f"{CI95[i, j, d]:.4f}  ", end='')
                print()
            print()

        print("Average CI Length (across all components):")
        print(f"{'Method':<10}", end='')
        for m in m_values:
            print(f"m={m:<6}", end='')
        print()
        for i, method in enumerate(['Full', 'Sub0', 'Sub1']):
            print(f"{method:<10}", end='')
            for j in range(len(m_values)):
                print(f"{np.mean(CI95[i, j, :]):.4f}  ", end='')
            print()
        print()

        print("\nQuantile Differences:")
        print(f"{'Method':<10}", end='')
        for m in m_values:
            print(f"m={m:<6}", end='')
        print()
        for i, method in enumerate(['Sub0', 'Sub1']):
            print(f"{method:<10}", end='')
            for j in range(len(m_values)):
                print(f"{diff[i, j]:.4f}  ", end='')
            print()

        print("\nRelative Deviations:")
        print(f"{'Method':<10}", end='')
        for m in m_values:
            print(f"m={m:<6}", end='')
        print()
        for i, method in enumerate(['Sub0', 'Sub1']):
            print(f"{method:<10}", end='')
            for j in range(len(m_values)):
                print(f"{rdev[i, j]:.4f}  ", end='')
            print()

        print("\nEffective Sample Size:")
        print(f"{'Method':<10}", end='')
        for m in m_values:
            print(f"m={m:<6}", end='')
        print()
        for i, method in enumerate(['Full', 'Sub0', 'Sub1']):
            print(f"{method:<10}", end='')
            for j in range(len(m_values)):
                print(f"{ESS_table[i, j]:.2f}  ", end='')
            print()

        print("\nComputation Time (seconds):")
        print(f"{'Method':<10}", end='')
        for m in m_values:
            print(f"m={m:<6}", end='')
        print()
        for i, method in enumerate(['Full', 'Sub0', 'Sub1']):
            print(f"{method:<10}", end='')
            for j in range(len(m_values)):
                print(f"{timetable[i, j]:.2f}  ", end='')
            print()

    return all_results


if __name__ == "__main__":
    main()
warnings.filterwarnings('ignore')
