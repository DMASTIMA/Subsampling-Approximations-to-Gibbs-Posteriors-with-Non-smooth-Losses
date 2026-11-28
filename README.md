# Subsampling-Approximations-to-Gibbs-Posteriors-with-Non-smooth-Losses
Python code of Subsampling Approximations to Gibbs Posteriors with Non-smooth Losses

## 1. Bayesian Huber Regression

Huber_Regression_sampler.py: It provides the Metropolis-Hastings samplers with MALA proposals for the Full-batch MH algorithm, as well as the zero-order control variate (Sub0) and first-order control variate (Sub1) subsampling algorithms. It supports mini-batch sampling with or without replacement.

Huber_Regression_results.py: It tracks and compares 95% credible interval coverage, 95% credible interval length, quantile differences, mean differences—alongside computational efficiency metrics, including Effective Sample Size (ESS) and computation time, across varying mini-batch sizes and dimensions.

## 2. Ranking from Noisy Pairwise Comparisons

ranking_sampler.py: It provides the Random Walk Metropolis-Hastings samplers for the Full-batch MH algorithm and a three-stage subsampling MH algorithm (Naive pilot → Sub1 pilot → final Sub1 sampling). The subsampling algorithm progressively updates the reference point using the pilot samples. It supports mini-batch sampling with or without replacement.

ranking_results.py: It compares the Full-batch MH and three-stage subsampling MH algorithms across varying mini-batch sizes. It evaluates 95% credible interval coverage, 95% credible interval length, quantile differences, mean differences, ESS, and computation time.

## 3. Robust Classification under Gaussian Perturbations

Robust_classification_sampler.py: It provides the Random Walk Metropolis-Hastings samplers for the Full MH and Smooth MH algorithms, as well as a two-stage subsampling MH algorithm (Naive pilot → Sub1 sampling). The Full MH algorithm uses the classification risk under Gaussian perturbations, while Smooth MH uses a sigmoid-smoothed version of this risk. The subsampling algorithm uses the Naive pilot to estimate a reference point and constructs first-order control variates from the smoothed risk gradients.

Robust_classification_results.py: It compares the Full MH, Smooth MH, and subsampling MH algorithms across varying Gaussian perturbation batch sizes. It evaluates 95% credible interval lengths, quantile differences, mean differences, ESS, and computation time.

Robust_classification_plot.py: It generates marginal posterior density plots for the three model parameters to compare the Full MH, Smooth MH, and subsampling MH algorithms, with the reference point marked in each plot.

## 4. G-and-K Mixture Model Fitting

G_And_K_Mixture_sampler.py: It provides the samplers for the Full MH, Naive, and Sub0 algorithms for a two-component g-and-k mixture model with a squared 2-Wasserstein loss. It combines a random walk proposal for the mixture weight with MALA proposals for the component parameters. The subsampling algorithms use mini-batches of latent variables.

G_And_K_Mixture_results.py: It evaluates the Full MH, Naive, and Sub0 algorithms on the BATSE duration data across repeated runs. It compares the L1 and L2 distances between the mean predictive density and the observed-data density estimate, alongside computation time. The Sub0 algorithm uses the posterior mean from a Naive run as its reference point.
