# bayes_opt
Looking into Gaussian processes and Bayesian optimization

As of now the repo contains several jupyter notebooks:

1) gp_sampling: sampling from an arbitrary GP with provided covariance matrix
2) gp_posterior: infering a posterior distribution for GP after sampling some of its points
3) gp_ucb_optimization: implementing Upper Confidence Bound paper (Srinivas et.al. - https://arxiv.org/pdf/0912.3995.pdf) - so far I only looked at a subset of ucb rules and settings
4) gp_nei_optimization: implementing Noise Expected Improvement paper (Letham et.al. - https://arxiv.org/pdf/1706.07094.pdf) - so far without constraints and with basic Monte Carlo sampling (no QMC)
