# bayes_opt
Looking into Gaussian processes and Bayesian optimization

As of now the repo contains several jupyter notebooks (in the notebooks folder):

1) gp_sampling: sampling from an arbitrary GP with provided covariance matrix
2) gp_posterior: infering a posterior distribution for GP after sampling some of its points
3) gp_ucb_optimization: implementing Upper Confidence Bound paper (Srinivas et.al. - https://arxiv.org/pdf/0912.3995.pdf) - so far I only looked at a subset of ucb rules and settings
4) gp_nei_optimization: implementing Noise Expected Improvement paper (Letham et.al. - https://arxiv.org/pdf/1706.07094.pdf) - so far without constraints and with basic Monte Carlo sampling (no QMC)
5) gp_mcmc_ei_optimization: adding MCMC sampling of GP kernel hyperparameters during EI optimization (following advice in Snoek et.al. paper - https://arxiv.org/pdf/1206.2944.pdf)

Here is an example of EI Bayesian optimization with MCMC sampling of kernel hyperparameters (as in gp_mcmc_ei_optimization notebook):

![Alt Text](https://media.giphy.com/media/26DN9FdaO30vVTtBu/giphy.gif)

On top of that gp.py and bayes_opt.py in the main folder contain refactored code from the notebooks, check test_mcmc_impact.py to check their usage.
