# This script implements class for run Bayesian Optimizations
# For now it is implemented with EI as an acqusition function
# More acquisition functions may follow 

import math
import numpy as np
import scipy as sp
import scipy.stats
from gp import GP

# Dummy class just to hold details of MCMC
class MCMCSampler:
    def __init__(self, log_likelihood, mcmc_opts):
        # Class for doing MCMC sampling:
        # Below l is supposed to stand for kernel hyperparameters
        # log_likelihood: log_likelihood function (l -> log_likelihood)
        # mcmc_opts is supposed to be a map with the following entries:
        # 'prior': function for prior pdf (l -> pdf value)
        # 'icdf': function for inverse cdf ([0, 1] -> l)
        # 'jump': function for mcmc exploration (l -> l)
        # 'burn_period': number of mcmc iterations to discard before sampling
        # 'mcmc_samples': number of kernel hyperparameters to return
        self.log_likelihood = log_likelihood
        self.mcmc_opts = mcmc_opts

    def posterior_sample(self):
        # Below l is supposed to stand for kernel hyperparameters
        # This function performs Bayesian MCMC sampling for Gaussian kernel hyperparameters
        # Specifically, the first point is sampled using inverse cdf for l_prior
        # Moves are suggested using l_jump function
        # Moves are accepted / rejected with Metropolis-Hastings algorithms
        # (i.e. true posterior density is proportional to exp(log_likelihood) * l_prior, 
        # ratio of posterior values give the probability of acception a move)
        # First burn_period samples of l are discarded and n_samples consecutive samples are the output of a function
        
        # MCMC is concerned with the ratio of true probabilities
        # However, for efficiency reasons we express everything through log-likelihoods
        log_posterior = lambda l: self.log_likelihood(l) + np.log(self.mcmc_opts["prior"](l))
        
        l = self.mcmc_opts["icdf"](np.random.rand())
        past_log_posterior = log_posterior(l)
        for _ in range(self.mcmc_opts["burn_period"]):
            # Adding try except block in case log_posterior sampling fails
            # May happen if l jumps to region outside og prior domain
            try:
                next_l = self.mcmc_opts["jump"](l)
                next_log_posterior = log_posterior(next_l)
                if np.log(np.random.randn()) < (next_log_posterior - past_log_posterior):
                    l = next_l
                    past_log_posterior = next_log_posterior
            except:
                pass
                
        sampled_l = []
        for _ in range(self.mcmc_opts["mcmc_samples"]):
            # Adding try except block in case log_posterior sampling fails
            # May happen if l jumps to region outside og prior domain
            try:
                next_l = self.mcmc_opts["jump"](l)
                next_log_posterior = log_posterior(next_l)
                if np.log(np.random.randn()) < (next_log_posterior - past_log_posterior):
                    l = next_l
                    past_log_posterior = next_log_posterior
            except:
                pass
            sampled_l.append(l)

        return sampled_l


class BayesOpt:
    def __init__(self, data_generator, init_sample_size, max_steps, sigma_obs=None,
                 is_mcmc=False, mcmc_opts=None):
        # Initializing Bayesian optimization objects:
        # I need to have an object that generates data and specifies domain of optimization
        # max_steps refer to the maximum number of sampled points
        self.max_steps = max_steps
        self.data_generator = data_generator

        # Initializing seen observations and adding a couple of variables for later bookkeeping
        self.domain = self.data_generator.domain
        pick_x = np.random.choice(range(len(self.domain)), size=init_sample_size, replace=False)
        self.x = self.domain[pick_x]
        self.y = self.data_generator.sample(self.x)
        self.best_y = np.max(self.y)
        self.mu_posterior = None
        self.std_posterior = None

        # Initializing underlying GP
        self.gp = GP(self.x, self.y)
        self.sigma_obs = sigma_obs

        # Initializing MCMC properties (mcmc_properties is supposed to be an instance of MCMCProperties class)
        self.is_mcmc = is_mcmc
        self.mcmc_opts = mcmc_opts

    def add_obs(self, x, y):
        # Adding new observations
        # It is assumed x and y are passed as scalars
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)

    def determine_l(self):
        # This function returns kernel hyperparameters for current state of the system
        # It is either hyperparameters that optimize log-likelihood or
        # In case we have mcmc sampling it is the sample of posterior distribution of hyperparameters
        # The output of the function is in either case the array of elements (one element for max-likelihood estimator) 
        if not self.is_mcmc:
            # Getting maximum likelihood estimator (curently for [0, 1] interval)
            l = max(np.exp(np.linspace(np.log(0.01), np.log(1), 100)), 
                    key = lambda z: self.gp.log_likelihood(self.sigma_obs, z))
            return [l]
        if self.is_mcmc:
            l_sampler = MCMCSampler(lambda z: self.gp.log_likelihood(self.sigma_obs, z), self.mcmc_opts)
            return l_sampler.posterior_sample()

    def step(self):
        # The main function of BayesOpt class which performs one does a single optimization step
        # I estimate the kernel hyperparameters that best fit the data (either with mcmc or likelihood optimization)
        # Then I select the best point to sample (currently with EI acquisition function)
        # Then I sample the point and update my state

        # Sampling kernel hyperparameters
        sampled_l = self.determine_l()

        # Averaging GP posterior and EI over possible kernel hyperparameters
        # Note that as std is not quite an expectation, its averaging is a hack and not necessariy would give true std
        mu = np.zeros((len(self.domain),))
        std_1d = np.zeros((len(self.domain),))
        ei = np.zeros((len(self.domain),))
        for l in sampled_l:
            sampled_mu, sampled_std_1d = self.gp.gp_posterior(self.domain, self.sigma_obs, l, return_chol=False)

            z = (sampled_mu - self.best_y) / sampled_std_1d
            sampled_ei = sampled_std_1d * scipy.stats.norm.pdf(z) + z * sampled_std_1d * scipy.stats.norm.cdf(z)
            
            mu += sampled_mu
            std_1d += sampled_std_1d
            ei += sampled_ei

        # Sampling a new point
        new_x = self.domain[np.argmax(ei)]
        new_y = self.data_generator.sample(new_x)
        self.add_obs(new_x, new_y)
        self.gp.add_obs(new_x, new_y)
        self.best_y = max(new_y, self.best_y)
        
        self.mu_posterior = mu / len(sampled_l)
        self.std_posterior = std_1d / len(sampled_l)

    def run(self):
        # The function that runs whole optimizaion
        # For now it only does single steps
        # In the future some print and plot statements could be added
        for _ in range(self.max_steps):
            self.step()
