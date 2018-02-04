# The following function is supposed to test the impact of using MCMC samples for kernel hyperparameters

import math
import numpy as np
from bayes_opt import BayesOpt

# The base script to generate data - right now it is the sum of Fourier function scaled by random numbers
class DataGenerator:
    def __init__(self, n_fourier, sigma_obs):
        self.domain = np.linspace(0, 1, 1000)
        self.true_y = np.random.randn(1) * np.ones(1000)
        for i in range(1, n_fourier + 1):
            self.true_y += np.random.randn(1) * np.sin(i * math.pi * self.domain)
            self.true_y += np.random.randn(1) * np.cos(i * math.pi * self.domain)
        self.true_best_y = np.max(self.true_y)
        self.sigma_obs = sigma_obs
        self.true_y_dict = {x: y for (x, y) in zip(self.domain, self.true_y)}

    def sample(self, x):
        # Handling cases when x is both scalar and numpy array
        if type(x) == np.ndarray:
            return np.array(map(lambda z: self.true_y_dict[z], x)) + self.sigma_obs * np.random.randn(x.shape[0])
        else:
            return self.true_y_dict[x] + self.sigma_obs * np.random.randn()


N_FOURIER_RANGE = range(3, 11)
SIGMA_OBS_RANGE = np.linspace(0.01, 1, 100)
N_STEPS_RANGE = range(50, 200)
N_INIT_RANGE = range(1, 6)
MCMC_RANGE = range(10, 100)

N_SIM = 1

for _ in range(N_SIM):
    n_fourier = np.random.choice(N_FOURIER_RANGE)
    sigma_obs = np.random.choice(SIGMA_OBS_RANGE)
    init_sample_size = np.random.choice(N_INIT_RANGE)
    max_steps = np.random.choice(N_STEPS_RANGE)
    MCMC_OPTS = {"prior": lambda l: int(l > 0 and l < 1),
                 "icdf": lambda l: l,
                 "jump": lambda l: l + 0.05 * np.random.randn(),
                 "burn_period": 10000,
                 "mcmc_samples": np.random.choice(MCMC_RANGE)}
    data_gen = DataGenerator(n_fourier, sigma_obs)

    opt_engine = BayesOpt(data_gen, init_sample_size, max_steps, sigma_obs, 
        is_mcmc=True, mcmc_opts=MCMC_OPTS)
    opt_engine.run()
    avg_regret = data_gen.true_best_y - np.sum(opt_engine.y) / max_steps
    print avg_regret

    opt_engine = BayesOpt(data_gen, init_sample_size, max_steps, sigma_obs, 
        is_mcmc=False,  mcmc_opts=None)
    opt_engine.run()
    avg_regret = data_gen.true_best_y - np.sum(opt_engine.y) / max_steps
    print avg_regret



