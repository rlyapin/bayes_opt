# This script contains implementation of Gaussian processes and its methods
# For now the functionality is limited to 1D target variables and Gaussian kernels

import numpy as np
import scipy as sp

def k_gaussian(x1, x2, l=0.1):
    # calculating gaussian kernel matrix
    # the output has shape (len(x2), len(x1))
    # entry at [i, j] position is given by k(x2[i], x1[j])
    # dealing with gaussian kernels so k(x, y) = e ^ ((x - y) ** 2 / 2 * l ** 2)
    # TODO: add custom K function as a parameter instead of hardcoding Gaussian kernel
    
    # gaussian kernel hyperparameters - adjusts the distance between points and variance
    scale_kernel = 1
    
    x1_matrix = np.tile(x1, len(x2)).reshape((len(x2), len(x1)))
    x2_matrix = np.tile(x2, len(x1)).reshape((len(x1), len(x2))).transpose()
    
    k_matrix = np.exp(-(x1_matrix - x2_matrix) ** 2 / (2 * l * l)) * scale_kernel ** 2
    
    return k_matrix


class GP:
    def __init__(self, init_x, init_y):
        # Initializing Gaussian Process:
        # self.x and self.y are supposed to hold all observations seen by GP
        self.x = init_x
        self.y = init_y
    
    def add_obs(self, x, y):
        # Adding new observations
        # It is assumed x and y are passed as scalars
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)

    def log_likelihood(self, sigma_obs, l):
        # The following function calculates the log-likelihood of observed data
        # wrt to prior distribution for GP, i.e. zero mean and sigma given by k_gaussian(l) + sigma_obs ** 2 * I
        
        # Under that model the log-likelihood is given by 
        # -0.5 * y' * sigma(-1) * y - 0.5 * n * log(2pi) - 0.5 * log|sigma|
        
        # To make sense of the code below note that we express log-likelihood through the cholesky decomposition of sigma
        # Then |sigma| = |chol| ** 2 (det of product is product of det)
        # |chol| = prod(chol_ii) (because cholesky matrix is lower triangular)
        # Thus, 0.5 * log|sigma| = sum(log(chol_ii))
        
        sigma = k_gaussian(self.x, self.x, l) + np.eye(len(self.x)) * sigma_obs ** 2
        chol = np.linalg.cholesky(sigma)
        
        # Calculating alpha = sigma(-1) * y (or solution to sigma * alpha = y) using cholesky matrix
        # (This trick is taken from sklearn implementation of GP)
        alpha = sp.linalg.cho_solve((chol, True), self.y).reshape((-1, 1))
        
        log_lik = -0.5 * np.dot(self.y.reshape(1, -1), alpha)
        log_lik -= 0.5 * len(self.x) * np.log(2 * np.pi)
        log_lik -= np.trace(np.log(np.absolute(chol)))
        
        return log_lik[0][0]

    def gp_posterior(self, x, sigma_obs, l, return_chol=False):
        # Calculating posterior for gaussian processes
        # I am specifically interested in posterior mean, std and cholesky matrix for postrior at sampled points (for nei)
        # it is assumed that observations have some additional gaussian noise
        
        # Important: the method cannot handle sigma_obs=0 if I want to predict for sample_x
        # Mostly numerical issues: with zero noise matrix to invert may not be positive-semidefinite
        
        # Separately calculating matrix used to calculate both mean and variance
        K = np.dot(k_gaussian(self.x, x, l),
                   np.linalg.inv(k_gaussian(self.x, self.x, l) + np.eye(len(self.x)) * sigma_obs ** 2)
                  )
        
        mu = np.dot(K, self.y)
        sigma = k_gaussian(x, x, l) - np.dot(K, k_gaussian(x, self.x, l))
        std_1d = np.sqrt([sigma[i, i] for i in range(len(mu))])

        if return_chol:
            noise = 1e-8
            while True:
                try:
                    chol = np.linalg.cholesky(sigma)
                    break
                except:
                    print "Problems with getting cholesky matrix, adding noise to main diagonal"
                    sigma += noise * np.eye(len(self.x))
            
            return mu.reshape(-1), std_1d.reshape(-1), chol            
        
        return mu.reshape(-1), std_1d.reshape(-1)
