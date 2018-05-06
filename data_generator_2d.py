import numpy as np

class DataGenerator2D:
	# Main class to hold signal generating object
	# It is supposed to hold objects from here: https://www.sfu.ca/~ssurjano/optimization.html
	# It is assumed the function will hold its minimum of 0 and maximum of 1 
	# The domain of the function would be [0, 1] x [0, 1]
    def __init__(self, fun, name, fun_type):
    	# Initializing signals:
    	# fun: a lambda 2D function
    	# name: a name of target function
    	# fun_type: a type of optimization problem (i.e. namy local optima, bowl shaped and so on)
        self.fun = fun
        self.name = name
        self.fun_type = fun_type
 
    def sample(self, x):
    	return np.apply_along_axis(self.fun, 1, x)


# A helper array to hold target signals
TARGET_SIGNALS = [DataGenerator2D(lambda x: 0.05 * (-20 * np.exp(-0.2 * np.sqrt(0.5 * (64 * x[0] - 32) ** 2 + 0.5 * (64 * x[1] - 32) ** 2)) - np.exp(0.5 * np.cos(2 * np.pi * (64 * x[0] - 32)) + 0.5 * np.cos(2 * np.pi * (64 * x[1] - 32))) + 20 + np.exp(1)), "Ackley", "many_local_minima")
				 ]