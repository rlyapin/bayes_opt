import numpy as np
import itertools
from sample_functions_2d import *

class DataGenerator2D:
    # Main class to hold signal generating object
    # It is supposed to hold objects from here: https://www.sfu.ca/~ssurjano/optimization.html
    # It is assumed the function will hold its minimum of 0 and maximum of 1 
    # The domain of the function would be [0, 1] x [0, 1]
    def __init__(self, fun, x_opt, desc):
        # Initializing signals:
        # fun: a lambda 2D function
        # x_opt: a location of minimum
        # desc: a type of optimization problem (i.e. namy local optima, bowl shaped and so on)
        self.fun = fun
        self.x_opt = x_opt
        self.desc = desc
 
    def sample(self, x):
        return np.apply_along_axis(self.fun, 1, x)


def f(x, dim, max_1d_basis, seed):
    # The function generating a random smooth function
        
    # Generating basis for continuous functions in 1D (trogonometric ones are fine by Stone-Weierstrass)
    basis_f = [lambda x: 1]
    np.random.seed(seed)
    for i in range(1, max_1d_basis):
        basis_f.append(lambda x: np.sin(i * np.pi * (x + np.random.rand())))
        basis_f.append(lambda x: np.cos(i * np.pi * (x + np.random.rand())))

    
    f_val = 0.0

    # Adding a bit of polynomials to break symmetries (I am assuming 2D here)
    np.random.seed(seed)
    f_val += np.random.randn() * x[0]
    f_val += np.random.randn() * x[1]
    f_val += np.random.randn() * x[0] ** 2
    f_val += np.random.randn() * x[1] ** 2
    f_val += np.random.randn() * x[0] * x[1]
    
    # Constructing basis in arbitarry dimensional space by tensor products
    # Fixing seed to have consistent function evaluations
    np.random.seed(seed)
    for basis_pick in itertools.product(range(len(basis_f)), repeat=dim):
        basis_eval_1d = map(lambda i: basis_f[basis_pick[i]](x[i]), range(dim))
        f_val += np.random.randn() * reduce(lambda y,z : y * z, basis_eval_1d) 
        
    normalizer = ((len(basis_f) + 1) / 2) ** (0.5 * dim)
        
    return f_val / normalizer


def produce_random_data_generator(max_1d_basis, seed):
    # The function to produce a random DataGenerator2D object
    # The core of object lies in f() function
    # After I fetch it I just need to record minimum and move function in ~ [0, 2] range

    raw_f = lambda x: f(x, 2, max_1d_basis, seed)

    # Finding minimum
    x1 = np.linspace(0, 1, 100)
    x2 = np.linspace(0, 1, 100)

    X1, X2 = np.meshgrid(x1,x2)
    grid_flat = np.vstack([X1.ravel(), X2.ravel()]).transpose()
    z = np.apply_along_axis(raw_f, 1, grid_flat)

    min_f = np.min(z)
    x_opt = grid_flat[np.argmin(z)]

    # Adding description to the function
    desc = "Custom function: Basis-" + str(max_1d_basis) + "-Seed-" + str(seed)

    return DataGenerator2D(lambda x: 0.5 * (raw_f(x) - min_f), x_opt, desc)


# A helper array to hold target signals
# Be wary there is implicit links between normalization in functions are provided optimums
TARGET_SIGNALS = [DataGenerator2D(ackley, np.array([0.5, 0.5]), "Many local minima: Ackley"),
                  DataGenerator2D(bukin_n6, np.array([0.5, 0.66666]), "Many local minima: Bukin-N6"),
                  # DataGenerator2D(cross_in_tray, np.array([[0.162725, 0.162725], [0.162725, 1 - 0.162725], [1 - 0.162725, 0.162725], [1 - 0.162725, 1 - 0.162725]]), "Many local minima: Cross-In-Tray"),
                  DataGenerator2D(drop_wave, np.array([0.5, 0.5]), "Many local minima: Drop-Wave"),
                  DataGenerator2D(eggholder, np.array([1.0, 0.89475]), "Many local minima: Eggholder"),
                  DataGenerator2D(griewank, np.array([0.5, 0.5]), "Many local minima: Griewank"),
                  # DataGenerator2D(holder, np.array([[0.0972, 0.0168], [0.0972, 1 - 0.0168], [1 - 0.0972, 0.0168], [1 - 0.0972, 1 - 0.0168]]), "Many local minima: Holder"),
                  DataGenerator2D(levy, np.array([0.55, 0.55]), "Many local minima: Levy"),
                  DataGenerator2D(levy_n13, np.array([0.55, 0.55]), "Many local minima: Levy-N13"),
                  DataGenerator2D(rastrigin, np.array([0.5, 0.5]), "Many local minima: Rastrigin"),
                  DataGenerator2D(schaffer_n2, np.array([0.5, 0.5]), "Many local minima: Schaffer-N2"),
                  DataGenerator2D(schaffer_n4, np.array([0.5, 0.50626]), "Many local minima: Schaffer-N4"),
                  DataGenerator2D(schwefel, np.array([0.92096, 0.92096]), "Many local minima: Schwefel"),
                  DataGenerator2D(bohachevsky, np.array([0.5, 0.5]), "Bowled shaped: Bohachevsky"),
                  DataGenerator2D(perm_beta, np.array([0.75, 0.625]), "Bowled shaped: Perm-beta"),
                  DataGenerator2D(sphere, np.array([0.5, 0.5]), "Bowled shaped: Sphere"),
                  DataGenerator2D(power_sum, np.array([0.5, 0.5]), "Bowled shaped: Power-sum"),
                  DataGenerator2D(square_sum, np.array([0.5, 0.5]), "Bowled shaped: Square-sum"),
                  DataGenerator2D(trid, np.array([0.75, 0.75]), "Bowled shaped: Trid"),
                  DataGenerator2D(matyas, np.array([0.5, 0.5]), "Plate shaped: Matyas"),
                  DataGenerator2D(mccormick, np.array([0.173238, 0.207544]), "Plate shaped: McCormick"),
                  DataGenerator2D(zakharov, np.array([0.5, 0.5]), "Plate shaped: Zakharov"),
                  DataGenerator2D(three_hump_camel, np.array([0.5, 0.5]), "Valley shaped: Three-Hump-Camel"),
                  DataGenerator2D(dixon_price, np.array([0.6, 0.57071]), "Valley shaped: Dixon-Price"),
                  DataGenerator2D(rosenbrock, np.array([0.75, 0.75]), "Valley shaped: Rosenbrock"),
                  DataGenerator2D(easom, np.array([0.5157, 0.5157]), "Drops: Easom"),
                  DataGenerator2D(michalewicz, np.array([0.70028, 0.49974]), "Drops: Michalewicz"),
                  DataGenerator2D(perm_d_beta, np.array([0.75, 1.0]), "Other: Perm-d-beta"),
                  DataGenerator2D(styblinski_tang, np.array([0.2096, 0.2096]), "Other: Styblinski-Tang"),
                  DataGenerator2D(beale, np.array([0.875, 0.5625]), "Other: Beale")
                 ]

TARGET_SIGNALS += [produce_random_data_generator(2, i) for i in range(1, 4)]
TARGET_SIGNALS += [produce_random_data_generator(3, i) for i in range(1, 4)]
TARGET_SIGNALS += [produce_random_data_generator(4, i) for i in range(1, 4)]
TARGET_SIGNALS += [produce_random_data_generator(5, i) for i in range(1, 4)]






