
import numpy as np

# The file is supposed to hold the collection of 2D optimization problems
# All of them are taken from here and tweaked a little bit: https://www.sfu.ca/~ssurjano/optimization.html
# It is assumed all of them are defined over [0, 1] x [0, 1] range
# All functions are normalized to have a minimum of 0
# It is also assumed there is not much variance in function values (everything in ~[0, 2] range or so)


def ackley(x):
    x0 = 64 * x[0] - 32
    x1 = 64 * x[1] - 32

    a = 20
    b = 0.2
    c = 2 * np.pi

    exp_1_term = np.exp(-b * np.sqrt(0.5 * x0 ** 2 + 0.5 * x1 ** 2))
    exp_2_term = np.exp(0.5 * np.cos(c * x0) + 0.5 * np.cos(c * x1))

    val = 0.05 * (-a * exp_1_term - exp_2_term + a + np.exp(1))

    return val 


def bukin_n6(x):
    x0 = 10 * x[0] - 15
    x1 = 6 * x[1] - 3

    val = np.sqrt(np.abs(x1 - 0.01 * x0 ** 2)) + 0.0001 * np.abs(x0 + 10)

    return val 


def cross_in_tray(x):
    x0 = 4 * x[0] - 2
    x1 = 4 * x[1] - 2

    val = 2.06261 - 0.0001 * np.power(np.abs(np.sin(x0) * np.sin(x1) * np.exp(np.abs(100 - np.sqrt(x0 ** 2 + x1 ** 2) / np.pi))) + 1, 0.1)

    return val 


def drop_wave(x):
    x0 = 4 * x[0] - 2
    x1 = 4 * x[1] - 2

    norm = np.sqrt(x0 ** 2 + x1 ** 2)

    val = 1 - (1 + np.cos(12 * norm)) / (0.5 * norm ** 2 + 2)

    return val 


def eggholder(x):
    x0 = 1024 * x[0] - 512
    x1 = 1024 * x[1] - 512

    sin_1_term = np.sin(np.sqrt(np.abs(x1 + 0.5 * x0 + 47)))
    sin_2_term = np.sin(np.sqrt(np.abs(x0 - x1 - 47)))

    val = 0.001 * (959.6407 - (x1 + 47) * sin_1_term - x0 * sin_2_term)

    return val 


def griewank(x):
    x0 = 1200 * x[0] - 600
    x1 = 1200 * x[1] - 600

    val = 0.01 * (x0 ** 2 / 4000.0 + x1 ** 2 / 4000.0 - np.cos(x0) - np.cos(x1 / np.sqrt(2)) + 1)

    return val 


def holder(x):
    x0 = 20 * x[0] - 10
    x1 = 20 * x[1] - 10

    norm = np.sqrt(x0 ** 2 + x1 ** 2)
    exp_term = np.exp(np.abs(1 - norm / np.pi))

    val = 0.1 * (19.2085 - np.abs(np.sin(x0) * np.cos(x1) * exp_term))

    return val 


def levy(x):
    x0 = 20 * x[0] - 10
    x1 = 20 * x[1] - 10

    w0 = 1 + 0.25 * (x0 - 1)
    w1 = 1 + 0.25 * (x1 - 1)

    sin_1_term = (w0 - 1) ** 2 * (1 + 10 * np.sin(np.pi * w0 + 1) ** 2)
    sin_2_term = (w1 - 1) ** 2 * (1 + np.sin(2 * np.pi * w1) ** 2)

    val = 0.02 * (np.sin(np.pi * w0) ** 2 + sin_1_term + sin_2_term)

    return val 


def levy_n13(x):
    x0 = 20 * x[0] - 10
    x1 = 20 * x[1] - 10

    sin_1_term = (x0 - 1) ** 2 * (1 + np.sin(3 * np.pi * x1) ** 2)
    sin_2_term = (x1 - 1) ** 2 * (1 + np.sin(2 * np.pi * x1) ** 2)

    val = 0.005 * (np.sin(3 * np.pi * x0) ** 2 + sin_1_term + sin_2_term)

    return val 


def rastrigin(x):
    x0 = 10 * x[0] - 5
    x1 = 10 * x[1] - 5

    sin_1_term = (x0 - 1) ** 2 * (1 + np.sin(3 * np.pi * x1) ** 2)
    sin_2_term = (x1 - 1) ** 2 * (1 + np.sin(2 * np.pi * x1) ** 2)

    val = 0.02 * (20 + x0 ** 2 + x1 ** 2 - 10 * np.cos(2 * np.pi * x0) - 10 * np.cos(2 * np.pi * x1))

    return val 


def schaffer_n2(x):
    x0 = 200 * x[0] - 100
    x1 = 200 * x[1] - 100

    norm = np.sqrt(x0 ** 2 + x1 ** 2)
    sin_term = np.sin(x0 ** 2 - x1 ** 2) ** 2

    val = 0.5 + (sin_term - 0.5) / (np.abs(1 + 0.001 * norm ** 2) ** 2)

    return val 


def schaffer_n4(x):
    x0 = 200 * x[0] - 100
    x1 = 200 * x[1] - 100

    norm = np.sqrt(x0 ** 2 + x1 ** 2)
    cos_term = np.cos(np.sin(np.abs(x0 ** 2 - x1 ** 2))) ** 2

    val = 0.5 + (cos_term - 0.5) / (np.abs(1 + 0.001 * norm ** 2) ** 2) - 0.292579

    return val 


def schwefel(x):
    x0 = 1000 * x[0] - 500
    x1 = 1000 * x[1] - 500

    norm = np.sqrt(x0 ** 2 + x1 ** 2)
    cos_term = np.cos(np.sin(np.abs(x0 ** 2 - x1 ** 2)))

    val = 0.001 * (418.9829 * 2 - x0 * np.sin(np.sqrt(np.abs(x0))) - x1 * np.sin(np.sqrt(np.abs(x1)))) 

    return val 


def bohachevsky(x):
    x0 = 200 * x[0] - 100
    x1 = 200 * x[1] - 100

    val = 0.0001 * (x0 ** 2 + 2 * x1 ** 2 - 0.3 * np.cos(3 * np.pi * x0) - 0.4 * np.cos(4 * np.pi * x1) + 0.7) 

    return val 


def perm_beta(x):
    x0 = 4 * x[0] - 2
    x1 = 4 * x[1] - 2

    beta = 0

    val = 0.1 * ((1 + beta) * (x0 - 1) ** 2 + (1 + beta) * (x0 ** 2 - 1) ** 2 + (2 + beta) * (x1 - 0.5) ** 2 + (2 + beta) * (x1 ** 2 - 0.25) ** 2)

    return val 


def sphere(x):
    x0 = 10 * x[0] - 5
    x1 = 10 * x[1] - 5

    val = 0.05 * (x0 ** 2 + x1 ** 2)

    return val 


def power_sum(x):
    x0 = 4 * x[0] - 2
    x1 = 4 * x[1] - 2

    val = 0.25 * (x0 ** 2 + np.abs(x1) ** 3)

    return val 


def square_sum(x):
    x0 = 10 * x[0] - 5
    x1 = 10 * x[1] - 5

    val = 0.05 * (x0 ** 2 + 2 * x1 ** 2)

    return val 


def trid(x):
    x0 = 8 * x[0] - 4
    x1 = 8 * x[1] - 4

    val = 0.05 * (2 + (x0 - 1) ** 2 + (x1 - 1) ** 2 - x0 * x1)

    return val 


def matyas(x):
    x0 = 10 * x[0] - 5
    x1 = 10 * x[1] - 5

    val = 0.05 * (0.26 * x0 ** 2 + 0.26 * x1 ** 2 - 0.48 * x0 * x1)

    return val 


def mccormick(x):
    x0 = 5.5 * x[0] - 1.5
    x1 = 7 * x[1] - 3

    val = 0.1 * (1.9133 + np.sin(x0 + x1) + (x0 - x1) ** 2 - 1.5 * x0 + 2.5 * x1 + 1)

    return val 


def zakharov(x):
    x0 = 10 * x[0] - 5
    x1 = 10 * x[1] - 5

    val = 0.001 * (x0 ** 2 + x1 ** 2 + (0.5 * x0 + x1) ** 2 + (0.5 * x0 + x1) ** 4)

    return val 


def three_hump_camel(x):
    x0 = 10 * x[0] - 5
    x1 = 10 * x[1] - 5

    val = 0.001 * (2 * x0 ** 2 - 1.05 * x0 ** 4 + x0 ** 6 / 6.0 + x0 * x1 + x1 ** 2)

    return val 


def dixon_price(x):
    x0 = 10 * x[0] - 5
    x1 = 10 * x[1] - 5

    val = 0.0005 * ((x0 - 1) ** 2 + 2 * (2 * x1 ** 2 - x0) ** 2)

    return val 


def rosenbrock(x):
    x0 = 4 * x[0] - 2
    x1 = 4 * x[1] - 2

    val = 0.001 * (100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2)

    return val 


def easom(x):
    x0 = 200 * x[0] - 100
    x1 = 200 * x[1] - 100

    val = 1 - np.cos(x0) * np.cos(x1) * np.exp(- (x0 - np.pi) ** 2 - (x1 - np.pi) ** 2)

    return val 


def michalewicz(x):
    x0 = np.pi * x[0]
    x1 = np.pi * x[1]

    m = 10

    val = 1.8013 - np.sin(x0) * np.sin(x0 ** 2 / np.pi) ** (2 * m) - np.sin(x1) * np.sin(2 * x1 ** 2 / np.pi) ** (2 * m)

    return val 


def perm_d_beta(x):
    x0 = 4 * x[0] - 2
    x1 = 4 * x[1] - 2

    beta = 0

    val = 0.02 * (((1 + beta) * (x0 - 1)  + (2 + beta) * (0.5 * x1 - 1)) ** 2 + ((1 + beta) * (x0 ** 2 - 1)  + (4 + beta) * (0.25 * x1 ** 2 - 1)) ** 2)

    return val 


def styblinski_tang(x):
    x0 = 10 * x[0] - 5
    x1 = 10 * x[1] - 5

    val = 0.005 * (2 * 39.16599 + x0 ** 4 - 16 * x0 ** 2 + 5 * x0 + x1 ** 4 - 16 * x1 ** 2 + 5 * x1)

    return val 


def beale(x):
    x0 = 8 * x[0] - 4
    x1 = 8 * x[1] - 4

    val = 0.00005 * ((1.5 - x0 + x0 * x1) ** 2 + (2.25 - x0 + x0 * x1 ** 2) ** 2 + (2.625 - x0 + x0 * x1 ** 3) ** 2)

    return val 

