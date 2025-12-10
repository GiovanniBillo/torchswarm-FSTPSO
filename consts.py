import math
import numpy as np

from scipy.integrate import odeint

RESULTS_DIR = 'results'
TRUE_OPTIMA = {
    "Ackley": 0.0,
    "Sphere": 0.0,
    "Rastrigin": 0.0,
    "Alpine": 0.0,
    "Griewank": 0.0,
    "Michalewicz": -4.687658,  # approx optimal for dim=5, m=10
    "Plateau": 0.0,           # for your modified Plateau (floor+0.5)^2 version
    "Quintic": 0.0,
    "Rosenbrock": 0.0,
    "Vincent": -5.0,          # sum sin(10 log x) over 5 dims, each min = -1
    "XinSheYang": -1.0,       # global optimum approximately -1
    "Bohachevsky": 0.0,
    "Eggholder": -959.6407,   # global known minimum (512, 404)
    "Shubert": -186.7309,     # for 2D version
    "LotkaVolterra": 0.0
}


BOUNDS = {
    "Ackley":       (-30, 30),
    "Sphere":      (-5.12, 5.12),
    "Alpine":       (-10, 10),
    "Bohachevsky":  (-15, 15),
    "Griewank":     (-600, 600),
    "Michalewicz":  (0, math.pi),
    "Plateau":      (-5.12, 5.12),
    "Quintic":      (-10, 10),
    "Rastrigin":    (-5.12, 5.12),
    "Rosenbrock":   (-5, 10),
    "Shubert":      (-10, 10),
    "Vincent":      (0.25, 10),
    "XinSheYang":   (-2*math.pi, 2*math.pi),
    "Eggholder":      (-512, 512),
}


