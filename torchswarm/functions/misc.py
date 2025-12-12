from torchswarm.functions.benchmarks import Function

from scipy.integrate import odeint
import numpy as np

PROBLEM_BOUNDS = {
        "LotkaVolterra": (0.0001, 0.2) ## how can one bound be if these represent growth rate, death rate and stuff like that??
    # "Reflectance":       (-30, 30),

    # "Reflectance":      ['bounds for backscattering'], ['bounds for absorption'] # if any
        }
TRUE_PARAMS={"LotkaVolterra": np.array([[0.1, 0.02],[0.01, 0.1]]), 
                # "Reflectance":      ['bounds for backscattering'], ['bounds for absorption'] # if any
             }


def lotka_volterra(state, t, params):

    r = params[0][0]
    a = params[0][1]
    b = params[1][0]
    z = params[1][1]

    X = state[0]
    Y = state[1]

    dxdt = r*X - a*X*Y
    dydt = b*X*Y - z*Y

    return [dxdt, dydt]

t = np.linspace(0, 100, 100)
initial_conditions = np.array([30, 10])
solution = odeint(lotka_volterra, initial_conditions, t, args=(TRUE_PARAMS["LotkaVolterra"],))

REAL_SOLUTIONS = {"LotkaVolterra": solution} 
    # "Reflectance":       (-30, 30),}


class LotkaVolterra(Function):
    def __init__(self, ground_truth=None, real_params=None):

        self.real_params = real_params
        self.ground_truth = ground_truth
        self.name = self.__class__.__name__
        self.bounds = PROBLEM_BOUNDS[self.__class__.__name__]
        self.real_params = real_params 
        print("my name is:", self.name)
        if self.ground_truth:
            self.ground_truth = ground_truth
        else:
            self.ground_truth = REAL_SOLUTIONS[self.name]

        print("initial_condition:", self.ground_truth[0])
        if self.real_params:
            self.real_params = real_params
        else:
            self.real_params = TRUE_PARAMS[self.name]

        self.real_params = real_params

    def evaluate(self, pos):
        t = np.linspace(0, 100, 100)
        initial_conditions = self.ground_truth[0]
        # print("position passed to LotkaVolterra:", pos)  

        assert pos.shape == TRUE_PARAMS["LotkaVolterra"].shape, f"AssertionError: pos should have size {TRUE_PARAMS["LotkaVolterra"].shape} but has shape {pos.shape}" 
        try:
            sol = odeint(lotka_volterra, initial_conditions, t, args=(pos,))
        except Exception as e:
            print(f"ODE solver failure inside {self.name}, params={pos}: {e}")
            return 1e12

        mse = np.mean((sol - self.ground_truth) ** 2)
        # print(f" -> {self.name} MSE = {mse}")
        return mse

class Reflectance(Function):
    def __init__(self):
        self.bounds = PROBLEM_BOUNDS[self.__class__.__name__]
    def evaluate(self, pos):
        
        gt = "Something you retrieve from the dataset itself" # i.e idk, initialize a particle swarm of the same size of the observations that were taken everyday and then compare them in order 
        g_0 = 0.089
        g_1 = 0.1245

        bbp = pos[0]
        atot = pos[1]
        first = g_0*(bbp/(bbp + atot)) 
        second = g_1*(bbp/(bbp + atot))**2
        pred = first + second 

        out = pred - gt 
        return out.squeeze()

