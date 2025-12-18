import torch
import numpy as np

from torchswarm.data.data_utils import get_bounds 
NPZ_PATH = 'torchswarm/data/esa_oc_412_valid_samples.npz'
# inverse_chlor_a_solution, inverse_chlor_a_problem_bounds = get_esa_metadata(wavelength=lam) 
PROBLEM_BOUNDS = {
        "LotkaVolterra": (0.0001, 0.2), ## how can one bound be if these represent growth rate, death rate and stuff like that??
        "Reflectance":  get_bounds(NPZ_PATH, columns=['chlor_a', 'rrs_412', 'atot_412'])

    # "Reflectance":      ['bounds for backscattering'], ['bounds for absorption'] # if any
        }

TRUE_PARAMS={"LotkaVolterra": torch.Tensor([[0.1, 0.02],[0.01, 0.1]]).unsqueeze(0), 
             "Reflectance": torch.Tensor([])  
             }
# initial_conditions = torch.Tensor([30, 10]).unsqueeze(0)
initial_conditions = torch.Tensor([30, 10])

solution = solve_lotka_volterra(TRUE_PARAMS['LotkaVolterra'], initial_conditions, t) 

esa_data = np.load(NPZ_PATH) 
solution_inverse_chlor_a = data['chlor_a'] 

REAL_SOLUTIONS = {"LotkaVolterra": solution, 
                  "Reflectance": solution_inverse_chlor_a
                  } 
if __name__=="__main__":
    print(REAL_SOLUTIONS)
    print(PROBLEM_BOUNDS)
