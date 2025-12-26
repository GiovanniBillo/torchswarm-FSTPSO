import torch
import numpy as np

from torchswarm.data.data_utils import get_bounds 
NPZ_PATH = 'torchswarm/data/esa_oc_412_valid_samples.npz'
# inverse_chlor_a_solution, inverse_chlor_a_problem_bounds = get_esa_metadata(wavelength=lam) 
PROBLEM_BOUNDS = {
        "LotkaVolterra": (0.0001, 0.2), ## how can one bound be if these represent growth rate, death rate and stuff like that??
        "Reflectance":  get_bounds(NPZ_PATH, columns=['chlor_a', 'rrs_412', 'atot_412']) # returns a dictionary of tuples

    # "Reflectance":      ['bounds for backscattering'], ['bounds for absorption'] # if any
        }
PROBLEM_BOUNDS["Reflectance"] = {
    "bbp":  (1e-5, 0.5),
    "atot": (1e-4, 20.0),
    "g0":   (0.05, 0.15),
    "g1":   (0.05, 0.20),
    "a1":   (-2.0, 2.0),
    "a2":   (-6.0, 6.0),
    "a3":   (-6.0, 6.0),
    "a4":   (-6.0, 6.0),
    "a5":   (-6.0, 6.0),
}

TRUE_PARAMS={"LotkaVolterra": torch.Tensor([[0.1, 0.02],[0.01, 0.1]]).unsqueeze(0), 
             "Reflectance": torch.Tensor([])  
             }
# initial_conditions = torch.Tensor([30, 10]).unsqueeze(0)
initial_conditions = torch.Tensor([30, 10])

# solution = solve_lotka_volterra(TRUE_PARAMS['LotkaVolterra'], initial_conditions, t) 

lv_solution_placeholder = None
esa_data = np.load(NPZ_PATH) 
# data = np.load("esa_oc_412_valid_samples.npz")

rrs_all   = torch.tensor(esa_data["rrs_412"],   dtype=torch.float32)
atot_all  = torch.tensor(esa_data["atot_412"],  dtype=torch.float32)
chlor_all = torch.tensor(esa_data["chlor_a"],   dtype=torch.float32)

M = chlor_all.shape[0]
print("Total samples:", M)

B = 20000  # start with 10k–50k

idx = torch.randperm(M)[:B]

rrs_batch   = rrs_all[idx]
atot_batch  = atot_all[idx]
chlor_batch = chlor_all[idx]

solution_inverse_chlor_a = torch.from_numpy(esa_data['chlor_a']) 

REAL_SOLUTIONS = {"LotkaVolterra": lv_solution_placeholder, 
                  "Reflectance": solution_inverse_chlor_a
                  } 

if __name__=="__main__":
    print(REAL_SOLUTIONS)
    print(PROBLEM_BOUNDS)
