
import torch
import math
from debug_utils import _vprint
from consts import BOUNDS
import warnings 

class Function:
    def __init__(self):
        self.dimensions = None
        self.bounds = None
    def evaluate(self, pos):
        raise NotImplementedError
    def get_bounds(self, pos, how=["tuple", "array"]):
        if how == "tuple":
            if isinstance(self.bounds, tuple):
                return self.bounds
            if isinstance(self.bounds, numpy.ndarray):
                warnings.warn("Warning: attempting to return a tuple when bounds were provided as array. Defaulting to array...")
                return torch.Tensor(([self.bounds[0], self.bounds[1]]*self.dimensions)) 
        elif how == "array":
            if isinstance(self.bounds, torch.Tensor):
                return self.bounds
            if isinstance(self.bounds, tuple):
                warnings.warn("Warning: attempting to return an array when bounds were provided as array. Defaulting to array...")

                return torch.Tensor(([self.bounds[0], self.bounds[1]]*self.dimensions)) 
            

def _ensure_2d(pos):
    """Ensure pos is [batch, dim]."""
    if pos.dim() == 1:
        pos = pos.unsqueeze(0)
    return pos


## Good reference for typical benchmark optimization functions: https://www.sfu.ca/~ssurjano/ackley.html or whatever function one might need
# ---------------------------------------------------------
#  ACKLEY
# ---------------------------------------------------------

## tensorized/parallel version
class Ackley(Function):
    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]
    def evaluate(self, swarm):
        # swarm: (N, D, C)  (C may be 1)
        # reduce over D and C -> output (N,)
        dims = tuple(range(1, swarm.ndim))  # (1,2) for (N,D,C)

        mean_sq = torch.mean(swarm**2, dim=dims)                      # (N,)
        mean_cos = torch.mean(torch.cos(2 * torch.pi * swarm), dim=dims)  # (N,)

        term1 = -20 * torch.exp(-0.2 * torch.sqrt(mean_sq))
        term2 = -torch.exp(mean_cos)
        return term1 + term2 + 20 + torch.e                           # (N,)

## united version?
# class Ackley(Function):
#     def __init__(self):
#         self.bounds = BOUNDS[self.__class__.__name__]

#     def evaluate(self, x):
#         """
#         x: (N, D) or (N, D, C)
#         returns: (N,)
#         """
#         if x.ndim == 2:
#             dims = (1,)
#         elif x.ndim == 3:
#             dims = (1, 2)
#         else:
#             raise ValueError(f"Invalid input shape {x.shape}")

#         mean_sq = torch.mean(x**2, dim=dims)
#         mean_cos = torch.mean(torch.cos(2 * torch.pi * x), dim=dims)

#         term1 = -20 * torch.exp(-0.2 * torch.sqrt(mean_sq))
#         term2 = -torch.exp(mean_cos)

#         return term1 + term2 + 20 + torch.e
#     def eval_single(self, x):
#         # x: (D,) or (D,C)
#         return self.evaluate(x.unsqueeze(0))[0].item()

