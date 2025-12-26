import numpy as np
import torch
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
