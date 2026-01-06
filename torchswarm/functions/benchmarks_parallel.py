
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

# ---------------------------------------------------------
# helpers
# ---------------------------------------------------------

def _reduce_dims(swarm: torch.Tensor):
    """Return reduction dims for (N,D) or (N,D,C) -> reduce all but N."""
    assert swarm.ndim in (2, 3), f"Expected swarm ndim 2 or 3, got {swarm.shape}"
    return tuple(range(1, swarm.ndim))

def _collapse_class(x: torch.Tensor):
    """
    For 2D-only funcs: x may be (N,C) or (N,).
    Collapse C by mean -> (N,)
    """
    if x.ndim == 2:
        return x.mean(dim=-1)
    return x

# ---------------------------------------------------------
#  SPHERE
# ---------------------------------------------------------
class Sphere:
    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]

    def evaluate(self, swarm):
        dims = _reduce_dims(swarm)
        return torch.sum(swarm ** 2, dim=dims)  # (N,)

# ---------------------------------------------------------
#  RASTRIGIN
# ---------------------------------------------------------
class Rastrigin:
    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]

    def evaluate(self, swarm):
        dims = _reduce_dims(swarm)
        # D is swarm.shape[1] even if (N,D,C)
        D = swarm.shape[1]
        return 10.0 * D + torch.sum(swarm**2 - 10.0 * torch.cos(2 * torch.pi * swarm), dim=dims)  # (N,)

# ---------------------------------------------------------
#  ALPINE
# ---------------------------------------------------------
class Alpine:
    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]

    def evaluate(self, swarm):
        dims = _reduce_dims(swarm)
        return torch.sum(torch.abs(swarm * torch.sin(swarm) + 0.1 * swarm), dim=dims)  # (N,)

# ---------------------------------------------------------
#  BOHACHEVSKY (2D)
# ---------------------------------------------------------
class Bohachevsky:
    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]

    def evaluate(self, swarm):
        assert swarm.ndim in (2, 3), f"Invalid swarm shape {swarm.shape}"
        assert swarm.shape[1] == 2, f"Bohachevsky is 2D. Got D={swarm.shape[1]}"

        x = swarm[:, 0]  # (N,) or (N,C)
        y = swarm[:, 1]  # (N,) or (N,C)

        x = _collapse_class(x)
        y = _collapse_class(y)

        return x**2 + 2*y**2 - 0.3*torch.cos(3*torch.pi*x) - 0.4*torch.cos(4*torch.pi*y) + 0.7  # (N,)
# ---------------------------------------------------------
#  ROSENBROCK
#  f(x) = sum_{i=1..D-1} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
# ---------------------------------------------------------
class Rosenbrock:
    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]

    def evaluate(self, swarm: torch.Tensor) -> torch.Tensor:
        assert swarm.ndim in (2, 3), f"Invalid swarm shape {swarm.shape}"
        assert swarm.shape[1] >= 2, f"Rosenbrock needs D>=2, got D={swarm.shape[1]}"

        if swarm.ndim == 2:
            # (N,D)
            x = swarm[:, :-1]   # (N,D-1)
            y = swarm[:, 1:]    # (N,D-1)
            return torch.sum(100.0 * (y - x**2)**2 + (1.0 - x)**2, dim=1)  # (N,)

        # (N,D,C)
        x = swarm[:, :-1, :]   # (N,D-1,C)
        y = swarm[:, 1:, :]    # (N,D-1,C)
        return torch.sum(100.0 * (y - x**2)**2 + (1.0 - x)**2, dim=(1, 2))  # (N,)

class Griewank(Function):
    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]

    def evaluate(self, swarm):
        """
        swarm: (N, D, C)  (C may be 1)
        returns: (N,)
        Griewank: 1 + (1/4000) * sum(x_i^2) - prod(cos(x_i / sqrt(i)))
        where i = 1..K, K = D*C
        """
        assert isinstance(swarm, torch.Tensor), f"swarm must be Tensor, got {type(swarm)}"
        assert swarm.ndim in (2, 3), f"swarm must be (N,D) or (N,D,C), got {swarm.shape}"

        # Ensure (N, D, C)
        if swarm.ndim == 2:
            swarm = swarm.unsqueeze(-1)  # (N,D,1)

        N, D, C = swarm.shape
        K = D * C

        x = swarm.reshape(N, K)  # (N,K)

        sum_term = torch.sum(x ** 2, dim=1) / 4000.0  # (N,)

        i = torch.arange(1, K + 1, device=x.device, dtype=x.dtype)  # (K,)
        denom = torch.sqrt(i)  # (K,)

        prod_term = torch.prod(torch.cos(x / denom), dim=1)  # (N,)

        return sum_term - prod_term + 1.0


# ---------------------------------------------------------
#  MICHALEWICZ (k=10)
# ---------------------------------------------------------
class Michalewicz:
    def __init__(self, k=10):
        self.k = k
        self.bounds = BOUNDS[self.__class__.__name__]

    def evaluate(self, swarm):
        assert swarm.ndim in (2, 3), f"Invalid swarm shape {swarm.shape}"
        D = swarm.shape[1]
        device = swarm.device
        dtype = swarm.dtype

        if swarm.ndim == 2:
            # (N,D)
            i = torch.arange(1, D + 1, device=device, dtype=dtype)[None, :]  # (1,D)
            inner = torch.sin(i * swarm**2 / math.pi)
            return -torch.sum(torch.sin(swarm) * (inner ** (2 * self.k)), dim=1)  # (N,)
        else:
            # (N,D,C)
            C = swarm.shape[2]
            i = torch.arange(1, D + 1, device=device, dtype=dtype)[None, :, None]  # (1,D,1)
            inner = torch.sin(i * swarm**2 / math.pi)
            return -torch.sum(torch.sin(swarm) * (inner ** (2 * self.k)), dim=(1, 2))  # (N,)

# ---------------------------------------------------------
#  PLATEAU (your modified floor+0.5)^2 version)
#  NOTE: your TRUE_OPTIMA assumes 0.0; this matches sum((floor(x)+0.5)^2)
# ---------------------------------------------------------
class Plateau:
    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]

    def evaluate(self, swarm):
        dims = _reduce_dims(swarm)
        return torch.sum((torch.floor(swarm) + 0.5) ** 2, dim=dims)  # (N,)

# ---------------------------------------------------------
#  SHUBERT (2D)
#  Standard 2D Shubert: f(x,y)= (sum_{i=1..5} i cos((i+1)x+i)) * (sum_{i=1..5} i cos((i+1)y+i))
# ---------------------------------------------------------
class Shubert:
    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]

    def evaluate(self, swarm):
        assert swarm.ndim in (2, 3), f"Invalid swarm shape {swarm.shape}"
        assert swarm.shape[1] == 2, f"Shubert here is 2D. Got D={swarm.shape[1]}"
        device = swarm.device
        dtype = swarm.dtype

        x = _collapse_class(swarm[:, 0])  # (N,)
        y = _collapse_class(swarm[:, 1])  # (N,)

        i = torch.arange(1, 6, device=device, dtype=dtype)  # (5,)
        # (N,5)
        sx = torch.sum(i[None, :] * torch.cos((i[None, :] + 1) * x[:, None] + i[None, :]), dim=1)
        sy = torch.sum(i[None, :] * torch.cos((i[None, :] + 1) * y[:, None] + i[None, :]), dim=1)
        return sx * sy  # (N,)

# ---------------------------------------------------------
#  VINCENT
#  f(x)= -sum sin(10 log x) (often min ~ -D). Your TRUE_OPTIMA has -5 for D=5.
#  Your serial version returns sum(sin(...)); to be consistent with TRUE_OPTIMA(-5), we return -sum(...)
# ---------------------------------------------------------
class Vincent:
    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]

    def evaluate(self, swarm):
        dims = _reduce_dims(swarm)
        # domain x > 0; clamp to avoid log issues
        x = torch.clamp(swarm, min=1e-12)
        return -torch.sum(torch.sin(10.0 * torch.log(x)), dim=dims)  # (N,)

# ---------------------------------------------------------
#  XIN-SHE YANG (one common form)
#  Your serial version: sum(abs(x)) * exp(sum(sin(x^2))) - 1
# ---------------------------------------------------------
class XinSheYang:
    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]

    def evaluate(self, swarm):
        dims = _reduce_dims(swarm)
        sum_abs = torch.sum(torch.abs(swarm), dim=dims)         # (N,)
        sum_sin = torch.sum(torch.sin(swarm**2), dim=dims)      # (N,)
        return sum_abs * torch.exp(sum_sin) - 1.0               # (N,)

# ---------------------------------------------------------
#  EGGHOLDER (2D)
#  Standard definition: f(x,y)= -(y+47) sin(sqrt(|x/2+y+47|)) - x sin(sqrt(|x-(y+47)|))
# ---------------------------------------------------------
class Eggholder:
    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]

    def evaluate(self, swarm):
        assert swarm.ndim in (2, 3), f"Invalid swarm shape {swarm.shape}"
        assert swarm.shape[1] == 2, f"Eggholder is 2D. Got D={swarm.shape[1]}"

        x = _collapse_class(swarm[:, 0])  # (N,)
        y = _collapse_class(swarm[:, 1])  # (N,)

        term1 = -(y + 47.0) * torch.sin(torch.sqrt(torch.abs(x / 2.0 + (y + 47.0))))
        term2 = -x * torch.sin(torch.sqrt(torch.abs(x - (y + 47.0))))
        return term1 + term2  # (N,)

