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

class Ackley(Function):
    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]
    def evaluate(self, pos):
        # print("SHAPE OF POS AT INPUT:", pos.shape)
        M = pos.shape[0] # this was [1]. wrong! this way only 1 dimension is considered
        # print("selected shape:", M)
        # print("original:", pos)
        # print("differences in sum wrt dimensions (dim=1):", torch.sum(pos**2, dim=1))
        # print("differences in sum wrt dimensions (dim=0)):", torch.sum(pos**2, dim=0))

        term1 = torch.sqrt(torch.sum(pos**2) / M)
        term2 = torch.sum(torch.cos(2 * torch.pi * pos)) / M

        out = -20 * torch.exp(-0.2 * term1) - torch.exp(term2) + 20 + torch.e
        return out.squeeze()


# ---------------------------------------------------------
#  SPHERE
# ---------------------------------------------------------
class Sphere(Function):
    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]
    def evaluate(self, pos):
        pos = _ensure_2d(pos)

        # print("original:", pos)
        # print("differences in sum wrt dimensions (dim=1):", torch.sum(pos, dim=1))
        # print("differences in sum wrt dimensions (dim=0)):", torch.sum(pos, dim=0))
        return torch.sum(pos**2)


# ---------------------------------------------------------
#  RASTRIGIN
# ---------------------------------------------------------
class Rastrigin(Function):
    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]
    def evaluate(self, pos):
        pos = _ensure_2d(pos)
        M = pos.shape[0]

        # print("selected shape:", M)
        # print("original:", pos)
        # print("differences in sum wrt dimensions (dim=1):", torch.sum(pos, dim=1))
        # print("differences in sum wrt dimensions (dim=0)):", torch.sum(pos, dim=0))

        return 10 * M + torch.sum(pos**2 - 10 * torch.cos(2 * torch.pi * pos))


# ---------------------------------------------------------
#  ALPINE
# ---------------------------------------------------------
class Alpine(Function):
    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]
    def evaluate(self, pos):
        pos = _ensure_2d(pos)
        return torch.sum(torch.abs(pos * torch.sin(pos) + 0.1 * pos))


# ---------------------------------------------------------
#  EGGHOLDER
# ---------------------------------------------------------
class Eggholder(Function):
    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]
    def evaluate(self, pos):
        pos = _ensure_2d(pos)
        # defined only for 2D
        x = pos[0,0]
        y = pos[1,0]
        term1 = -(y + 47) * torch.sin(torch.sqrt(torch.abs(x / 2 + (y + 47))))
        term2 = -x * torch.sin(torch.sqrt(torch.abs(x - (y + 47))))
        return term1 + term2


# ---------------------------------------------------------
#  BOHACHEVSKY
# ---------------------------------------------------------
class Bohachevsky(Function):
    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]
    def evaluate(self, pos):
        pos = _ensure_2d(pos)
        assert pos.shape[0] == 2, f"AssertionError: Bohachevsky function got input with > 2 dimensions."
        # x = pos[:, :-1]
        # y = pos[:, 1:]

        # return torch.sum(
        #     x**2 + 2 * y**2
        #     - 0.3 * torch.cos(3 * torch.pi * x)
        #     - 0.4 * torch.cos(4 * torch.pi * y)
        #     + 0.7
        # )
        x = pos[0]
        y = pos[1]

        return x**2 + 2*y**2 - 0.3 * torch.cos(3 * torch.pi * x) \
               - 0.4 * torch.cos(4 * torch.pi * y) + 0.7


# ---------------------------------------------------------
#  GRIEWANK
# ---------------------------------------------------------
class Griewank(Function):

    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]
    def evaluate(self, pos):
        pos = _ensure_2d(pos)
        M = pos.shape[0]

        sum_term = torch.sum(pos**2) / 4000
        i = torch.arange(1, M+1, device=pos.device).float().unsqueeze(1)   # (M,1)
        prod_term = torch.prod(torch.cos(pos / torch.sqrt(i)))

        return sum_term - prod_term + 1


# ---------------------------------------------------------
#  MICHALEWICZ (k = 10)
# ---------------------------------------------------------
class Michalewicz(Function):

    def __init__(self, k=10):
        self.k = k
        self.bounds = BOUNDS[self.__class__.__name__]

    def evaluate(self, pos):
        pos = _ensure_2d(pos)
        M = pos.shape[0]

        i = torch.arange(1, M + 1, device=pos.device).float().unsqueeze(1)
        return -torch.sum(
            torch.sin(pos) * torch.pow(torch.sin(i * pos**2 / math.pi), 2 * self.k)
        )


# ---------------------------------------------------------
#  PLATEAU
# ---------------------------------------------------------
## version in the paper, which however is wrong/too easy to optimize
# class Plateau(Function):

#     def __init__(self):
#         self.bounds = BOUNDS[self.__class__.__name__]
#     def evaluate(self, pos):
#         pos = _ensure_2d(pos)
#         return 30 + torch.sum(torch.floor(pos))

class Plateau(Function):
    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]

    def evaluate(self, pos):
        pos = _ensure_2d(pos)
# ---------------------------------------------------------
class Shubert(Function):

    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]
    def evaluate(self, pos):
        pos = _ensure_2d(pos)
        M = pos.shape[0]
        results = torch.ones(pos.shape[0], device=pos.device)

        for m in range(M):
            xm = pos[:, m]
            inner_sum = torch.sum(
                torch.arange(1, 6, device=pos.device).float()
                * torch.cos(
                    (torch.arange(1, 6, device=pos.device).float() + 1) * xm.unsqueeze(1)
                    + torch.arange(1, 6, device=pos.device).float()
                )
            )
            results *= inner_sum
        return results


# ---------------------------------------------------------
#  VINCENT
# ---------------------------------------------------------
class Vincent(Function):

    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]
    def evaluate(self, pos):
        pos = _ensure_2d(pos)
        # ensure domain x > 0
        return torch.sum(torch.sin(10 * torch.log(pos)))


# ---------------------------------------------------------
#  XIN-SHE YANG
# ---------------------------------------------------------
class XinSheYang(Function):

    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]
    def evaluate(self, pos):
        pos = _ensure_2d(pos)
        sum_squares = torch.sum(pos**2)
        sum_sin = torch.sum(torch.sin(pos**2))
        return (torch.sum(torch.abs(pos))) * torch.exp(sum_sin) - 1

