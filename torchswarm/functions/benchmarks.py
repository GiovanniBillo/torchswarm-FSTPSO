import torch
import math
from debug_utils import _vprint
from consts import BOUNDS, PROBLEM_BOUNDS
from scipy.integrate import odeint

class Function:
    def __init__(self):
        self.dimensions = None
        self.bounds = None
    def evaluate(self, pos):
        raise NotImplementedError


def _ensure_2d(pos):
    """Ensure pos is [batch, dim]."""
    if pos.dim() == 1:
        pos = pos.unsqueeze(0)
    return pos

def Reflectance(Function):
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

## Good reference for typical benchmark optimization functions: https://www.sfu.ca/~ssurjano/ackley.html or whatever function one might need
# ---------------------------------------------------------
#  ACKLEY
# ---------------------------------------------------------
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

class LotkaVolterra(Function):
    def __init__(self, target_traj):
        self.target_traj = target_traj
    def eval(self, individual):
        t = np.linspace(0, 100, 100)
        initial_conditions=self.target_traj[0]
        sol = odeint(lotka_volterra, initial_conditions, t, args=(individual,))
        return np.mean((sol - self.target_traj) ** 2)

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
#  ALPINE
# ---------------------------------------------------------
class Alpine(Function):
    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]
    def evaluate(self, pos):
        pos = _ensure_2d(pos)
        return torch.sum(torch.abs(pos * torch.sin(pos) + 0.1 * pos))


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
        return torch.sum(torch.floor(pos + 0.5)**2)


# ---------------------------------------------------------
#  QUINTIC
# ---------------------------------------------------------
class Quintic(Function):

    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]
    def evaluate(self, pos):
        pos = _ensure_2d(pos)
        x = pos
        return torch.sum(
            torch.abs(x**5 - 3 * x**4 + 4 * x**3 + 2 * x**2 - 10 * x - 4)
        )


# ---------------------------------------------------------
#  ROSENBROCK
# ---------------------------------------------------------
class Rosenbrock(Function):
    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]

    def evaluate(self, pos):
        pos = _ensure_2d(pos)  # shape (batch, D)
        # x = pos

        # Compute term-by-term: dimension d interacts only with d+1
        # return torch.sum(
        #     100 * (x[:, 1:] - x[:, :-1]**2)**2 +
        #     (1 - x[:, :-1])**2,
        #     dim=1
        # )
        x = pos[:,0]  # (M,)

        return torch.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

# ---------------------------------------------------------
#  SHUBERT
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

