import torch
import math
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
    "Eggholder":      (-5.12, 5.12),
}

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


# ---------------------------------------------------------
#  ACKLEY
# ---------------------------------------------------------

class Ackley(Function):
    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]
    def evaluate(self, pos):
        M = pos.shape[1]

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
        return torch.sum(pos**2)


# ---------------------------------------------------------
#  RASTRIGIN
# ---------------------------------------------------------
class Rastrigin(Function):
    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]
    def evaluate(self, pos):
        pos = _ensure_2d(pos)
        M = pos.shape[1]
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
        x = pos[0]
        y = pos[1]
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
        x = pos[:, :-1]
        y = pos[:, 1:]

        return torch.sum(
            x**2 + 2 * y**2
            - 0.3 * torch.cos(3 * torch.pi * x)
            - 0.4 * torch.cos(4 * torch.pi * y)
            + 0.7
        )


# ---------------------------------------------------------
#  GRIEWANK
# ---------------------------------------------------------
class Griewank(Function):

    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]
    def evaluate(self, pos):
        pos = _ensure_2d(pos)
        M = pos.shape[1]

        sum_term = torch.sum(pos**2) / 4000
        i = torch.arange(1, M + 1, device=pos.device).float()
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
        M = pos.shape[1]

        i = torch.arange(1, M + 1, device=pos.device).float()
        return -torch.sum(
            torch.sin(pos) * torch.pow(torch.sin(i * pos**2 / math.pi), 2 * self.k)
        )


# ---------------------------------------------------------
#  PLATEAU
# ---------------------------------------------------------
class Plateau(Function):

    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]
    def evaluate(self, pos):
        pos = _ensure_2d(pos)
        return 30 + torch.sum(torch.floor(pos))


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
        pos = _ensure_2d(pos)
        x = pos[:, :-1]
        y = pos[:, 1:]

        return torch.sum(100 * (y - x**2)**2 + (x - 1)**2)


# ---------------------------------------------------------
#  SHUBERT
# ---------------------------------------------------------
class Shubert(Function):

    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]
    def evaluate(self, pos):
        pos = _ensure_2d(pos)
        M = pos.shape[1]
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

