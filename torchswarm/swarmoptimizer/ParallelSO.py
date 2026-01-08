import time
import os 
from datetime import datetime
import torch
import copy
import math

from debug_utils import _vprint

def normalize_bounds(bounds, dim, device=None, dtype=None):
    """
    Normalize bounds into (lower, upper) tensors of shape (dim,).

    bounds:
        - tuple: (low, high)
        - dict:  {name: (low, high)}

    returns:
        lower: (dim,)
        upper: (dim,)
    """

    if isinstance(bounds, tuple):
        low, high = bounds
        lower = torch.full((dim,), low, device=device, dtype=dtype)
        upper = torch.full((dim,), high, device=device, dtype=dtype)

    elif isinstance(bounds, dict):
        assert len(bounds) == dim, \
            f"bounds dict length {len(bounds)} != dim {dim}"

        lower = []
        upper = []
        for key, (lo, hi) in bounds.items():
            lower.append(lo)
            upper.append(hi)

        lower = torch.tensor(lower, device=device, dtype=dtype)
        upper = torch.tensor(upper, device=device, dtype=dtype)

    else:
        raise TypeError(
            f"bounds must be tuple or dict, got {type(bounds)}"
        )

    assert lower.shape == (dim,)
    assert upper.shape == (dim,)
    assert torch.all(lower < upper)

    return lower, upper

def clamp_with_bounds(swarm, bounds):
    """
    Clamp swarm tensor using bounds.

    swarm:
        (N, D) or (N, D, C)

    bounds:
        tuple or dict

    returns:
        clamped swarm (same shape)
    """

    assert swarm.ndim in (2, 3), \
        f"swarm must be 2D or 3D, got {swarm.shape}"

    device = swarm.device
    dtype = swarm.dtype
    D = swarm.shape[1]

    lower, upper = normalize_bounds(bounds, D, device, dtype)

    if swarm.ndim == 2:
        # (N, D)
        return torch.max(
            torch.min(swarm, upper),
            lower
        )

    else:
        # (N, D, C)
        lower = lower.view(1, D, 1)
        upper = upper.view(1, D, 1)

        return torch.max(
            torch.min(swarm, upper),
            lower
        )
def _normalize_bounds(bounds, dim=None, device=None, dtype=torch.float32):
    """
    Normalize bounds into (lower, upper, keys).

    Supports:
      - tuple (low, high)
      - dict[str, (low, high)]

    Returns:
      lower: (D,) tensor
      upper: (D,) tensor
      keys:  list[str]
    """

    # -------------------------
    # Case 1: scalar tuple
    # -------------------------
    if isinstance(bounds, tuple):
        assert len(bounds) == 2, \
            f"tuple bounds must be (low, high), got {bounds}"

        assert dim is not None, \
            "dim must be provided when bounds is a tuple"

        low, high = bounds

        keys = [f"x{i}" for i in range(dim)]
        lower = torch.full((dim,), low, device=device, dtype=dtype)
        upper = torch.full((dim,), high, device=device, dtype=dtype)

    # -------------------------
    # Case 2: dict of tuples
    # -------------------------
    elif isinstance(bounds, dict):
        keys = list(bounds.keys())

        lower = torch.tensor(
            [bounds[k][0] for k in keys],
            device=device, dtype=dtype
        )
        upper = torch.tensor(
            [bounds[k][1] for k in keys],
            device=device, dtype=dtype
        )

    else:
        raise TypeError(
            f"bounds must be tuple or dict, got {type(bounds)}"
        )

    # -------------------------
    # Sanity checks
    # -------------------------
    assert lower.shape == upper.shape, "lower/upper shape mismatch"
    assert torch.all(lower < upper), "each lower bound must be < upper bound"

    return lower, upper, keys

class ParallelSwarmOptimizer:
    def init_swarm(self,
        bounds,
        swarm_size: int,
        dim: int | None = None,
        classes: int = 1,
        device=None,
        dtype=torch.float32,
    ):
        """
        Initialize swarm positions.

        Returns:
            params: (N, D, C)
            keys:   list[str] (length D)
        """

        # ---------------------------------
        # Normalize bounds
        # ---------------------------------
        lower, upper, keys = _normalize_bounds(
            bounds=bounds,
            dim=dim,
            device=device,
            dtype=dtype
        )

        D = lower.numel()
        N = swarm_size
        C = classes
        print(f"DIMENSIONS INITIALIZED:{D}")
        # ---------------------------------
        # Vectorized initialization
        # ---------------------------------
        # rand: (N, D, C)
        rand = torch.rand((N, D, C), device=device, dtype=dtype)

        # reshape bounds for broadcasting
        lower = lower.view(1, D, 1)
        upper = upper.view(1, D, 1)

        self.swarm = lower + rand * (upper - lower)

        # ---------------------------------
        # Sanity checks
        # ---------------------------------
        assert self.swarm.shape == (N, D, C), \
            f"params must be (N,D,C), got {params.shape}"

        # return params, keys
        return

    def _init_swarm(self):
        return self.init_swarm(bounds=self.bounds, swarm_size=self.swarm_size, dim=self.dimensions, classes=self.classes, device=self.device)

    def _init_vel(self):
        '''
        utility to reset velocity in derived classes
        '''
        self.swarm_velocities = torch.zeros(
            (self.swarm_size, self.dimensions, self.classes), device=self.device
        )
        return

    def __init__(self, sol_shape, fitness_function, swarm_size=100, particle=None, verbose=False, **kwargs):
        self.swarm_size = swarm_size
        self.max_iterations = kwargs.get('max_iterations') if kwargs.get('max_iterations') else 100
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = kwargs.get("device") if kwargs.get("device") else device
        self.fitness_function = fitness_function
        self.bounds = fitness_function.bounds
        # A sol_shape parameter would probably be more coincise than dimensions and classes

        # self.seed = torch.manual_seed(kwargs.get("seed")) if kwargs.get("seed") else 0 
        self.seed = torch.manual_seed(kwargs.get("seed")) if kwargs.get("seed") else torch.randint(100, (1,)).item() # ensures randomization between runs 
        torch.manual_seed(self.seed)

        self.sol_shape = sol_shape 
        self.dimensions = sol_shape[0] 
        self.classes = sol_shape[1] 
        self.verbose=verbose
        self.name = self.__class__.__name__
        self.swarm = None
        self.swarm_velocities = None

        self._init_swarm()
        self._init_vel() 


        self.inertia = 0.9 
        self.social = 1.5 
        self.cognitive = 1.5 

        self.w = torch.full((self.swarm_size,), self.inertia)
        self.c_soc = torch.full((self.swarm_size,), self.social)
        self.c_cog = torch.full((self.swarm_size,), self.cognitive)

        # optima initialization
        self.local_best_values = torch.full(
            (self.swarm_size,), float("inf"), device=self.swarm.device
        )
        self.local_best_positions = self.swarm.clone()

        first_fitness = self.fitness_function.evaluate(self.swarm)
        self.global_best_value = torch.min(first_fitness) 
        self.global_best_position = self.swarm[torch.argmin(first_fitness)] 
        
        self.apply_clamp_velocities = False
        print(f"Initialized {self.name} object.")
    
    def optimize(self, function):
        pass

    def update_hyperparameters(self, iteration):
        '''
        Hook for subclasses.
        Does nothing for standard static hyperparameters.
        '''
        return

    # NB: it's normal that PSO is slower to convergence with the same number of iterations: it's due to the parallel implementation happening synchronously.  
    def clamp_velocities(self):
        """
        Clamp swarm velocities tensor using bounds.

        swarm_velocities:
            (N, D) or (N, D, C)

        bounds:
            typically per-dimension (D,) from normalize_bounds(...)
        """
        v = self.swarm_velocities
        assert v.ndim in (2, 3), f"swarm_velocities must be 2D or 3D, got {v.shape}"

        device, dtype = v.device, v.dtype
        D = v.shape[1]

        lower, upper = normalize_bounds(self.bounds, D, device, dtype)  # expected (D,) or scalar

        if v.ndim == 2:
            # (N, D): rely on broadcasting of (D,)
            return torch.clamp(v, min=lower, max=upper)

        # (N, D, C)
        lower = lower.view(1, D, 1)   # per-dimension bounds
        upper = upper.view(1, D, 1)

        # Ensure L/U broadcast correctly.
        # - if scalars: ok
        # - if per-particle: must be (N,1,1)
        if isinstance(self.L, torch.Tensor) and self.L.ndim == 3 and self.L.shape[0] == 1 and self.L.shape[1] == v.shape[0]:
            self.L = self.L.transpose(0, 1)  # (1,N,1) -> (N,1,1)
        if isinstance(self.U, torch.Tensor) and self.U.ndim == 3 and self.U.shape[0] == 1 and self.U.shape[1] == v.shape[0]:
            self.U = self.U.transpose(0, 1)

        L = self.L.view(-1, 1, 1)   # (N,) -> (N,1,1)
        U = self.U.view(-1, 1, 1)   # (N,) -> (N,1,1)

        lower = L * lower
        upper = U * upper

        return torch.clamp(v, min=lower, max=upper)

    def run(self, verbosity=True):
        for i in range(self.max_iterations):
            tic = time.monotonic()
            r1 = torch.rand_like(self.swarm_velocities)
            r2 = torch.rand_like(self.swarm_velocities)

            current_fitness = self.fitness_function.evaluate(self.swarm)
            # update local best
            assert current_fitness.shape == (self.swarm_size,), \
                f"fitness must be (N,), got {current_fitness.shape}"

            assert self.local_best_values.shape == (self.swarm_size,), \
                f"pbest values must be (N,), got {self.local_best_values.shape}"

            assert self.local_best_positions.shape == self.swarm.shape, \
                f"pbest positions must match swarm shape {self.swarm.shape}, got {self.local_best_positions.shape}"

            better_local_fitness_idx = current_fitness < self.local_best_values
            assert better_local_fitness_idx.dtype == torch.bool, \
                f"mask must be bool, got {better_local_fitness_idx.dtype}"
            assert better_local_fitness_idx.shape == (self.swarm_size,), \
                f"mask must be (N,), got {better_local_fitness_idx.shape}"


            self.local_best_positions[better_local_fitness_idx] = self.swarm[better_local_fitness_idx]
            self.local_best_values[better_local_fitness_idx] = current_fitness[better_local_fitness_idx] ## do we actually need to update local and global

            assert self.local_best_values.shape == torch.Size([ self.swarm_size ]), f"AssertionError: shape should be {(self.swarm_size)} but is {better_local_fitness_idx.shape}"

            min_val, min_idx = torch.min(self.local_best_values, dim=0)
            assert min_val.ndim == 0, f"gbest fitness must be scalar, got shape {min_val.shape}"

            if min_val < self.global_best_value:
                self.global_best_value = min_val
                self.global_best_position = self.local_best_positions[min_idx].clone()

            assert isinstance(self.global_best_value, torch.Tensor) and self.global_best_value.ndim == 0, \
                f"global_best_value must be scalar tensor, got {type(self.global_best_value)} shape {getattr(self.global_best_value,'shape',None)}"
            assert self.global_best_position.shape == (self.dimensions, self.classes), \
                f"global_best_position must be (D,C), got {self.global_best_position.shape}"

            # update velocities
            assert self.inertia is not None and self.social is not None and self.cognitive is not None, \
                "Set inertia/social/cognitive before running"
            assert self.swarm.device == self.local_best_positions.device == self.local_best_values.device, \
                "swarm and best tensors must be on same device"
            
            ## vectorized form
            self.swarm_velocities = (
                self.w[:, None, None] * self.swarm_velocities
                + r1 * self.c_cog[:, None, None] * (self.local_best_positions - self.swarm)
                + r2 * self.c_soc[:, None, None] * (self.global_best_position - self.swarm)
            )
            if self.apply_clamp_velocities:
                self.clamp_velocities()
                _vprint(self.verbose, "All velocities were successfully clamped!")

            # move
            self.swarm = self.swarm + self.swarm_velocities
            self.swarm = clamp_with_bounds(self.swarm, self.bounds) 

            # hyperparameter update(useful for subclasses)
            self.update_hyperparameters(i)

            toc = time.monotonic()
            print('Iteration {:.0f} >> global best fitness {:.3f}  | iteration time {:.3f}'.format(i + 1, self.global_best_value.item(), toc - tic))
        best_val = self.global_best_value
        best_pos = self.global_best_position

        print("Done")

        return best_val, best_pos
