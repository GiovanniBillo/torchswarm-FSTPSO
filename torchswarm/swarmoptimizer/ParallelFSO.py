import torch
import math
from torchswarm.swarmoptimizer.ParallelSO import ParallelSwarmOptimizer
from FRBS import Frbs
from debug_utils import _vprint

def calculate_delta_max(bounds, dim, device=None, dtype=None):
    """
    Compute delta_max for FST-PSO.

    bounds:
        - tuple: (low, high)
        - dict:  {name: (low, high)}

    dim:
        number of dimensions (D)

    returns:
        scalar float delta_max
    """

    if isinstance(bounds, tuple):
        low, high = bounds
        delta = torch.full((dim,), high - low, device=device, dtype=dtype)

    elif isinstance(bounds, dict):
        assert len(bounds) == dim, \
            f"bounds dict length {len(bounds)} != dim {dim}"

        delta = torch.tensor(
            [hi - lo for (lo, hi) in bounds.values()],
            device=device,
            dtype=dtype
        )

    else:
        raise TypeError(
            f"bounds must be tuple or dict, got {type(bounds)}"
        )

    assert torch.all(delta > 0), "All bounds must have positive width"

    # Euclidean diameter of the box
    return torch.sqrt(torch.sum(delta ** 2)).item()

class ParallelFuzzySwarmOptimizer(ParallelSwarmOptimizer):
                 
    def __init__(self, use_swarm_size_heuristic=True, *args, **kwargs):

        if use_swarm_size_heuristic:
            self.sol_shape = kwargs.get('sol_shape')
            self.dimensions = self.sol_shape[0]
            self.swarm_size = math.floor(10 + 2*math.sqrt(self.dimensions)) 

            super().__init__(swarm_size=self.swarm_size, *args, **kwargs)
            _vprint(self.verbose, "ADAPTIVE SWARM_SIZE:", self.swarm_size)    

        else:
            super().__init__(*args, **kwargs)

        self.delta_max = self._calculate_delta_max(self.bounds)
        self.frbs = Frbs(self.delta_max, verbose=self.verbose)

        self._init_swarm()
        self._init_vel()
    

        # make delta max a tensor
        self.delta_max = torch.full(
            (self.swarm_size,),
            self.delta_max,
            dtype=torch.float32,
            device=self.swarm.device  # or wherever your swarm lives
        )

        device = self.swarm.device
        self.max_vel_clamp = 0.2
        self.min_vel_clamp = 0.1
        # Per-particle adaptive hyperparameters
        self.w     = torch.full((self.swarm_size,), self.inertia, device=device)
        self.c_soc = torch.full((self.swarm_size,), self.social, device=device)
        self.c_cog = torch.full((self.swarm_size,), self.cognitive, device=device)
        self.U = torch.full((self.swarm_size,), self.max_vel_clamp, device=device)
        self.L = torch.full((self.swarm_size,), self.min_vel_clamp, device=device)
        
        self.prev_swarm = None
        self.prev_local_best_values = None
        self.f_triangle = torch.tensor(float("inf"), device=device)
        
        self.apply_clamp_velocities = True 
    def _calculate_delta_max(self, bounds):
        return calculate_delta_max(self.bounds, self.dimensions, device=self.device, dtype=torch.float32)

    def compute_delta(self):
        diff = self.swarm - self.prev_swarm
        return torch.norm(diff, dim=1).squeeze()

    def compute_phi(self):
        delta = self.compute_delta()

        f_curr = self.local_best_values
        f_prev = self.prev_local_best_values
        ftri   = torch.maximum(self.f_triangle, torch.tensor(1e-6))

        phi = (delta / self.delta_max) * ((f_curr - f_prev) / torch.abs(ftri))
        return torch.clamp(phi, -1.0, 1.0)

    def update_hyperparameters(self, iteration):
        if iteration == 0:
            self.prev_swarm = self.swarm.clone()
            self.prev_local_best_values = self.local_best_values.clone()
            self.f_triangle = torch.max(self.local_best_values)
            return

        # delta: (N,), phi: (N,)
        delta = self.compute_delta()   # or however you named it
        phi = self.compute_phi()       # (N,)

        new_params = self.frbs.forward(delta, phi)  # dict of (N,)

        self.w     = new_params["Inertia"]
        self.c_soc = new_params["Social"]
        self.c_cog = new_params["Cognitive"]
        # If you use L,U elsewhere:
        self.L     = new_params["L"]
        self.U     = new_params["U"]

        self.prev_swarm = self.swarm.clone()
        self.prev_local_best_values = self.local_best_values.clone()

