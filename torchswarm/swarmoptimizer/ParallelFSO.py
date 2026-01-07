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
        # return math.sqrt(self.dimensions * (bounds[1] - bounds[0])**2)

    def compute_delta(self, i):
        diff = self.swarm[i] - self.prev_swarm[i]
        return torch.norm(diff)

    def compute_phi(self, i):
        delta = self.compute_delta(i)

        f_curr = self.local_best_values[i]
        f_prev = self.prev_local_best_values[i]
        ftri   = torch.maximum(self.f_triangle, torch.tensor(1e-6))

        phi = (delta / self.delta_max) * ((f_curr - f_prev) / torch.abs(ftri))
        return torch.clamp(phi, -1.0, 1.0)

    def update_hyperparameters(self, iteration):
        if iteration == 0:
            self.prev_swarm = self.swarm.clone()
            self.prev_local_best_values = self.local_best_values.clone()
            self.f_triangle = torch.max(self.local_best_values)
            return

        for i in range(self.swarm_size):
            delta = self.compute_delta(i)
            phi = self.compute_phi(i)

            delta_m, phi_m = self.frbs.compute_memberships(delta, phi)
            rules = self.frbs.define_rules(delta_m, phi_m)
            new_params = self.frbs.sugeno(rules)

            self.w[i]     = new_params["Inertia"]
            self.c_soc[i] = new_params["Social"]
            self.c_cog[i] = new_params["Cognitive"]
            self.U[i] = new_params["U"]
            self.L[i] = new_params["L"]

        self.prev_swarm = self.swarm.clone()
        self.prev_local_best_values = self.local_best_values.clone()

