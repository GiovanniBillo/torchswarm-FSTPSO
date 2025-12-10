import torch
import numpy as np

from torchswarm.utils.rotation_utils import get_rotation_matrix, get_inverse_matrix, get_phi_matrix
from torchswarm.utils.parameters import SwarmParameters
from FRBS import Frbs
from debug_utils import _vprint
from torchswarm.particle.P import Particle

class FuzzyParticle(Particle):

    def __init__(self, dimensions, max_iterations, w=0.5, c_soc=2, c_cog=2, U=0.2, L=0.1, verbose=False, **kwargs): # TODO: tune parameters further
        self.verbose = verbose
        self.dimensions = dimensions
        self.seed = torch.manual_seed(kwargs.get("seed")) if kwargs.get("seed") else torch.manual_seed(0) 
        self.w = torch.full((max_iterations,), w) # inertia coefficient
        self.c_soc = torch.full((max_iterations,), c_soc) # social coefficient
        self.c_cog = torch.full((max_iterations,), c_cog) # cognitive coefficient
        self.U = torch.full((max_iterations,), U) # minimum velocity clamp 
        self.L = torch.full((max_iterations,), L) # maximum velocity clamp
        
        self.seed = kwargs.get("seed") if kwargs.get("seed") else 0
        torch.manual_seed(self.seed)

        num_params = 5
        self.parameters = torch.zeros((max_iterations, num_params))

        classes = kwargs.get("classes") if kwargs.get("classes") else 1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = kwargs.get("device") if kwargs.get("device") else self.device

        if kwargs.get("bounds"):
            
            self.bounds = kwargs.get("bounds")
            self.bounds = torch.tensor([self.bounds] * self.dimensions, dtype=torch.float32)
            _vprint(verbose,"Bounds passed to FuzzyParticleINstance:", self.bounds, "of type:", type(self.bounds))
            if isinstance(self.bounds, tuple):
                # simple case: bounds are the same for every dimension
                self.position = (self.bounds[0] - self.bounds[1]) * torch.rand(dimensions, classes).to(self.device) + self.bounds[1]
            elif isinstance(self.bounds, torch.Tensor):
                # more complicated case: a m-dimensional array of tuples specifies the boundaries of each dimension
                self.position = torch.zeros(dimensions, classes)
                for m in range(dimensions):
                    self.position[m] = (self.bounds[m][0] - self.bounds[m][1]) * torch.rand(1, classes).to(self.device) + self.bounds[m][1]
        else:
            self.bounds = None
            self.position = torch.rand(dimensions, classes).to(self.device)

        # TODO: I fear this might be a completely different animal: by modifying velocity bounds in each dimension the particles learn better, but who says that the bounds have to be the same for each dimension?

        if kwargs.get("velbounds"):
            self.velbounds = kwargs.get("velbounds")
            self.velbounds = torch.tensor([self.velbounds] * self.dimensions, dtype=torch.float32)
        else:
            self.velbounds = torch.empty((self.dimensions, 2))
            for m in range(dimensions):
                minbound, maxbound = self.bounds[m][0], self.bounds[m][1]
                velmin, velmax = U*(maxbound - minbound), L*(maxbound - minbound)
                self.velbounds[m] = torch.tensor([velmin, velmax],dtype=torch.float32)


        self.velocity = torch.zeros((dimensions, classes)).to(self.device)
        self.velocity_old = torch.zeros((dimensions, classes)).to(self.device)
        self.pbest_position = self.position
        self.pbest_value = torch.Tensor([float("inf")]).to(self.device)


    def update_velocity(self, gbest_position, it):

        _vprint(self.verbose,"Updating velocity...")
        r1 = torch.rand(1).to(self.device)
        r2 = torch.rand(1).to(self.device)

        # r1 = torch.rand(self.dimensions)
        # r2 = torch.rand(self.dimensions)

        for i in range(0, self.dimensions):
            # TODO: correct! could/should be one for each dimension (like the bounds)
            velmin = self.velbounds[i][0]
            velmax = self.velbounds[i][1]
            
            maxbound = self.bounds[i][1]
            minbound = self.bounds[i][0]

            velmin = self.L[it]*(maxbound - minbound)
            velmax = self.U[it]*(maxbound - minbound)

            _vprint(self.verbose, "DEBUG velocity update:")
            _vprint(self.verbose, f"  i = {i}, it = {it}")
            _vprint(self.verbose, f"  self.w[it].shape                 = {self.w[it].shape}")
            _vprint(self.verbose, f"  self.velocity_old[i].shape       = {self.velocity_old[i].shape}")
            _vprint(self.verbose, f"  self.c_soc[it].shape                = {self.c_soc[it].shape}")
            _vprint(self.verbose, f"  self.c_cog[it].shape             = {self.c_cog[it].shape}")
            _vprint(self.verbose, f"  self.position[i].shape           = {self.position[i].shape}")
            _vprint(self.verbose, "  ------------------------------------------------------")
            _vprint(self.verbose, f"  Current Position: {self.position}")


            old = self.velocity.clone() 
            self.velocity[i] = self.w[it] * self.velocity[i] \
                                       + self.c_soc[it] * r1 * (self.pbest_position[i] - self.position[i]) \
                                       + self.c_cog[it] * r2 * (gbest_position[i] - self.position[i])


            assert not torch.allclose(old, self.velocity[i]), f"Velocity has not changed" 

            rd = torch.rand(1).to(self.device)

            _vprint(self.verbose, f"  {self.velocity[i].shape} vs {self.velbounds[0].shape}")
            if self.velocity[i] < velmin: # if below the minimum velocity 
                _vprint(self.verbose,"velocity out of minbound! Using damping strategy")
                self.velocity[i] = rd*torch.sign(self.velocity[i]) * velmin 
            elif self.velocity[i] > velmax: # if above the max velocity 
                _vprint(self.verbose,"velocity out of minbound! Using damping strategy")
                self.velocity[i] = rd*torch.sign(self.velocity[i]) * velmax 

        swarm_parameters = SwarmParameters()
        swarm_parameters.r1 = r1
        swarm_parameters.r2 = r2

        self.parameters[i] = torch.Tensor([self.w[it], self.c_soc[it], self.c_cog[it], self.L[it], self.U[it]]) 

        return swarm_parameters

    def move(self):
        _vprint(self.verbose,"Moving fuzzy particle!!!")
        for i in range(0, self.dimensions):
            self.position[i] = self.position[i] + self.velocity[i]

            # resetting positions outside of allowed values by damping strategy
            if self.position[i] < self.bounds[i][0]:
                self.position[i] = self.bounds[i][0]
                self.velocity[i] = - torch.rand(1).to(self.device)* self.velocity[i]
            elif self.position[i] > self.bounds[i][1]:
                self.position[i] = self.bounds[i][1]
                self.velocity[i] = - torch.rand(1).to(self.device) * self.velocity[i]
