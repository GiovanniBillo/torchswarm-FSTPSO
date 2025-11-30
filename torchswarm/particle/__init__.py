import torch
import numpy as np

from torchswarm.utils.rotation_utils import get_rotation_matrix, get_inverse_matrix, get_phi_matrix
from torchswarm.utils.parameters import SwarmParameters
from FRBS import Frbs
from debug_utils import _vprint
class Particle:
    def __init__(self, dimensions, w=0.5, c1=2, c2=2, **kwargs):
        self.dimensions = dimensions
        self.w = w
        self.c1 = c1
        self.c2 = c2
        classes = kwargs.get("classes") if kwargs.get("classes") else 1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = kwargs.get("device") if kwargs.get("device") else self.device
        if kwargs.get("bounds"):
            self.bounds = kwargs.get("bounds")
            self.position = (self.bounds[0] - self.bounds[1]) * torch.rand(dimensions, classes).to(self.device) + self.bounds[1]
        else:
            self.bounds = None
            self.position = torch.rand(dimensions, classes).to(self.device)
        self.velocity = torch.zeros((dimensions, classes)).to(self.device)
        self.pbest_position = self.position
        self.pbest_value = torch.Tensor([float("inf")]).to(self.device)

    def update_velocity(self, gbest_position):
        r1 = torch.rand(1).to(self.device)
        r2 = torch.rand(1).to(self.device)
        for i in range(0, self.dimensions):
            self.velocity[i] = self.w * self.velocity[i] \
                               + self.c1 * r1 * (self.pbest_position[i] - self.position[i]) \
                               + self.c2 * r2 * (gbest_position[i] - self.position[i])

        swarm_parameters = SwarmParameters()
        swarm_parameters.r1 = r1
        swarm_parameters.r2 = r2
        return swarm_parameters

    def move(self):
        for i in range(0, self.dimensions):
            self.position[i] = self.position[i] + self.velocity[i]
        if self.bounds:
            self.position = torch.clamp(self.position, self.bounds[0], self.bounds[1])


class RotatedParticle(Particle):
    def __init__(self, dimensions, w, c1=2, c2=2, **kwargs):
        super(RotatedParticle, self).__init__(dimensions, w, c1, c2, **kwargs)

    def update_velocity(self, gbest_position):
        r1 = torch.rand(1).to(self.device)
        r2 = torch.rand(1).to(self.device)
        a_matrix = get_rotation_matrix(self.dimensions, np.pi / 5, 0.4)
        a_inverse_matrix = get_inverse_matrix(a_matrix)
        x = a_inverse_matrix * get_phi_matrix(self.dimensions, self.c1, r1) * a_matrix
        self.velocity = self.w * self.velocity \
                        + torch.matmul(
            (a_inverse_matrix * get_phi_matrix(self.dimensions, self.c1, r1) * a_matrix).float().to(self.device),
            (self.pbest_position - self.position).float().to(self.device)) \
                        + torch.matmul(
            (a_inverse_matrix * get_phi_matrix(self.dimensions, self.c2, r2) * a_matrix).float().to(self.device),
            (gbest_position - self.position).float().to(self.device))
        swarm_parameters = SwarmParameters()
        swarm_parameters.r1 = r1
        swarm_parameters.r2 = r2
        return swarm_parameters


class ExponentiallyWeightedMomentumParticle(Particle):
    def __init__(self, dimensions, beta=0.9, c1=2, c2=2, **kwargs):
        super(ExponentiallyWeightedMomentumParticle, self).__init__(dimensions, 0, c1, c2, **kwargs)
        self.beta = beta
        self.momentum = torch.zeros((dimensions, 1)).to(self.device)

    def update_velocity(self, gbest_position):
        r1 = torch.rand(1).to(self.device)
        r2 = torch.rand(1).to(self.device)
        for i in range(0, self.dimensions):
            momentum_t = self.beta * self.momentum[i] + (1 - self.beta) * self.velocity[i]
            self.velocity[i] = momentum_t \
                               + self.c1 * r1 * (self.pbest_position[i] - self.position[i]) \
                               + self.c2 * r2 * (gbest_position[i] - self.position[i])
            self.momentum[i] = momentum_t
        swarm_parameters = SwarmParameters()
        swarm_parameters.r1 = r1
        swarm_parameters.r2 = r2
        return swarm_parameters


class RotatedEWMParticle(ExponentiallyWeightedMomentumParticle):
    def __init__(self, dimensions, beta=0.9, c1=2, c2=2, **kwargs):
        super(RotatedEWMParticle, self).__init__(dimensions, beta, c1, c2, **kwargs)

    def update_velocity(self, gbest_position):
        r1 = torch.rand(1).to(self.device)
        r2 = torch.rand(1).to(self.device)
        momentum_t = self.beta * self.momentum + (1 - self.beta) * self.velocity
        a_matrix = get_rotation_matrix(self.dimensions, np.pi / 5, 0.4)
        a_inverse_matrix = get_inverse_matrix(a_matrix)
        x = a_inverse_matrix * get_phi_matrix(self.dimensions, self.c1, r1) * a_matrix
        self.velocity = momentum_t \
                        + torch.matmul(
            (a_inverse_matrix * get_phi_matrix(self.dimensions, self.c1, r1) * a_matrix).float().to(self.device),
            (self.pbest_position - self.position).float().to(self.device)) \
                        + torch.matmul(
            (a_inverse_matrix * get_phi_matrix(self.dimensions, self.c2, r2) * a_matrix).float().to(self.device),
            (gbest_position - self.position).float().to(self.device))

        swarm_parameters = SwarmParameters()
        swarm_parameters.r1 = r1
        swarm_parameters.r2 = r2
        return swarm_parameters

class FuzzyParticle(Particle):
        # maybe initialize like this?
        # super(RotatedParticle, self).__init__(dimensions, w, c1, c2, **kwargs)

    def __init__(self, dimensions, max_iterations, w=0.5, c1=2, c2=2, U=0.2, L=0.1, verbose=True, **kwargs):
        self.verbose = verbose
        self.dimensions = dimensions
        self.w = torch.full((max_iterations,), w) # inertia coefficient
        self.c1 = torch.full((max_iterations,), c1) # social coefficient
        self.c2 = torch.full((max_iterations,), c2) # cognitive coefficient
        self.U = torch.full((max_iterations,), U) # minimum velocity clamp 
        self.L = torch.full((max_iterations,), L) # maximum velocity clamp
        
        num_params = 5
        self.parameters = torch.zeros((max_iterations, num_params))#TODO: for clarity, this should be probablyi turned into its own class

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
                    ## Or maybe it can already be done without loops? TODO
                # self.position = (self.bounds[0] - self.bounds[1]) * torch.rand(dimensions, classes).to(self.device) + self.bounds[1]
        else:
            self.bounds = None
            self.position = torch.rand(dimensions, classes).to(self.device)

        # TODO: I fear this might be a completely different animal: by modifying velocity bounds in each dimension the particles learn better, but who says that the bounds have to be the same for each dimension?

        if kwargs.get("velbounds"):
            self.velbounds = kwargs.get("velbounds")
            # self.position = (self.velbounds[0] - self.velbounds[1]) * torch.rand(dimensions, classes).to(self.device) + self.velbounds[1]
        else:
            velmin, velmax = 0, 0 
            self.velbounds = torch.tensor([velmin, velmax]) 
            # self.position = torch.rand(dimensions, classes).to(self.device)

        self.position_old =  torch.rand(dimensions, classes).to(self.device)
        self.velocity = torch.zeros((dimensions, classes)).to(self.device)
        self.velocity_old = torch.zeros((dimensions, classes)).to(self.device)
        self.pbest_position = self.position
        self.pbest_position_old = None 
        self.pbest_value = torch.Tensor([float("inf")]).to(self.device)

        _vprint(verbose, "!!! INITIALIZING NOT YET IMPLEMENTED FUZZYPARTICLE!!!")

    def update_velocity(self, gbest_position_old):

        r1 = torch.rand(1).to(self.device)
        r2 = torch.rand(1).to(self.device)

        # r1 = torch.rand(self.dimensions)
        # r2 = torch.rand(self.dimensions)
        it = 1

        for i in range(0, self.dimensions):

            self.velbounds[0] = self.L[it]*(self.bounds[i][1] - self.bounds[i][0])
            self.velbounds[1] = self.U[it]*(self.bounds[i][1] - self.bounds[i][0])

            self.velocity_old[i] = self.velocity[i]

            # self.velocity[i] = self.w * self.velocity[i] \
            #                    + self.c1 * r1 * (self.pbest_position[i] - self.position[i]) \
            #                    + self.c2 * r2 * (gbest_position[i] - self.position[i])
            # using equation (5) in the paper, which apparently takes everything from one step BEFORE the current one

            _vprint(self.verbose, "DEBUG velocity update:")
            _vprint(self.verbose, f"  i = {i}, it = {it}")
            _vprint(self.verbose, f"  self.w[it].shape                 = {self.w[it].shape}")
            _vprint(self.verbose, f"  self.velocity_old[i].shape       = {self.velocity_old[i].shape}")
            _vprint(self.verbose, f"  self.c1[it].shape                = {self.c1[it].shape}")
            _vprint(self.verbose, f"  self.pbest_position_old[i].shape = {self.pbest_position_old[i].shape}")
            _vprint(self.verbose, f"  self.position_old[i].shape       = {self.position_old[i].shape}")
            _vprint(self.verbose, f"  self.c2[it].shape                = {self.c2[it].shape}")
            _vprint(self.verbose, f"  gbest_position_old[i].shape      = {gbest_position_old[i].shape}")
            _vprint(self.verbose, f"  self.position[i].shape           = {self.position[i].shape}")
            _vprint(self.verbose, "  ------------------------------------------------------")
            _vprint(self.verbose, f"  Current Position: {self.position}")

            self.velocity[i] = self.w[it] * self.velocity_old[i] \
                               + self.c1[it] * r1 * (self.pbest_position_old[i] - self.position_old[i]) \
                               + self.c2[it] * r2 * (gbest_position_old[i] - self.position[i])

            rd = torch.rand(1).to(self.device)

            ## reference, from lecture 
            # for i, v in enumerate(max_velocities):
            #     if np.abs(new_velocity[i]) < v[0]:
            #         new_velocity[i] = np.sign(new_velocity[i]) * v[0]
            #     elif np.abs(new_velocity[i]) > v[1]:
            #         new_velocity[i] = np.sign(new_velocity[i]) * v[1]
            
            # TODO: not sure this is the right implementation

            _vprint(self.verbose, f"  {self.velocity[i].shape} vs {self.velbounds[0].shape}")
            if self.velocity[i] < self.velbounds[0]: # if below the minimum velocity 
                self.velocity[i] = rd*torch.sign(self.velocity[i]) * self.velbounds[0] 
                print("velocity out of minbound!!! USE DAMPING STRATEGY")
                pass
            elif self.velocity[i] > self.velbounds[0]: # if above the max velocity 
                self.velocity[i] = rd*torch.sign(self.velocity[i]) * self.velbounds[1] 
                print("velocity out of maxbound!!! USE DAMPING STRATEGY")
                pass
            it += 1

        # I don't know what this is for, but I leave it here in order not to break anything...
        swarm_parameters = SwarmParameters()
        swarm_parameters.r1 = r1
        swarm_parameters.r2 = r2

        self.parameters[i] = torch.Tensor([self.w[it], self.c1[it], self.c2[it], self.L[it], self.U[it]]) 
        print("!!! UPDATING VELOCITY ON NOT YET IMPLEMENTED FUZZYPARTICLE!!!")

        return swarm_parameters

    def move(self):
        for i in range(0, self.dimensions):
            self.position_old[i] - self.position[i]
            self.position[i] = self.position[i] + self.velocity[i]

            # resetting positions outside of allowed values by damping strategy
            if self.position[i] < self.bounds[i][0]:
                self.position[i] = self.bounds[i][0]
                self.velocity[i] = - torch.rand(1).to(self.device)* self.velocity[i]
            elif self.position[i] > self.bounds[i][1]:
                self.position[i] = self.bounds[i][1]
                self.velocity[i] = - torch.rand(1).to(self.device) * self.velocity[i]

        # if self.bounds:
        #     self.position = torch.clamp(self.position, self.bounds[0], self.bounds[1])

        print("!!! MOVING NOT YET IMPLEMENTED FUZZYPARTICLE!!!")
