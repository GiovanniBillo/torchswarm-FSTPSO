import time

import torch
import copy
import math

from torchswarm.particle.particle_factory import get_particle_instance
from torchswarm.utils.parameters import SwarmParameters
from FRBS import Frbs
from debug_utils import _vprint, ParticleStateTracker

class SwarmOptimizer:
    def __init__(self, sol_shape, swarm_size, swarm_optimizer_type="standard", particle=None, verbose=False, **kwargs):
        self.swarm_size = swarm_size
        if not particle:
            self.particle = get_particle_instance(swarm_optimizer_type)
        else:
            self.particle = particle
        self.max_iterations = kwargs.get('max_iterations') if kwargs.get('max_iterations') else 100
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = kwargs.get("device") if kwargs.get("device") else device
        self.swarm = []
        # A sol_shape parameter would probably be more coincise than dimensions and classes

        self.seed = torch.manual_seed(kwargs.get("seed")) if kwargs.get("seed") else 0 
        torch.manual_seed(self.seed)

        self.sol_shape = sol_shape 
        self.dimensions = sol_shape[0] 
        self.classes = sol_shape[1] 
        self.verbose=verbose
        self.name = self.__class__.__name__
        self.mode = "serial"
        print(f"Initialized {self.name} object.")
    
    def optimize(self, function):

        if self.mode == "serial":

            bounds = function.bounds
            self.fitness_function = function
            print("Initializing particle swarm...") 
            for i in range(self.swarm_size):
                        # self.swarm.append(self.particle(dimensions=dimensions, bounds=bounds, function=function, classes=classes))
                        self.swarm.append(self.particle(self.sol_shape, bounds=bounds, fitness_function=function))

            self.gbest_particle = None
            self.gbest_position = min((p.pbest_position for p in self.swarm), key=self.fitness_function.evaluate) 
            _vprint(self.verbose, "self.gbest_position:", self.gbest_position)
            self.gbest_value = self.fitness_function.evaluate(self.gbest_position) 
            _vprint(self.verbose, "self.gbest_value:", self.gbest_value)
            _vprint(self.verbose, f"{self.name}: Swarm Initialized for {function.__class__.__name__} with bounds:{bounds}.") 

        # if self.mode == "parallel":

        #     self.swarm = torch.empty(size=(self.swarm_size, sol_shape)).uniform_(bounds[0], bounds[1]) 
        #     self.gbest_particle = None
        #     first_fitness = self.fitness_function.evaluate(self.swarm)
        #     self.gbest_position = torch.min(first_fitness) 

    def run(self, verbosity=True):
        swarm_parameters = SwarmParameters()
        swarm_parameters.r1 = 0
        swarm_parameters.r2 = 0

        # --- Run
        if self.mode == "serial":
            for iteration in range(self.max_iterations):
                _vprint(self.verbose, f"{'='*80}\n iteration {iteration} \n{'='*80}")
                tic = time.monotonic()
                # --- Set PBest
                for particle in self.swarm:
                    fitness_candidate = self.fitness_function.evaluate(particle.position)
                    if (particle.pbest_value > fitness_candidate):
                        particle.pbest_value = fitness_candidate
                        particle.pbest_position = particle.position.clone()
                # --- Set GBest
                for particle in self.swarm:
                    best_fitness_candidate = self.fitness_function.evaluate(particle.position)
                    if self.gbest_value > best_fitness_candidate:
                        self.gbest_value = best_fitness_candidate
                        self.gbest_position = particle.position.clone()
                        self.gbest_particle = copy.deepcopy(particle) # TODO: not sure about the efficiency of this
                r1s = []
                r2s = []
                # --- For Each Particle Update Velocity
                for particle in self.swarm:
                    parameters = particle.update_velocity(self.gbest_position)
                    particle.move()
                    r1s.append(parameters.r1)
                    r2s.append(parameters.r2)
                toc = time.monotonic()
                swarm_parameters.r1 = (sum(r1s) / self.swarm_size).item()
                swarm_parameters.r2 = (sum(r2s) / self.swarm_size).item()
                if verbosity == True:
                    print('Iteration {:.0f} >> global best fitness {:.3f}  | iteration time {:.3f}'
                          .format(iteration + 1, self.gbest_value.item(), toc - tic))
            swarm_parameters.gbest_position = self.gbest_position
            swarm_parameters.gbest_value = self.gbest_value.item()
            swarm_parameters.c1 = self.gbest_particle.c1
            swarm_parameters.c2 = self.gbest_particle.c2
        # if self.mode == "parallel":

        #         local_best_values = torch.random_like(self.swarm)
        #         local_best_positions = torch.random_like(self.swarm) 
        #         global_best_value = torch.random_like(sol_shape)
        #         global_best_position = torch.random_like(sol_shape)

                


        return swarm_parameters

