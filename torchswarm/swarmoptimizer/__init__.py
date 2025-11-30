import time

import torch
import copy
import math

from torchswarm.particle.particle_factory import get_particle_instance
from torchswarm.utils.parameters import SwarmParameters
from FRBS import frbs

class SwarmOptimizer:
    def __init__(self, dimensions, swarm_size, swarm_optimizer_type="standard", particle=None, **kwargs):
        self.swarm_size = swarm_size
        if not particle:
            self.particle = get_particle_instance(swarm_optimizer_type)
        else:
            self.particle = particle
        self.max_iterations = kwargs.get('max_iterations') if kwargs.get('max_iterations') else 100
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = kwargs.get("device") if kwargs.get("device") else device
        self.swarm = []
        self.gbest_position = torch.Tensor([0]).to(device)
        self.gbest_particle = None
        self.gbest_value = torch.Tensor([float("inf")]).to(device)
        # for i in range(self.swarm_size):
        #     self.swarm.append(self.particle(dimensions, **kwargs))

        self.dimensions = dimensions

    def optimize(self, function):
        # Perhaps it's best to initalize here, with full information about the dimensions and search space
        # dimensions = function.dimensions
        dimensions = self.dimensions
        bounds = function.bounds
        self.fitness_function = function
        print("Initializing particle swarm...") 
        for i in range(self.swarm_size):
                    self.swarm.append(self.particle(dimensions=dimensions, bounds=bounds))

        print(f"Swarm Initialized for {function.__class__.__name__} with bounds:{bounds}.") 

    def run(self, verbosity=True):
        swarm_parameters = SwarmParameters()
        swarm_parameters.r1 = 0
        swarm_parameters.r2 = 0

        # --- Run
        for iteration in range(self.max_iterations):
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
                    self.gbest_particle = copy.deepcopy(particle)
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


        return swarm_parameters




class FuzzySwarmOptimizer:
    def __init__(self, dimensions, swarm_optimizer_type="fuzzy", particle=None, **kwargs):

        self.swarm_size = math.floor(10 + 2*math.sqrt(dimensions)) # set according to rule in paper, paragraph 3
        
        if not particle:
            self.particle = get_particle_instance(swarm_optimizer_type)
        else:
            self.particle = particle
        self.max_iterations = kwargs.get('max_iterations') if kwargs.get('max_iterations') else 100
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = kwargs.get("device") if kwargs.get("device") else device
        self.swarm = []
        self.swarm_old = []
        self.gbest_position = torch.Tensor([0]).to(device)
        self.gbest_position_old = torch.Tensor([0]).to(device)
        self.gbest_particle = None
        self.gbest_value = torch.Tensor([float("inf")]).to(device)
        # for i in range(self.swarm_size):
        #     self.swarm.append(self.particle(dimensions, **kwargs))

        self.dimensions = dimensions

        # extra stuff for fuzzy implementation
        self.f_triangle = None # highest value of fitness at the first iteration
        self.delta_max = None # length of diagonal of hyperrectangle in search space
        print("Initialized FuzzySwarmOptimizer object.")

    def optimize(self, function):
        # Perhaps it's best to initalize here, with full information about the dimensions and search space
        # dimensions = function.dimensions
        dimensions = self.dimensions
        bounds = function.bounds
        self.fitness_function = function
        print("Initializing particle swarm...") 
        for i in range(self.swarm_size):
                    print("Type of self.particle:", type(self.particle))
                    self.swarm.append(self.particle(dimensions=dimensions,max_iterations=self.max_iterations, bounds=bounds))
        assert isinstance(bounds, tuple), f"AssertionError: self.bounds object is not a tuple and its bounds for different dimensions are not equal"

        # define parameter delta_max for the search space
        print("bounds:", bounds, type(bounds))
        self.delta_max = bounds[1]**2 * dimensions # according to formula: d = (l_1)^2 + ... (l_N)**2, generalized for a situation where all bounds, once set, are considered to be equal in each dimension
        print(f"Fuzzy Swarm Initialized for {function.__class__.__name__} with bounds:{bounds} and delta_max={self.delta_max}.") 
    
    def calculate_delta(X_i, X_j):
        '''
            - considers the distance between two particles i and j
        '''
        # for i in self.swarm_size:
        #     for j in self.swarm_size:
        #     diff = self.swarm[i] - self.swarm[j, ] 
        #     Delta[i, :] = torch.linalg.norm(diff)

        diff = torch.sub(X_i - X_j)
        delta = torch.linalg.norm(diff)

        assert torch.isreal(res), f"AssertionError: delta computation for {X_i}, {X_j} did not give back a real number."
        return delta 
        
        print("Completed Delta computations.") 

    def calculate_phi(X, X_prev):
        '''
            - considers the positions of one SINGLE particle at current and previous iterations 
        '''
        delta = self.compute_delta(X, X_prev) 
        phi = (delta/self.delta_max)*torch.sub(torch.cmin(self.fitness_function(X), self.f_triangle, torch.cmin(self.fitness_function(X_prev), self.f_triangle)/torch.abs(self.f_triangle)))
        assert 0 < phi < 1, f"AssertionError: phi should be in range 0 - 1, but is {phi}" 
        return phi

    def get_params(X, X_prev):
        delta = self.compute_delta(X, X_prev) 
        phi = self.compute_phi(X, X_prev) 
        
        return delta, phi


    def run(self, verbosity=True):

        frbs = Frbs(self.delta_max)
        print("FRBS initialized for FuzzySwarmOptimizer and FRBS")

        swarm_parameters = SwarmParameters()
        swarm_parameters.r1 = 0
        swarm_parameters.r2 = 0

        # --- Run
        for iteration in range(self.max_iterations):
            self.swarm_old = self.swarm.copy() # save the values to compute phi 
            tic = time.monotonic()
            # --- Set PBest
            for particle in self.swarm:
                fitness_candidate = self.fitness_function.evaluate(particle.position)
                if (particle.pbest_value > fitness_candidate):
                    particle.pbest_value = fitness_candidate
                    particle.pbest_position_old = particle.pbest_position.copy() 
                    particle.pbest_position = particle.position.clone()
            # --- Set GBest
            for particle in self.swarm:
                best_fitness_candidate = self.fitness_function.evaluate(particle.position)
                if self.gbest_value > best_fitness_candidate:
                    self.gbest_position_old = self.gbest_position.copy() # saving the value which will be needed in update_velocity  
                    self.gbest_value = best_fitness_candidate
                    self.gbest_position = particle.position.clone()
                    self.gbest_particle = copy.deepcopy(particle)
            r1s = []
            r2s = []

            # --- For Each Particle Update Velocity
            for particle in self.swarm:
                parameters = particle.update_velocity(self.gbest_position_old)
                particle.move()
                r1s.append(parameters.r1)
                r2s.append(parameters.r2)
            toc = time.monotonic()
            swarm_parameters.r1 = (sum(r1s) / self.swarm_size).item()
            swarm_parameters.r2 = (sum(r2s) / self.swarm_size).item()

                                                                       
            # --- For Each Particle update hyperparameters autonomously
            for p in range(self.swarm_size):
                print("The delta and phi update should take place here!!")
                delta, phi= get_params(self.swarm[p], self.swarm_old[p])
                frbs.get_delta_membership(delta) 
                frbs.get_phi_membership(phi) 
            if verbosity == True:
                print('Iteration {:.0f} >> global best fitness {:.3f}  | iteration time {:.3f}'.format(iteration + 1, self.gbest_value.item(), toc - tic))

            if iteration == 0:
                self.f_triangle = self.gbest_value.item()


        swarm_parameters.gbest_position = self.gbest_position
        swarm_parameters.gbest_value = self.gbest_value.item()
        swarm_parameters.c1 = self.gbest_particle.c1
        swarm_parameters.c2 = self.gbest_particle.c2
        


        return swarm_parameters


