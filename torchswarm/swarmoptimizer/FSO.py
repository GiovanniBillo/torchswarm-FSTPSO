import time

import torch
import copy
import math

from torchswarm.particle.particle_factory import get_particle_instance
from torchswarm.utils.parameters import SwarmParameters
from FRBS import Frbs
from debug_utils import _vprint, ParticleStateTracker

class FuzzySwarmOptimizer:
    def __init__(self, dimensions, swarm_optimizer_type="fuzzy", particle=None, verbose=False, **kwargs):
        self.verbose=verbose
        self.swarm_size = math.floor(10 + 2*math.sqrt(dimensions)) # set according to rule in paper, paragraph 3
        
        if not particle:
            self.particle = get_particle_instance(swarm_optimizer_type)
        else:
            self.particle = particle
        self.max_iterations = kwargs.get('max_iterations') if kwargs.get('max_iterations') else 100
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = kwargs.get("device") if kwargs.get("device") else device
        
        self.seed = torch.manual_seed(kwargs.get("seed")) if kwargs.get("seed") else 0 
        torch.manual_seed(self.seed)

        self.swarm = []
        self.swarm_old = []
        self.dimensions = dimensions

        # extra stuff for fuzzy implementation
        self.f_triangle = torch.tensor(torch.inf)# highest value of fitness at the first iteration
        self.delta_max = None # length of diagonal of hyperrectangle in search space
        print("Initialized FuzzySwarmOptimizer object.")

    def calculate_delta_max(self, bounds):
        delta_max = 0
        for i in range(self.dimensions):
            delta_max += (bounds[1]-bounds[0])**2
        return math.sqrt(delta_max)


    def optimize(self, function):

            self.fitness_function = function
            print("Initializing particle swarm...")

            bounds = function.bounds
            low, high = bounds
            side = high - low

            # will be needed inside particles
            # self.delta_max = math.sqrt(self.dimensions) * abs(side)
            self.delta_max = self.calculate_delta_max(bounds)



            # --- initialize particles ---
            self.swarm = []
            for i in range(self.swarm_size):

                # instantiate particle
                p = self.particle(
                    dimensions=self.dimensions,
                    max_iterations=self.max_iterations,
                    bounds=bounds,
                    # seed=self.seed --> this initializes all the particles in the same position!!!
                    seed=i
                )

                # 1. initialize velocity (IMPORTANT — do NOT leave as zeros!)
                vel_scale = (high - low) * 0.1     # 10% of search range
                p.velocity = (torch.rand_like(p.position) - 0.5) * vel_scale

                # 2. initial pbest = current position
                p.pbest_position = p.position.clone()

                # 3. initial pbest_value = fitness(position)
                p.pbest_value = self.fitness_function.evaluate(p.position)

                self.swarm.append(p)

            # --- initialize global best ---
            self.gbest_particle = min(self.swarm, key=lambda p: p.pbest_value.item())
            self.gbest_value = self.gbest_particle.pbest_value.clone()
            self.gbest_position = self.gbest_particle.pbest_position.clone()

            print(f"Fuzzy Swarm of size {self.swarm_size} initialized for "
                  f"{function.__class__.__name__} with bounds {bounds} and delta_max={self.delta_max}.")

    def compute_delta(self, X_i, X_j):
        '''
            - considers the distance between two particles i and j
        '''
        diff = torch.sub(X_i.position, X_j.position)
        delta = torch.linalg.norm(diff, ord=2)

        assert torch.isreal(delta), f"AssertionError: delta computation for {X_i}, {X_j} did not give back a real number."
        assert 0< delta < self.delta_max, f"AssertionError: computed delta is outside of the allowed range [0, dmax] i.e 0, {self.delta_max}"

        return delta 
        
        print("Completed Delta computations.") 

    def compute_phi(self, X, X_prev):
        """
        Considers the positions of one SINGLE particle
        at current and previous iterations.
        """
        print("DEBUGGING PHI")
        pos = X.position
        pos_prev = X_prev.position
        print("positions:", pos, pos_prev)

        delta = self.compute_delta(X, X_prev)
         
        print("delta:", delta)
        # Evaluate fitness
        f_pos      = self.fitness_function.evaluate(pos)
        f_prev_pos = self.fitness_function.evaluate(pos_prev)
        ftri       = torch.max(self.f_triangle, torch.Tensor([1e-6])) # fix to avoid division by 0

        print("fitnesses:", f_pos, f_prev_pos, ftri)
        # Compute the min terms
        min_curr = torch.min(f_pos, ftri)
        min_prev = torch.min(f_prev_pos, ftri) 
        

        print("minima:", min_curr, min_prev)
        # Final phi expression
        phi = (delta / self.delta_max) * ((min_curr - min_prev)/ torch.abs(ftri)) # should be an absolute value right?

        # phi is a tensor → convert to python float for comparison
        phi_scalar = phi.item()

        print("phi:", phi)

        print("END OF DEBUGGING PHI")
        assert -1 < phi_scalar < 1, f"AssertionError: phi should be in range [-1, 1], but is {phi_scalar}"

        return phi


    def get_params(self, X, X_prev):
        delta = self.compute_delta(X, X_prev) 
        phi = self.compute_phi(X, X_prev) 
        
        return delta, phi


    def run(self, verbosity=True):
        frbs = Frbs(self.delta_max)
        print("FRBS initialized for FuzzySwarmOptimizer and FRBS")

        tracker = ParticleStateTracker()
        print("ParticleTracker initialized for FuzzySwarmOptimizer")
        swarm_parameters = SwarmParameters()
        swarm_parameters.r1 = 0
        swarm_parameters.r2 = 0

        # Per-particle previous values for debugging
        prev_delta = [None] * self.swarm_size
        prev_phi   = [None] * self.swarm_size
        prev_memb  = [None] * self.swarm_size  # flattened memberships
        prev_params = [None] * self.swarm_size  # dict of fuzzy params

        # --- Run
        for iteration in range(self.max_iterations):
            self.swarm_old = copy.deepcopy(self.swarm) # save the values to compute phi 
            tic = time.monotonic()

            # --- Set PBest
            for particle in self.swarm:
                fitness_candidate = self.fitness_function.evaluate(particle.position)
                if (particle.pbest_value > fitness_candidate):
                    print(f"Updated PBEST")
                    particle.pbest_value = fitness_candidate
                    particle.pbest_position = particle.position.clone()

            # --- Set GBest
            for particle in self.swarm:
                best_fitness_candidate = self.fitness_function.evaluate(particle.position)
                if self.gbest_value > best_fitness_candidate:

                    print(f"Updated Gbest")
                    self.gbest_value = best_fitness_candidate
                    self.gbest_position = particle.position.clone()
                    self.gbest_particle = copy.deepcopy(particle)
            r1s = []
            r2s = []

            # --- For Each Particle Update Velocity
            for particle in self.swarm:
                parameters = particle.update_velocity(self.gbest_position, iteration)
                particle.move()
                r1s.append(parameters.r1)
                r2s.append(parameters.r2)
            toc = time.monotonic()
            swarm_parameters.r1 = (sum(r1s) / self.swarm_size).item()
            swarm_parameters.r2 = (sum(r2s) / self.swarm_size).item()           
            
            # After first iteration (setting pbest, gbest, updating velocities and moving), get the value for self.f_triangle
            if iteration == 0:
               # self.f_triangle = self.gbest_value
               # _vprint(self.verbose,f"Set f_triangle to: {self.f_triangle}")
               vector_first_fitness = [self.fitness_function.evaluate(p.position) for p in self.swarm]
               print("vector of first fitnesses:", vector_first_fitness)

               self.f_triangle = max(vector_first_fitness) 
               print("ftriangle:", self.f_triangle)
               _vprint(self.verbose,f"Set f_triangle to: {self.f_triangle}")
      
            # --- For Each Particle update hyperparameters autonomously
            _vprint(self.verbose,"Updating Parameters based on values of Delta and Phi...")
            for i, p in enumerate(self.swarm):
                # 1) Compute delta, phi for particle i
                delta, phi = self.get_params(self.swarm[i], self.swarm_old[i])

                # 2) Compute memberships
                delta_m, phi_m = frbs.compute_memberships(delta, phi)
                _vprint(
                    verbosity,
                    f"[iter {iteration}] particle {i} delta={delta}, phi={phi}, "
                    f"delta_m={delta_m}, phi_m={phi_m}"
                )

                rules = frbs.define_rules(delta_m, phi_m)
                _vprint(verbosity, f"Rules:{rules}")
                new_params = frbs.sugeno(rules)

                # --- DEBUG CHECKS from iteration 1 onwards ---
                if iteration > 0:
                    # delta / phi change
                    if prev_delta[i] is not None:
                        if abs(delta - prev_delta[i]) < 1e-12:
                            print(
                                f"[WARN] iter {iteration}, particle {i}: delta did NOT change "
                                f"(delta={delta}, prev={prev_delta[i]})"
                            )
                    if prev_phi[i] is not None:
                        if abs(phi - prev_phi[i]) < 1e-12:
                            print(
                                f"[WARN] iter {iteration}, particle {i}: phi did NOT change "
                                f"(phi={phi}, prev={prev_phi[i]})"
                            )

                    # memberships change
                    memb_vec = [
                                list(delta_m.values()),
                                list(phi_m.values())
                                ]
                    if prev_memb[i] is not None:
                        if (memb_vec == prev_memb[i]):
                            print(
                                f"[WARN] iter {iteration}, particle {i}: memberships did NOT change"
                            )

                    # fuzzy parameters change
                    if prev_params[i] is not None:
                        for key in ["Inertia", "Social", "Cognitive", "L", "U"]:
                            curr_val = new_params[key]
                            prev_val = prev_params[i][key]
                            if torch.allclose(curr_val, prev_val, atol=1e-12):
                                print(
                                    f"[WARN] iter {iteration}, particle {i}: "
                                    f"param {key} did NOT change "
                                    f"(curr={curr_val.item()}, prev={prev_val.item()})"
                                )

                # 3) Apply new params for this iteration
                p.w[iteration]     = new_params["Inertia"]
                p.c_soc[iteration] = new_params["Social"]
                p.c_cog[iteration] = new_params["Cognitive"]

                # IMPORTANT: check L < U using the *new* values
                assert new_params["L"] < new_params["U"], (
                    f"AssertionError: L should be smaller than U, "
                    f"but L={new_params['L']} >= U={new_params['U']} at iteration {iteration}"
                )
                p.L[iteration] = new_params["L"]
                p.U[iteration] = new_params["U"]

                # 4) Store current values for next-iteration comparison
                prev_delta[i] = float(delta)
                prev_phi[i] = float(phi)
                print("delta, phi:", delta, phi)
                print("delta, phi memberships:", delta_m.values(), phi_m.values())
                prev_memb[i] = [list(delta_m.values()),
                                list(phi_m.values())
                                ]
                prev_params[i] = {k: v.detach().clone() for k, v in new_params.items()}

                # tracker.track(iteration, self.swarm)

                _vprint(
                    verbosity,
                    f"Successfully updated parameters for particle {i}: {new_params}"
                )
        swarm_parameters.gbest_position = self.gbest_position
        swarm_parameters.gbest_value = self.gbest_value.item()
        return swarm_parameters


