import time
import os 
from datetime import datetime
import torch
import copy
import math

from cli import get_args

args = get_args()

VERBOSE = args.verbose
MODEL = args.model
NRUNS = args.nruns
NITER = args.niter

# ---------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
summary_file = "summary_results.txt"
model_file = "std_results.txt" if MODEL == "std" else "fuzzy_results.txt"

# Clear previous logs
open(summary_file, "w").close()
open(model_file, "w").close()

def log(msg):
    print(msg)
    with open(summary_file, "a") as f:
        f.write(msg + "\n")
    with open(model_file, "a") as f:
        f.write(msg + "\n")

class Function:
    def __init__(self):
        self.dimensions = None
        self.bounds = None
    def evaluate(self, pos):
        raise NotImplementedError
BOUNDS = {
    "Ackley":       (-30, 30),
    }

class Ackley(Function):
    def __init__(self):
        self.bounds = BOUNDS[self.__class__.__name__]
    def evaluate(self, swarm):
        # swarm: (N, D, C)  (C may be 1)
        # reduce over D and C -> output (N,)
        dims = tuple(range(1, swarm.ndim))  # (1,2) for (N,D,C)

        mean_sq = torch.mean(swarm**2, dim=dims)                      # (N,)
        mean_cos = torch.mean(torch.cos(2 * torch.pi * swarm), dim=dims)  # (N,)

        term1 = -20 * torch.exp(-0.2 * torch.sqrt(mean_sq))
        term2 = -torch.exp(mean_cos)
        return term1 + term2 + 20 + torch.e                           # (N,)


class ParallelSwarmOptimizer:
    def __init__(self, sol_shape, swarm_size, fitness_function, swarm_optimizer_type="standard", particle=None, verbose=False, **kwargs):
        self.swarm_size = swarm_size
        self.max_iterations = kwargs.get('max_iterations') if kwargs.get('max_iterations') else 100
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = kwargs.get("device") if kwargs.get("device") else device
        self.fitness_function = fitness_function
        self.bounds = fitness_function.bounds
        # A sol_shape parameter would probably be more coincise than dimensions and classes

        self.seed = torch.manual_seed(kwargs.get("seed")) if kwargs.get("seed") else 0 
        torch.manual_seed(self.seed)

        self.sol_shape = sol_shape 
        self.dimensions = sol_shape[0] 
        self.classes = sol_shape[1] 
        self.verbose=verbose
        self.name = self.__class__.__name__

        self.swarm = torch.empty(
            (self.swarm_size, self.dimensions, self.classes), device=device
        ).uniform_(self.bounds[0], self.bounds[1])

        self.swarm_velocities = torch.zeros(
            (self.swarm_size, self.dimensions, self.classes), device=device
        )

        self.inertia = 0.9 
        self.social = 1.5 
        self.cognitive = 1.5 
        # optima initialization
        self.local_best_values = torch.full(
            (self.swarm_size,), float("inf"), device=self.swarm.device
        )
        self.local_best_positions = self.swarm.clone()

        first_fitness = self.fitness_function.evaluate(self.swarm)
        self.global_best_value = torch.min(first_fitness) 
        self.global_best_position = self.swarm[torch.argmin(first_fitness)] 

        print(f"Initialized {self.name} object.")
    
    def optimize(self, function):
        pass

    # NB: it's normal that PSO is slower to convergence with the same number of iterations: it's due to the parallel implementation happening synchronously.  

    def run(self, verbosity=True):
        for i in range(self.max_iterations):
            tic = time.monotonic()
            r1 = torch.rand_like(self.swarm_velocities)
            r2 = torch.rand_like(self.swarm_velocities)

            current_fitness = self.fitness_function.evaluate(self.swarm)
            print("CURRENT_FITNESS:", current_fitness)
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

            # update global best
            # better_global_fitness_idx =  self.local_best_values < self.global_best_value  
            # self.global_best_value = torch.min(self.local_best_values[better_global_fitness_idx]) 
            # global_best_idx = torch.argmax(current_fitness[better_global_fitness_idx])
            # self.global_best_value = self.swarm[global_best_idx] 

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
            
            self.swarm_velocities = (
                self.inertia * self.swarm_velocities
                + r1 * self.cognitive * (self.local_best_positions - self.swarm)
                + r2 * self.social * (self.global_best_position - self.swarm)
            )

            ## possible extension: also clamp max velocities
            # self.swarm = torch.clamp(self.swarm, self.bounds[0], self.bounds[1])

            # move
            self.swarm = self.swarm + self.swarm_velocities
            self.swarm = torch.clamp(self.swarm, self.bounds[0], self.bounds[1])

            toc = time.monotonic()
            print('Iteration {:.0f} >> global best fitness {:.3f}  | iteration time {:.3f}'.format(i + 1, self.global_best_value.item(), toc - tic))
        best_val = self.global_best_value
        best_pos = self.global_best_position

        print("Done")

        return best_val, best_pos

def run_test(func_class, sol_shape, name=None, filename="master_table.csv", args=args):
    if name is None:
        name = func_class.__name__

    header = f"{'='*80}\nTesting function: {name} (Solution shape={sol_shape}) using model={MODEL}\n{'='*80}"
    args = f"{'='*80}\nARGS == {args}:\n{'='*80}"
    log(header)
    log(args)

    ABF = 0 
    for run in range(1, NRUNS + 1):
        log(f"\n--- RUN {run}/{NRUNS} ---")

        # choose optimizer
        if MODEL == "std":
            opt = ParallelSwarmOptimizer(
                sol_shape,
                swarm_size=100,
                fitness_function = func_class(),
                swarm_optimizer_type="standard",
                max_iterations=NITER,
                verbose=VERBOSE,
            )
        elif MODEL == "fuzzy": 
            opt = FuzzySwarmOptimizer(
                sol_shape,
                swarm_optimizer_type="fuzzy",
                max_iterations=NITER,
            )
        else:
            print("Unrecognized model passed!")
            raise ValueError


        opt.optimize(func_class())
        best_val, best_pos = opt.run(verbosity=VERBOSE)
         
        # accumulate for average
        ABF += best_val.item()

        best_pos = best_pos.tolist() if hasattr(best_pos, 'tolist') else list(best_pos)

        # log to text files
        log(f"Best fitness in run {run}: {best_val}")
        log(f"Best position: {best_pos}")

        # save_csv(name, run, best_val, best_pos)

    ABF /= NRUNS


    log(f"{'-'*80}\nFinished {name}. Average Best Value: {ABF}\n{'-'*80}")
    # build_master_table(filename)
    print(f"FInal results saved at {filename}.") 

def main():
    benchmark_shape = torch.Size([5, 1])
    run_test(Ackley, sol_shape=benchmark_shape, args=args)

if __name__=="__main__":
    main()
