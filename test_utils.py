import argparse
import torch
import os
from datetime import datetime
import csv

print("Torch version:", torch.__version__)
print("Built with CUDA:", torch.version.cuda)

from torchswarm.swarmoptimizer.SO import SwarmOptimizer
from torchswarm.swarmoptimizer.FSO import FuzzySwarmOptimizer
from torchswarm.swarmoptimizer.ParallelSO import ParallelSwarmOptimizer

from consts import TRUE_OPTIMA
from cli import get_args

from debug_utils import save_csv, build_master_table # not really debug utils but whatever

args = get_args()
VERBOSE = args.verbose
MODEL = args.model
NRUNS = args.nruns
NITER = args.niter
MODE = args.mode

# ---------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
summary_file = "summary_results.txt"
# model_file = f"{MODEL}_{NRUNS}_{NITER}_{MODE}.txt" if MODEL == "std" else "fuzzy_results.txt"

model_file = f"{MODEL}_{NRUNS}_{NITER}_{MODE}.txt" 

# Clear previous logs
open(summary_file, "w").close()
open(model_file, "w").close()

def log(msg):
    print(msg)
    with open(summary_file, "a") as f:
        f.write(msg + "\n")
    with open(model_file, "a") as f:
        f.write(msg + "\n")

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
        if MODE == "serial": 
            # choose optimizer
            if MODEL == "std":
                opt = SwarmOptimizer(
                    sol_shape,
                    swarm_size=100,
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
            swarm_parameters = opt.run(verbosity=VERBOSE)
         
            best_val = opt.gbest_value
            best_pos = swarm_parameters.gbest_position

        elif MODE == "parallel":
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
                raise NotImplementedError
                # opt = FuzzySwarmOptimizer(
                #     sol_shape,
                #     swarm_optimizer_type="fuzzy",
                #     max_iterations=NITER,
                # )
            else:
                print("Unrecognized model passed!")
                raise ValueError
            opt.optimize(func_class())
            best_val, best_pos = opt.run(verbosity=VERBOSE)
         
        
        # accumulate for average
        ABF += best_val

        best_pos = best_pos.tolist() if hasattr(best_pos, 'tolist') else list(best_pos)

        # log to text files
        log(f"Best fitness in run {run}: {best_val}")
        log(f"Best position: {best_pos}")

        save_csv(name, run, best_val, best_pos)

    ABF /= NRUNS


    log(f"{'-'*80}\nFinished {name}. Average Best Value: {ABF}\n{'-'*80}")
    build_master_table(filename)
    print(f"FInal results saved at {filename}.") 

