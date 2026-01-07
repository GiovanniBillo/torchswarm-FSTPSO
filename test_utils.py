import argparse
import torch
import os
from datetime import datetime
import csv
from consts import RESULTS_DIR
print("Torch version:", torch.__version__)
print("Built with CUDA:", torch.version.cuda)

from torchswarm.swarmoptimizer.SO import SwarmOptimizer
from torchswarm.swarmoptimizer.FSO import FuzzySwarmOptimizer
from torchswarm.swarmoptimizer.ParallelSO import ParallelSwarmOptimizer
from torchswarm.swarmoptimizer.ParallelFSO import ParallelFuzzySwarmOptimizer

from consts import TRUE_OPTIMA
from cli import get_args

from debug_utils import save_csv, build_master_table, get_logger

args = get_args()
VERBOSE = args.verbose
MODEL = args.model
NRUNS = args.nruns
NITER = args.niter
MODE = args.mode

log, summary_path, model_path = get_logger(
    model=MODEL,
    nruns=NRUNS,
    niter=NITER,
    mode=MODE,
    path=RESULTS_DIR,
)

log(f"Logging to:\n- {summary_path}\n- {model_path}")

def run_test(func_class, sol_shape, name=None, filename="master_table.csv", args=args):
    if name is None:
        name = func_class.__name__

    header = f"{'='*80}\nTesting function: {name} (Solution shape={sol_shape}) using model={MODEL}\n{'='*80}"

    args_block = f"{'='*80}\nARGS == {args}:\n{'='*80}"
    log(header)
    log(args_block)

    ABF = 0 # average best fitness
    ABP = torch.zeros(sol_shape) # average best position
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
                    sol_shape=sol_shape,
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
                    max_iterations=NITER,
                    verbose=VERBOSE,
                )
            elif MODEL == "fuzzy": 
                opt = ParallelFuzzySwarmOptimizer(
                    sol_shape=sol_shape,
                    # swarm_size=100,
                    fitness_function = func_class(),
                    max_iterations=NITER,
                    verbose=VERBOSE,
                )
            else:
                print("Unrecognized model passed!")
                raise ValueError
            opt.optimize(func_class())
            best_val, best_pos = opt.run(verbosity=VERBOSE)
         
        
        # accumulate for average
        ABF += best_val
        ABP += best_pos 

        best_pos = best_pos.tolist() if hasattr(best_pos, 'tolist') else list(best_pos)


        # log to text files
        log(f"Best fitness in run {run}: {best_val}")
        log(f"Best position: {best_pos}")

        save_csv(name, run, best_val, best_pos)

    ABF /= NRUNS


    log(f"{'-'*80}\nFinished {name}. Average Best Value: {ABF}\n{'-'*80} Average Best Position: {ABP}\n{'-'*80}")
    # if hasattr(func_class, real_params):
    #     log(f"REAL SOLUTION:{func_class().real_params}")
    build_master_table(filename)
    print(f"Final results saved at {os.path.join(RESULTS_DIR, filename)}.")
