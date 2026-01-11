from __future__ import annotations
from consts import TRUE_OPTIMA, RESULTS_DIR
import os
import csv
import torch
from datetime import datetime

def _vprint(verbose: bool, *args, **kwargs) -> None:
    """Print only if verbose is True."""
    if verbose:
        print(*args, **kwargs)

class ParticleStateTracker:
    def __init__(self):
        self.prev_positions = {}
        self.prev_velocities = {}
        self.prev_params = {}   # per-particle fuzzy parameters

    def track(self, iteration, swarm):
        for idx, p in enumerate(swarm):

            # Extract current state
            pos = p.position.clone()
            vel = p.velocity.clone()

            params = {
                "w": p.w[iteration].clone(),
                "c_soc": p.c_soc[iteration].clone(),
                "c_cog": p.c_cog[iteration].clone(),
                "L": p.L[iteration].clone(),
                "U": p.U[iteration].clone(),
            }

            print("params:", params)
            if iteration > 0:  # skip iteration 0 (no previous iteration)

                # ---- ASSERT POSITION CHANGES ----
                prev_pos = self.prev_positions[idx]
                assert not torch.allclose(
                    pos, prev_pos, atol=1e-9
                ), f"[ERROR] Particle {idx} position did NOT change at iteration {iteration}. pos={pos}, prev_pos={prev_pos}"

                # ---- ASSERT VELOCITY CHANGES ----
                prev_vel = self.prev_velocities[idx]
                assert not torch.allclose(
                    vel, prev_vel, atol=1e-12
                ), f"[ERROR] Particle {idx} velocity did NOT change at iteration {iteration}. vel={vel}, prev_vel={prev_vel}"

                # ---- ASSERT FUZZY PARAMETER CHANGES ----
                prev_params = self.prev_params[idx]

                for key in params.keys():
                    # assert not torch.allclose(
                    #     params[key], prev_params[key], atol=1e-12
                    # ), (
                    #     f"[ERROR] Parameter {key} for particle {idx} was NOT updated "
                    #     f"at iteration {iteration}. Current={params[key]}, Previous={prev_params[key]}"
                    # )
                    if torch.allclose:
                        print(f"[ERROR] Parameter {key} for particle {idx} was NOT updated ")
                        print(f"at iteration {iteration}. Current={params[key]}, Previous={prev_params[key]}")


            # Store current state for next iteration
            self.prev_positions[idx] = pos
            self.prev_velocities[idx] = vel
            self.prev_params[idx] = params

            print("prev_params:", self.prev_params)

def save_csv(func_name, run_id, model, best_val, best_pos, seed, path=RESULTS_DIR):
    """
    Save benchmark results to a CSV file.
    Automatically creates the directory if it does not exist.
    """
    # Ensure directory exists
    os.makedirs(path, exist_ok=True)

    true_val = TRUE_OPTIMA.get(func_name, float('nan'))
    filepath = os.path.join(path, f"results_{model}_{func_name.lower()}.csv")

    file_exists = os.path.isfile(filepath)

    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["run", "best_found", "true_value", "error", "best_position", "seed"])
        writer.writerow([
            run_id,
            best_val,
            true_val,
            best_val - true_val,
            best_pos, 
            seed
        ])


def build_master_table(output_filename, path=RESULTS_DIR):
    """
    Build a master CSV summarizing best results for each function.
    Automatically creates the directory if it does not exist.
    """
    # Ensure directory exists
    os.makedirs(path, exist_ok=True)

    output_filepath = os.path.join(path, output_filename)

    with open(output_filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Function", "BestFound", "TrueValue", "Error"])

        for func_name in TRUE_OPTIMA.keys():
            func_csv = os.path.join(path, f"results_{func_name.lower()}.csv")

            if os.path.isfile(func_csv):
                with open(func_csv, "r") as fr:
                    rows = list(csv.reader(fr))
                    if len(rows) > 1:
                        last = rows[-1]
                        writer.writerow([
                            func_name,
                            last[1],  # best_found
                            last[2],  # true_value
                            last[3],  # error
                        ])
# ============================================================
# Results-dir aware text logging
# ============================================================

def ensure_results_dir(path=RESULTS_DIR) -> str:
    """Create results directory if missing and return its path."""
    os.makedirs(path, exist_ok=True)
    return path


def make_run_prefix(model: str, nruns: int, niter: int, mode: str) -> str:
    """Stable prefix used by both text logs and other artifacts."""
    return f"{model}_{nruns}_{niter}_{mode}"


def get_logger(model: str, nruns: int, niter: int, mode: str, *, path=RESULTS_DIR, timestamp: str | None = None):
    """
    Returns:
      - log(msg): prints + appends to both summary + model-specific file
      - summary_path
      - model_path

    All logs go into RESULTS_DIR.
    """
    ensure_results_dir(path)

    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = make_run_prefix(model, nruns, niter, mode)

    summary_path = os.path.join(path, f"summary_{prefix}_{ts}.txt")
    model_path   = os.path.join(path, f"log_{prefix}_{ts}.txt")

    # start fresh for this run
    open(summary_path, "w").close()
    open(model_path, "w").close()

    def log(msg: str):
        print(msg)
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        with open(model_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    return log, summary_path, model_path

