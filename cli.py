# include/cli.py

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Standard/FST PSO Training")
    # ---------------------------------------------------------
    # Command-line arguments
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="enable verbose output for PSO runs"
    )

    parser.add_argument(
        "--model",
        choices=["std", "fuzzy"],
        default="std",
        help="Select which PSO variant to run"
    )

    parser.add_argument(
        "--nruns",
        type=int,
        default=1,
        help="Number of runs to perform for each benchmark function"
    )

    parser.add_argument(
        "--niter",
        type=int,
        default=100,
        help="Number of iterations per run"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="serial",
        help="version of the algorithm to be used"
    )

    args = parser.parse_args()
    return args

