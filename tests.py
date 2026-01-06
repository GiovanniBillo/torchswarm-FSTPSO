import torch

# from torchswarm.functions.benchmarks import (
#     Ackley,
#     Sphere,
#     Rastrigin,
#     Eggholder,
#     # Alpine,
#     # Bohachevsky,
#     # Griewank,
#     # Michalewicz,
#     # Plateau,
#     # # Quintic,
#     # Rosenbrock,
#     # Shubert,
#     # Vincent,
#     # XinSheYang,
# )

# from torchswarm.functions.misc import LotkaVolterra

from torchswarm.functions.benchmarks_parallel import (
    Ackley,
    Sphere,
    Rastrigin,
    Eggholder,
    Alpine,
    Bohachevsky,
    Griewank,
    Michalewicz,
    Plateau,
    Rosenbrock,
    Shubert,
    Vincent,
    XinSheYang,
)

from torchswarm.functions.lotka_volterra import LotkaVolterra
# from torchswarm.functions.reflectance import Reflectance
from test_utils import run_test

# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    
    # Default benchmark shape used by "most" functions
    # (D=5, C=1)
    benchmark_shape = torch.Size([5, 1])

    # Special case: 2D benchmarks (D=2, C=1)
    benchmark_shape_2d = torch.Size([2, 1])

    # Lotka-Volterra special (matrix params)
    LV_shape = torch.Size([2, 2])

    # ---------------------------------------------------------
    # Functions supporting ANY dimension (here we test them at D=5)
    # ---------------------------------------------------------
    run_test(Ackley,      sol_shape=benchmark_shape)
    run_test(Sphere,      sol_shape=benchmark_shape)
    run_test(Rastrigin,   sol_shape=benchmark_shape)
    run_test(Alpine,      sol_shape=benchmark_shape)
    run_test(Griewank,    sol_shape=benchmark_shape)
    run_test(Michalewicz, sol_shape=benchmark_shape)
    run_test(Plateau,     sol_shape=benchmark_shape)
    run_test(Rosenbrock,  sol_shape=benchmark_shape)
    run_test(Vincent,     sol_shape=benchmark_shape)
    run_test(XinSheYang,  sol_shape=benchmark_shape)

    # ---------------------------------------------------------
    # Constraints / special cases (2D)
    # ---------------------------------------------------------
    run_test(Bohachevsky, sol_shape=benchmark_shape_2d)
    run_test(Eggholder,   sol_shape=benchmark_shape_2d)
    run_test(Shubert,     sol_shape=benchmark_shape_2d)

    # LotkaVolterra
    run_test(LotkaVolterra, sol_shape=LV_shape)


