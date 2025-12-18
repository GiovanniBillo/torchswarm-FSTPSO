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
    # Sphere,
    # Rastrigin,
    # Eggholder,
    # Alpine,
    # Bohachevsky,
    # Griewank,
    # Michalewicz,
    # Plateau,
    # # Quintic,
    # Rosenbrock,
    # Shubert,
    # Vincent,
    # XinSheYang,
)
from torchswarm.functions.misc_parallel import LotkaVolterra

from test_utils import run_test

# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    
    benchmark_shape = torch.Size([5, 1])
    benchmark_shape_special = torch.Size([2, 1])
    LV_shape = torch.Size([2, 2])
    # Functions supporting ANY dimension

    run_test(Ackley, sol_shape=benchmark_shape)
    # run_test(Ackley, dim=5)
    # run_test(Sphere, dim=5)
    # run_test(Rastrigin, dim=5)
    # run_test(Alpine, dim=5)
    # run_test(Griewank, dim=5)
    # run_test(Michalewicz, dim=5)
    # run_test(Plateau, dim=5)
    # run_test(Quintic, dim=5)
    # run_test(Rosenbrock, dim=5)
    # run_test(Vincent, dim=5)
    # run_test(XinSheYang, dim=5)

    # Constraints / special cases
    # run_test(Bohachevsky, dim=2)
    # run_test(Eggholder, sol_shape = benchmark_shape_special)
    # run_test(Shubert, dim=2)

    # LotkaVolterra
    run_test(LotkaVolterra, sol_shape=LV_shape)
