# Integration Test
import torch

from torchswarm.swarmoptimizer import SwarmOptimizer
from torchswarm.functions.benchmarks import Ackley, Sphere

class CubicFunction:
    def evaluate(self, x):
        return x ** 2 + torch.exp(x)

# empso = SwarmOptimizer(1, 100, swarm_optimizer_type="exponentially_weighted", max_iterations=10)
# empso.optimize(CubicFunction())

# print(empso.run(verbosity=True).__dict__)
empso = SwarmOptimizer(2, 100, swarm_optimizer_type="standard", max_iterations=10)
empso.optimize(Ackley())

print("results for the Ackley run:")
print(empso.run(verbosity=True).__dict__)
