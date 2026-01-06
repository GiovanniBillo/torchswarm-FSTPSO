
import torch
from torchswarm.functions.ocx import OCxChlorophyll
from test_utils import run_test

# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    
    ocx_shape = torch.Size([5, 1])
    run_test(OCxChlorophyll, sol_shape=ocx_shape)
