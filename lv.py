import torch
from torchswarm.functions.lotka_volterra import LotkaVolterra 
from test_utils import run_test

# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    
    # Lotka-Volterra special (matrix params)
    LV_shape = torch.Size([4, 1])
    
    # LotkaVolterra
    run_test(LotkaVolterra, sol_shape=LV_shape)
