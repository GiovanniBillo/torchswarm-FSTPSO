import torch
from torchswarm.functions.reflectance_correct import Reflectance 

from test_utils import run_test

# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    
    reflectance_shape = torch.Size([9, 1])
    ## we aim to infer all the parameters for a single wavelength: 
    ## - backscattering coefficient bbp
    ## - total absorptio atot
    ## - g_0  (factor constants)
    ## - g_1 (factor constants
    ## - a1, a2, a3, a4, a5 (final weights in determining the chlorophilla)

    ## in reality, we have some of these parameters already from data, so there is potential for a sort of "multilayer problem".
    ## also, there are 5 more wavelengths to determine, potentially

    run_test(Reflectance, sol_shape=reflectance_shape)
