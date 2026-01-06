import torch
import torch.nn as nn

from .func_consts import PROBLEM_BOUNDS, rrs_batch, atot_batch, chlor_batch, bbp_batch
from .func_consts import full_cols, batch_cols 
from .base import Function


# ============================================================
# Forward model
# ============================================================

class CalculateReflectance(nn.Module):
    """
    Static forward model.

    params: (N, 9, C) with C=1
    inputs: (B,)
    returns: chlor_a_hat (N, B)
    """
    def forward(self, params, rrs_412, atot_412, bbp_412):
        # ---------------------------
        # Shape & type checks
        # ---------------------------
        assert isinstance(params, torch.Tensor), f"params must be Tensor, got {type(params)}"
        assert params.ndim == 3 and params.shape[1] == 9, \
            f"params must be (N,9,C), got {params.shape}"

        N, D, C = params.shape
        assert C == 1, f"Expected C=1 for now, got C={C}"

        # squeeze class dim
        params = params.squeeze(-1)  # (N,9)
        assert params.shape == (N, 9)

        assert isinstance(rrs_412, torch.Tensor) and rrs_412.ndim == 1, "rrs_412 must be (B,)"
        assert isinstance(atot_412, torch.Tensor) and atot_412.ndim == 1, "atot_412 must be (B,)"
        assert isinstance(bbp_412, torch.Tensor) and bbp_412.ndim == 1, "bbp_412 must be (B,)"
        assert rrs_412.shape == atot_412.shape == bbp_412.shape, \
            f"Input shapes mismatch: {rrs_412.shape}, {atot_412.shape}, {bbp_412.shape}"

        B = rrs_412.shape[0]

        # ---------------------------
        # Unpack parameters
        # ---------------------------
        # We keep 9 params by interpreting first two as SCALE factors for observed IOPs
        bbp_scale  = params[:, 0:1]  # (N,1)
        atot_scale = params[:, 1:2]  # (N,1)

        g0 = params[:, 2:3]          # (N,1)
        g1 = params[:, 3:4]          # (N,1)

        a1 = params[:, 4:5]          # (N,1)
        a2 = params[:, 5:6]
        a3 = params[:, 6:7]
        a4 = params[:, 7:8]
        a5 = params[:, 8:9]

        # ---------------------------
        # Constraints
        # ---------------------------
        assert torch.all(bbp_scale > 0),  "bbp_scale must be > 0"
        assert torch.all(atot_scale > 0), "atot_scale must be > 0"

        # ---------------------------
        # Broadcast observations
        # ---------------------------
        bbp_412  = bbp_412[None, :]    # (1,B)
        atot_412 = atot_412[None, :]   # (1,B)

        # Apply global scaling (N,1) * (1,B) -> (N,B)
        bbp_eff  = bbp_scale  * bbp_412
        atot_eff = atot_scale * atot_412

        # ---------------------------
        # Reflectance proxy model
        # ---------------------------
        eps = 1e-12
        denom = torch.clamp(atot_eff + bbp_eff, min=eps)
        u = bbp_eff / denom                 # (N,B)

        r_rs = g0 * u + g1 * (u ** 2)       # (N,B)

        # Ensure positivity for log10
        r_rs = torch.clamp(r_rs, min=1e-12)
        log_r = torch.log10(r_rs)

        # ---------------------------
        # OCx polynomial
        # ---------------------------
        chlor_a_hat = (
            a1
            + a2 * log_r
            + a3 * log_r**2
            + a4 * log_r**3
            + a5 * log_r**4
        )

        assert chlor_a_hat.shape == (N, B), f"Expected (N,B), got {chlor_a_hat.shape}"
        return chlor_a_hat


# ============================================================
# Fitness
# ============================================================

def reflectance_fitness(params, rrs_412, atot_412, bbp_412, chlor_gt):
    """
    params:    (N,9,C)
    rrs_412:   (B,)
    atot_412:  (B,)
    bbp_412:   (B,)
    chlor_gt:  (B,)

    returns:
        fitness:   (N,)
        chlor_hat: (N,B)
    """
    assert isinstance(chlor_gt, torch.Tensor) and chlor_gt.ndim == 1, "chlor_gt must be (B,)"

    model = CalculateReflectance()
    chlor_hat = model(params, rrs_412, atot_412, bbp_412)  # (N,B)

    # Log-space MSE (mandatory for Chl)
    eps = 1e-12
    log_gt  = torch.log10(torch.clamp(chlor_gt,  min=eps))[None, :]  # (1,B)
    log_hat = torch.log10(torch.clamp(chlor_hat, min=eps))           # (N,B)

    mse = torch.mean((log_hat - log_gt) ** 2, dim=1)  # (N,)
    assert mse.shape[0] == params.shape[0], f"fitness shape mismatch: {mse.shape} vs N={params.shape[0]}"

    return mse, chlor_hat


# ============================================================
# PSO wrapper
# ============================================================

class Reflectance(Function):
    """
    PSO-compatible reflectance calibration problem.
    """

    def __init__(self, rrs_412=rrs_batch, atot_412=atot_batch, bbp_412=bbp_batch, chlor_gt=chlor_batch):
        self.name = self.__class__.__name__
        self.bounds = PROBLEM_BOUNDS[self.name]

        # Store batch data
        self.rrs_412  = rrs_412
        self.atot_412 = atot_412
        self.bbp_412  = bbp_412
        self.chlor_gt = chlor_gt

        # Checks
        assert self.rrs_412.ndim == 1
        assert self.atot_412.ndim == 1
        assert self.bbp_412.ndim == 1
        assert self.chlor_gt.ndim == 1
        assert len(self.rrs_412) == len(self.atot_412) == len(self.bbp_412) == len(self.chlor_gt), \
            "All input arrays must have same length (B,)"

        self.chlor_a_hat = None

    def evaluate(self, params):
        """
        params: (N,9,C)
        returns: fitness (N,)
        """
        assert isinstance(params, torch.Tensor)
        assert params.ndim == 3 and params.shape[1] == 9, \
            f"params must be (N,9,C), got {params.shape}"

        fitness, self.chlor_a_hat = reflectance_fitness(
            params=params,
            rrs_412=self.rrs_412,
            atot_412=self.atot_412,
            bbp_412=self.bbp_412,
            chlor_gt=self.chlor_gt
        )
        return fitness
