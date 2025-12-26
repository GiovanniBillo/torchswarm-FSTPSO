import torch
import torch.nn as nn

from .func_consts import PROBLEM_BOUNDS, rrs_batch, atot_batch, chlor_batch
from .base import Function


# ============================================================
# Forward model
# ============================================================

class CalculateReflectance(nn.Module):
    """
    Static forward model.

    params: (N, 9)
    returns: chlor_a_hat (N, B)
    """
    def forward(self, params, rrs_412, atot_412):
        """
        params:   (N, 9, C)   with C=1
        rrs_412:  (B,)
        atot_412: (B,)
        returns:  chlor_a_hat (N, B)
        """

        # ---------------------------
        # Shape & type checks
        # ---------------------------
        assert isinstance(params, torch.Tensor)
        assert params.ndim == 3 and params.shape[1] == 9, \
            f"params must be (N,9,C), got {params.shape}"

        N, _, C = params.shape
        assert C == 1, \
            f"Reflectance currently supports C=1 only, got C={C}"

        # 🔴 FIX: remove class dimension
        params = params.squeeze(-1)   # (N,9)

        assert params.shape == (N, 9)

        assert rrs_412.ndim == 1
        assert atot_412.ndim == 1
        assert rrs_412.shape == atot_412.shape

        B = rrs_412.shape[0]

        # ---------------------------
        # Unpack parameters (N,1)
        # ---------------------------
        bbp  = params[:, 0:1]
        atot = params[:, 1:2]

        g0 = params[:, 2:3]
        g1 = params[:, 3:4]

        a1 = params[:, 4:5]
        a2 = params[:, 5:6]
        a3 = params[:, 6:7]
        a4 = params[:, 7:8]
        a5 = params[:, 8:9]

        # ---------------------------
        # Physical constraints
        # ---------------------------
        assert torch.all(bbp > 0)
        assert torch.all(atot > 0)

        # ---------------------------
        # Broadcast observations
        # ---------------------------
        rrs_412  = rrs_412[None, :]   # (1, B)
        atot_412 = atot_412[None, :]  # (1, B)

        # ---------------------------
        # Reflectance model
        # ---------------------------
        u = bbp / (bbp + atot)        # (N,1)

        r_rs = g0 * u + g1 * u**2     # (N,1)
        r_rs = r_rs.expand(N, B)      # ✅ now legal

        eps = 1e-8
        r_rs = torch.clamp(r_rs, min=eps)

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

        assert chlor_a_hat.shape == (N, B)
        return chlor_a_hat

    # def forward(self, params, rrs_412, atot_412):
    #     """
    #     params:   (N, 9)
    #     rrs_412:  (B,)
    #     atot_412: (B,)
    #     """

    #     # ---------------------------
    #     # Shape & type checks
    #     # ---------------------------
    #     assert isinstance(params, torch.Tensor)
    #     # assert params.ndim == 2 and params.shape[1] == 9
    #     assert params.ndim == 3 and params.shape[1] == 9

    #     assert isinstance(rrs_412, torch.Tensor)
    #     assert isinstance(atot_412, torch.Tensor)

    #     assert rrs_412.ndim == 1
    #     assert atot_412.ndim == 1
    #     assert rrs_412.shape[0] == atot_412.shape[0]

    #     N = params.shape[0]
    #     B = rrs_412.shape[0]

    #     # ---------------------------
    #     # Unpack parameters (N,1)
    #     # ---------------------------
    #     bbp  = params[:, 0:1]
    #     atot = params[:, 1:2]

    #     g0 = params[:, 2:3]
    #     g1 = params[:, 3:4]

    #     a1 = params[:, 4:5]
    #     a2 = params[:, 5:6]
    #     a3 = params[:, 6:7]
    #     a4 = params[:, 7:8]
    #     a5 = params[:, 8:9]

    #     # ---------------------------
    #     # Physical constraints
    #     # ---------------------------
    #     assert torch.all(bbp > 0)
    #     assert torch.all(atot > 0)

    #     # ---------------------------
    #     # Broadcast observations
    #     # ---------------------------
    #     rrs_412  = rrs_412[None, :]   # (1, B)
    #     atot_412 = atot_412[None, :]  # (1, B)

    #     # ---------------------------
    #     # Reflectance model
    #     # ---------------------------
    #     u = bbp / (bbp + atot)        # (N,1)

    #     r_rs = g0 * u + g1 * u**2     # (N,1)
    #     r_rs = r_rs.expand(N, B)      # (N,B)

    #     eps = 1e-8
    #     r_rs = torch.clamp(r_rs, min=eps)

    #     log_r = torch.log10(r_rs)

    #     # ---------------------------
    #     # OCx polynomial
    #     # ---------------------------
    #     chlor_a_hat = (
    #         a1
    #         + a2 * log_r
    #         + a3 * log_r**2
    #         + a4 * log_r**3
    #         + a5 * log_r**4
    #     )

    #     assert chlor_a_hat.shape == (N, B)
    #     return chlor_a_hat


# ============================================================
# Fitness
# ============================================================

def reflectance_fitness(params, rrs_412, atot_412, chlor_gt):
    """
    params:    (N,9)
    rrs_412:   (B,)
    atot_412:  (B,)
    chlor_gt:  (B,)

    returns:
        fitness: (N,)
        chlor_hat: (N,B)
    """

    # ---------------------------
    # Checks
    # ---------------------------
    assert chlor_gt.ndim == 1
    B = chlor_gt.shape[0]

    model = CalculateReflectance()
    chlor_hat = model(params, rrs_412, atot_412)  # (N,B)

    # ---------------------------
    # Log-space loss (MANDATORY)
    # ---------------------------
    eps = 1e-8
    log_gt = torch.log10(torch.clamp(chlor_gt, min=eps))[None, :]  # (1,B)
    log_hat = torch.log10(torch.clamp(chlor_hat, min=eps))         # (N,B)

    mse = torch.mean((log_hat - log_gt) ** 2, dim=1)  # (N,)

    assert mse.ndim == 1
    assert mse.shape[0] == params.shape[0]

    return mse, chlor_hat


# ============================================================
# PSO wrapper
# ============================================================

class Reflectance(Function):
    """
    PSO-compatible reflectance calibration problem.
    """

    def __init__(self, rrs_412=rrs_batch, atot_412=atot_batch, chlor_gt=chlor_batch):
        self.name = self.__class__.__name__
        self.bounds = PROBLEM_BOUNDS[self.name]

        # Store data (full dataset or pre-batched)
        self.rrs_412 = rrs_412
        self.atot_412 = atot_412
        self.chlor_gt = chlor_gt

        assert self.rrs_412.ndim == 1
        assert self.atot_412.ndim == 1
        assert self.chlor_gt.ndim == 1
        assert len(self.rrs_412) == len(self.chlor_gt)

        self.chlor_a_hat = None

    def evaluate(self, params):
        """
        params: (N,9)
        returns: fitness (N,)
        """

        # assert params.ndim == 2 and params.shape[1] == 9, f"AssertionError: actual {params.ndim} and {params.shape}"

        assert params.ndim == 3 and params.shape[1] == 9, f"AssertionError: actual {params.ndim} and {params.shape}"

        fitness, self.chlor_a_hat = reflectance_fitness(
            params=params,
            rrs_412=self.rrs_412,
            atot_412=self.atot_412,
            chlor_gt=self.chlor_gt
        )

        return fitness

