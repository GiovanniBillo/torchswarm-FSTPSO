import torch
import torch.nn as nn

from .func_consts import PROBLEM_BOUNDS, batch_cols
from .base import Function


# ============================================================
# Forward model: OCx in log-space
# ============================================================

class CalculateOCx(nn.Module):
    """
    OCx polynomial in log10-space.

    params: (N, 5, C) with C=1  (or (N,5))
    inputs: Rrs bands (B,)
    returns: log10(chl_hat) (N, B)
    """

    def __init__(self, use_412: bool = False):
        super().__init__()
        self.use_412 = use_412

    def forward(self, params, rrs_443, rrs_490, rrs_510, rrs_560, rrs_412=None):
        # ---------------------------
        # params checks
        # ---------------------------
        assert isinstance(params, torch.Tensor), f"params must be Tensor, got {type(params)}"

        if params.ndim == 3:
            assert params.shape[1] == 5, f"params must be (N,5,C), got {params.shape}"
            N, D, C = params.shape
            assert C == 1, f"Expected C=1 for now, got C={C}"
            params = params.squeeze(-1)  # (N,5)
        elif params.ndim == 2:
            assert params.shape[1] == 5, f"params must be (N,5), got {params.shape}"
            N = params.shape[0]
        else:
            raise AssertionError(f"params must be (N,5) or (N,5,1), got {params.shape}")

        # ---------------------------
        # inputs checks
        # ---------------------------
        for name, x in [("rrs_443", rrs_443), ("rrs_490", rrs_490), ("rrs_510", rrs_510), ("rrs_560", rrs_560)]:
            assert isinstance(x, torch.Tensor) and x.ndim == 1, f"{name} must be (B,), got {type(x)} {getattr(x,'shape',None)}"

        B = rrs_560.shape[0]
        assert rrs_443.shape == rrs_490.shape == rrs_510.shape == rrs_560.shape, "All Rrs inputs must share shape (B,)"

        if self.use_412:
            assert rrs_412 is not None, "use_412=True but rrs_412 is None"
            assert isinstance(rrs_412, torch.Tensor) and rrs_412.ndim == 1 and rrs_412.shape == rrs_560.shape, \
                f"rrs_412 must be (B,) and match others, got {getattr(rrs_412,'shape',None)}"

        # ---------------------------
        # compute X = log10(max_blue / green)
        # ---------------------------
        eps = 1e-12

        if self.use_412:
            blue = torch.maximum(torch.maximum(rrs_443, rrs_490), torch.maximum(rrs_510, rrs_412))
        else:
            blue = torch.maximum(torch.maximum(rrs_443, rrs_490), rrs_510)

        blue = torch.clamp(blue, min=eps)
        green = torch.clamp(rrs_560, min=eps)

        ratio = blue / green                     # (B,)
        X = torch.log10(torch.clamp(ratio, min=eps))  # (B,)

        # ---------------------------
        # polynomial: log10(chl)
        # ---------------------------
        a0 = params[:, 0:1]
        a1 = params[:, 1:2]
        a2 = params[:, 2:3]
        a3 = params[:, 3:4]
        a4 = params[:, 4:5]

        Xb = X[None, :]  # (1,B)

        log_chl_hat = (
            a0
            + a1 * Xb
            + a2 * Xb**2
            + a3 * Xb**3
            + a4 * Xb**4
        )

        assert log_chl_hat.shape == (N, B), f"Expected (N,B), got {log_chl_hat.shape}"
        return log_chl_hat


# ============================================================
# Fitness: log-space MSE
# ============================================================

def ocx_fitness(params, rrs_443, rrs_490, rrs_510, rrs_560, chlor_gt, use_412=False, rrs_412=None):
    """
    params:   (N,5,1) or (N,5)
    inputs:   Rrs bands (B,)
    target:   chlor_gt (B,)
    returns:
      fitness (N,)
      log_chl_hat (N,B)
    """
    assert isinstance(chlor_gt, torch.Tensor) and chlor_gt.ndim == 1, f"chlor_gt must be (B,), got {getattr(chlor_gt,'shape',None)}"

    model = CalculateOCx(use_412=use_412)
    log_chl_hat = model(params, rrs_443, rrs_490, rrs_510, rrs_560, rrs_412=rrs_412)

    eps = 1e-12
    log_gt = torch.log10(torch.clamp(chlor_gt, min=eps))[None, :]  # (1,B)

    mse = torch.mean((log_chl_hat - log_gt) ** 2, dim=1)  # (N,)
    assert mse.shape[0] == (params.shape[0] if params.ndim >= 2 else 0)
    return mse, log_chl_hat


# ============================================================
# PSO wrapper
# ============================================================

class OCxChlorophyll(Function):
    """
    Calibrate OCx coefficients [a0..a4] using Rrs bands -> chlor_a.

    Expected to be run with batches: B ~ 10k-100k, while swarm N is heuristic-sized.
    """

    def __init__(self, batch_cols: dict = batch_cols, use_412: bool = False):
        self.name = self.__class__.__name__
        self.bounds = PROBLEM_BOUNDS.get("OCx", PROBLEM_BOUNDS.get(self.name))

        assert isinstance(batch_cols, dict), "batch_cols must be a dict of torch tensors"
        self.use_412 = use_412

        # Required columns (you said you have these)
        required = ["rrs_443", "rrs_490", "rrs_510", "rrs_560", "chlor_a"]
        missing = [k for k in required if k not in batch_cols]
        assert not missing, f"Missing required columns in batch_cols: {missing}"

        if self.use_412:
            assert "rrs_412" in batch_cols, "use_412=True but batch_cols lacks 'rrs_412'"

        # Store
        self.rrs_443 = batch_cols["rrs_443"]
        self.rrs_490 = batch_cols["rrs_490"]
        self.rrs_510 = batch_cols["rrs_510"]
        self.rrs_560 = batch_cols["rrs_560"]
        self.chlor_gt = batch_cols["chlor_a"]
        self.rrs_412 = batch_cols["rrs_412"] if self.use_412 else None

        # Shape checks
        B = self.chlor_gt.shape[0]
        for name, x in [("rrs_443", self.rrs_443), ("rrs_490", self.rrs_490), ("rrs_510", self.rrs_510), ("rrs_560", self.rrs_560), ("chlor_a", self.chlor_gt)]:
            assert isinstance(x, torch.Tensor) and x.ndim == 1, f"{name} must be 1D"
            assert x.shape[0] == B, f"{name} length mismatch: {x.shape[0]} vs {B}"

        # Optional: filter invalid values once (recommended)
        # You can keep it simple and just rely on clamp(eps), but filtering helps.
        eps = 1e-12
        mask = (
            torch.isfinite(self.rrs_443) &
            torch.isfinite(self.rrs_490) &
            torch.isfinite(self.rrs_510) &
            torch.isfinite(self.rrs_560) &
            torch.isfinite(self.chlor_gt) &
            (self.rrs_560 > eps) &
            (self.chlor_gt > eps) &
            ((self.rrs_443 > eps) | (self.rrs_490 > eps) | (self.rrs_510 > eps))
        )
        if self.use_412:
            mask = mask & torch.isfinite(self.rrs_412) & (self.rrs_412 > eps)

        # apply mask
        self.rrs_443 = self.rrs_443[mask]
        self.rrs_490 = self.rrs_490[mask]
        self.rrs_510 = self.rrs_510[mask]
        self.rrs_560 = self.rrs_560[mask]
        self.chlor_gt = self.chlor_gt[mask]
        if self.use_412:
            self.rrs_412 = self.rrs_412[mask]

        self.log_chl_hat = None

    def evaluate(self, params):
        """
        params: (N,5,1) 
        returns: fitness (N,)
        """
        assert isinstance(params, torch.Tensor), "params must be torch.Tensor"
        assert (params.ndim == 3 and params.shape[1] == 5) or (params.ndim == 2 and params.shape[1] == 5), \
            f"params must be (N,5,1) or (N,5), got {params.shape}"

        # Move data to same device/dtype as params
        device = params.device
        dtype = params.dtype

        r443 = self.rrs_443.to(device=device, dtype=dtype)
        r490 = self.rrs_490.to(device=device, dtype=dtype)
        r510 = self.rrs_510.to(device=device, dtype=dtype)
        r560 = self.rrs_560.to(device=device, dtype=dtype)
        chl  = self.chlor_gt.to(device=device, dtype=dtype)
        r412 = self.rrs_412.to(device=device, dtype=dtype) if self.use_412 else None

        fitness, self.log_chl_hat = ocx_fitness(
            params=params,
            rrs_443=r443,
            rrs_490=r490,
            rrs_510=r510,
            rrs_560=r560,
            chlor_gt=chl,
            use_412=self.use_412,
            rrs_412=r412
        )
        return fitness

