from torchswarm.functions.benchmarks import Function

import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import odeint as scipy_odeint
from torchdiffeq import odeint  # torchdiffeq odeint

from .func_consts import PROBLEM_BOUNDS, TRUE_PARAMS


# ============================================================
# SciPy RHS (used by LotkaVolterraSafe)
# ============================================================

def lotka_volterra(state, t, params):
    """
    state: (2,) numpy array-like
    params: (4,) or (4,1) or (2,2) numpy array-like

    returns: (2,)
    """
    params = np.asarray(params)

    # accept legacy shapes
    if params.shape == (2, 2):
        r = params[0, 0]
        a = params[0, 1]
        b = params[1, 0]
        z = params[1, 1]
    else:
        params = params.reshape(-1)
        assert params.shape[0] == 4, f"Expected 4 params, got {params.shape}"
        r, a, b, z = params

    X, Y = state[0], state[1]
    dxdt = r * X - a * X * Y
    dydt = b * X * Y - z * Y
    return np.array([dxdt, dydt], dtype=np.float64)


# ============================================================
# Torchdiffeq RHS (parallel)
# ============================================================

class LotkaVolterraODE(nn.Module):
    def __init__(self, params):
        """
        params: (N, 4, 1) or (N, 4)
        """
        super().__init__()

        assert isinstance(params, torch.Tensor), \
            f"params must be torch.Tensor, got {type(params)}"

        if params.ndim == 3:
            assert params.shape[1] == 4 and params.shape[2] == 1, \
                f"params must have shape (N,4,1), got {params.shape}"
            self.params = params.squeeze(-1)  # (N,4)
        elif params.ndim == 2:
            assert params.shape[1] == 4, \
                f"params must have shape (N,4), got {params.shape}"
            self.params = params
        else:
            raise AssertionError(f"params must be (N,4,1) or (N,4), got {params.shape}")

    def forward(self, t, state):
        """
        state: (N, 2)
        returns: (N, 2)
        """
        assert isinstance(state, torch.Tensor), \
            f"state must be torch.Tensor, got {type(state)}"
        assert state.ndim == 2 and state.shape[1] == 2, \
            f"state must have shape (N,2), got {state.shape}"

        N = self.params.shape[0]
        assert state.shape[0] == N, \
            f"state batch size {state.shape[0]} does not match params batch size {N}"

        # unpack params
        r = self.params[:, 0]
        a = self.params[:, 1]
        b = self.params[:, 2]
        z = self.params[:, 3]

        X = state[:, 0]
        Y = state[:, 1]

        dxdt = r * X - a * X * Y
        dydt = b * X * Y - z * Y

        out = torch.stack([dxdt, dydt], dim=1)
        assert out.shape == state.shape, \
            f"ODE output must match state shape {state.shape}, got {out.shape}"
        return out


def solve_lotka_volterra(params, y0, t):
    """
    params: (N,4,1) or (N,4)
    y0: (2,) or (1,2) or (N,2)
    t: (T,)
    returns: (T, N, 2)
    """
    assert isinstance(params, torch.Tensor), \
        f"params must be torch.Tensor, got {type(params)}"

    if params.ndim == 3:
        assert params.shape[1:] == (4, 1), \
            f"params must be (N,4,1), got {params.shape}"
        N = params.shape[0]
    elif params.ndim == 2:
        assert params.shape[1] == 4, \
            f"params must be (N,4), got {params.shape}"
        N = params.shape[0]
    else:
        raise AssertionError(f"params must be (N,4,1) or (N,4), got {params.shape}")

    assert isinstance(t, torch.Tensor), \
        f"t must be torch.Tensor, got {type(t)}"
    assert t.ndim == 1, \
        f"t must be 1D tensor (T,), got {t.shape}"

    assert isinstance(y0, torch.Tensor), \
        f"y0 must be torch.Tensor, got {type(y0)}"

    # --- normalize y0 to (N,2) ---
    if y0.ndim == 1:
        assert y0.shape[0] == 2, \
            f"y0 with ndim=1 must have shape (2,), got {y0.shape}"
        y0 = y0.unsqueeze(0).expand(N, 2)

    elif y0.ndim == 2:
        assert y0.shape[1] == 2, \
            f"y0 with ndim=2 must have shape (*,2), got {y0.shape}"
        if y0.shape[0] == 1:
            y0 = y0.expand(N, 2)
        else:
            assert y0.shape[0] == N, \
                f"y0 batch size {y0.shape[0]} does not match params batch size {N}"
    else:
        raise AssertionError(
            f"y0 must have shape (2,), (1,2) or (N,2); got {y0.shape}"
        )

    assert y0.shape == (N, 2), \
        f"y0 must be normalized to (N,2), got {y0.shape}"
    device = params.device
    dtype = params.dtype

    t = t.to(device=device, dtype=dtype)
    y0 = y0.to(device=device, dtype=dtype)

    ode_func = LotkaVolterraODE(params)
    sol = odeint(ode_func, y0, t)

    assert sol.ndim == 3 and sol.shape == (t.shape[0], N, 2), \
        f"solution must be (T,N,2), got {sol.shape}"

    return sol


def lotka_volterra_fitness(params, y0, t, ground_truth):
    """
    params: (N,4,1) or (N,4)
    ground_truth: (T,2)
    returns: (N,)
    """
    assert isinstance(ground_truth, torch.Tensor), \
        f"ground_truth must be torch.Tensor, got {type(ground_truth)}"
    assert ground_truth.ndim == 2 and ground_truth.shape[1] == 2, \
        f"ground_truth must be (T,2), got {ground_truth.shape}"

    sol = solve_lotka_volterra(params, y0, t)  # (T,N,2)

    T, N, _ = sol.shape
    assert ground_truth.shape[0] == T, \
        f"ground_truth length {ground_truth.shape[0]} does not match solution time {T}"

    gt = ground_truth[:, None, :]  # (T,1,2)
    mse = torch.mean((sol - gt) ** 2, dim=(0, 2))  # (N,)

    assert mse.shape == (N,), \
        f"fitness must be (N,), got {mse.shape}"
    return mse


# ============================================================
# Ground truth generation (torch)
# ============================================================
with torch.no_grad():
    t = torch.linspace(0, 100, 100, device="cpu")
    initial_conditions = torch.tensor([30.0, 10.0], device="cpu")
    solution = solve_lotka_volterra(TRUE_PARAMS["LotkaVolterra"].cpu(), initial_conditions, t).cpu()
    REAL_SOLUTIONS = {"LotkaVolterra": solution}

# t = torch.linspace(0, 100, 100)
# initial_conditions = torch.tensor([30.0, 10.0])

# solution = solve_lotka_volterra(TRUE_PARAMS["LotkaVolterra"], initial_conditions, t)
# REAL_SOLUTIONS = {"LotkaVolterra": solution}


# ============================================================
# Safe SciPy version (kept for debugging)
# ============================================================

class LotkaVolterraSafe(Function):
    def __init__(self, ground_truth=None, real_params=None):
        self.name = self.__class__.__name__
        self.bounds = PROBLEM_BOUNDS[self.name]

        self.ground_truth = (
            ground_truth if ground_truth is not None
            else REAL_SOLUTIONS["LotkaVolterra"].detach().cpu().numpy()
        )

        self.real_params = (
            real_params if real_params is not None
            else TRUE_PARAMS["LotkaVolterra"].detach().cpu().numpy()
        )

        self.t = np.linspace(0, 100, 100)
        # for scipy, initial conditions as (2,)
        self.initial_conditions = self.ground_truth[0]  # (2,)

    def evaluate(self, params):
        """
        params: (N,4,1) or (N,4) or legacy (N,2,2)
        returns: (N,)
        """
        if isinstance(params, torch.Tensor):
            params = params.detach().cpu().numpy()

        assert params.ndim in (2, 3), f"Expected params ndim 2 or 3, got {params.shape}"
        N = params.shape[0]
        fitness = np.empty(N, dtype=np.float64)

        for i in range(N):
            try:
                sol = scipy_odeint(
                    lotka_volterra,
                    self.initial_conditions,
                    self.t,
                    args=(params[i],)
                )
                fitness[i] = np.mean((sol - self.ground_truth) ** 2)
            except Exception as e:
                print(f"[{self.name}] ODE failure at particle {i}: {e}")
                fitness[i] = 1e12

        return torch.tensor(fitness, dtype=torch.float32)


# ============================================================
# Main LV problem (parallel torchdiffeq)
# ============================================================

class LotkaVolterra(Function):
    def __init__(self, ground_truth=None, real_params=None):
        self.name = self.__class__.__name__
        self.bounds = PROBLEM_BOUNDS[self.name]

        gt = REAL_SOLUTIONS["LotkaVolterra"]  # (T,N,2) with N=1 for true params
        self.ground_truth = (
            ground_truth if ground_truth is not None
            else gt[:, 0, :]  # (T,2)
        )

        self.real_params = (
            real_params if real_params is not None
            else TRUE_PARAMS["LotkaVolterra"]
        )

        self.t = torch.linspace(0, 100, 100)
        self.initial_conditions = self.ground_truth[0]  # (2,)

        # Assertions
        assert isinstance(self.ground_truth, torch.Tensor)
        assert self.ground_truth.ndim == 2 and self.ground_truth.shape[1] == 2, \
            f"ground_truth must be (T,2), got {self.ground_truth.shape}"
        assert isinstance(self.initial_conditions, torch.Tensor)
        assert self.initial_conditions.shape == (2,), \
            f"initial_conditions must be (2,), got {self.initial_conditions.shape}"

    def evaluate(self, params):
        """
        params: (N,4,1) or (N,4)
        returns: (N,)
        """
        device = params.device
        dtype = params.dtype
        return lotka_volterra_fitness(
            params=params,
            y0=self.initial_conditions.to(device=device, dtype=dtype),
            t=self.t.to(device=device, dtype=dtype),
            ground_truth=self.ground_truth.to(device=device, dtype=dtype),
        )
