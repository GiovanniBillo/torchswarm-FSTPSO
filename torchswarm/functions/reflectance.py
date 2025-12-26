import torch
import torch.nn as nn

from .func_consts import PROBLEM_BOUNDS, TRUE_PARAMS, REAL_SOLUTIONS
from .base import Function   # assuming same base class as LV

class CalculateReflectance(nn.Module):
    """
    Static forward model:
        params (N,9) -> chlor_a_hat (N,)
    """

    def __init__(self):
        super().__init__()

    def forward(self, params):
        """
        params: (N, 9)
        returns: (N,)
        """

        # -------- type & shape checks --------
        assert isinstance(params, torch.Tensor), \
            f"params must be torch.Tensor, got {type(params)}"
        assert params.ndim == 2 and params.shape[1] == 9, \
            f"params must have shape (N,9), got {params.shape}"

        # -------- unpack parameters --------
        bbp  = params[:, 0]
        atot = params[:, 1]

        g0 = params[:, 2]
        g1 = params[:, 3]

        a1 = params[:, 4]
        a2 = params[:, 5]
        a3 = params[:, 6]
        a4 = params[:, 7]
        a5 = params[:, 8]

        # -------- physical constraints --------
        assert torch.all(bbp > 0),  "bbp must be > 0"
        assert torch.all(atot > 0), "atot must be > 0"

        # -------- reflectance model --------
        u_lambda = bbp / (atot + bbp)

        r_rs = g0 * u_lambda + g1 * u_lambda**2

        assert torch.all(r_rs > 0), \
            "r_rs must be positive to apply log10"

        log_r = torch.log10(r_rs)

        # -------- OCx polynomial --------
        chlor_a_hat = (
            a1
            + a2 * log_r
            + a3 * log_r**2
            + a4 * log_r**3
            + a5 * log_r**4
        )

        assert chlor_a_hat.ndim == 1 and chlor_a_hat.shape[0] == params.shape[0], \
            f"output must be (N,), got {chlor_a_hat.shape}"

        return chlor_a_hat

def reflectance_fitness(params, ground_truth):
    """
    params: (N,9)
    ground_truth: (N,) or scalar
    returns: (N,)
    """

    # -------- checks --------
    assert isinstance(ground_truth, torch.Tensor), \
        f"ground_truth must be torch.Tensor, got {type(ground_truth)}"

    if ground_truth.ndim == 0:
        ground_truth = ground_truth.expand(params.shape[0])

    assert ground_truth.ndim == 1, \
        f"ground_truth must be (N,), got {ground_truth.shape}"

    assert params.shape[0] == ground_truth.shape[0], \
            f"params and ground_truth batch size mismatch: {params.shape} != {ground_truth.shape}"

    model = CalculateReflectance()
    chlor_a_hat = model(params)

    # -------- MSE per particle --------
    mse = (chlor_a_hat - ground_truth)**2

    assert mse.shape == (params.shape[0],), \
        f"fitness must be (N,), got {mse.shape}"

    return mse, chlor_a_hat

class Reflectance(Function):
    """
    PSO-compatible reflectance inversion problem
    """

    def __init__(self, ground_truth=None, real_params=None):
        self.name = self.__class__.__name__
        self.bounds = PROBLEM_BOUNDS[self.name]

        self.ground_truth = (
            ground_truth
            if ground_truth is not None
            else REAL_SOLUTIONS[self.name]
        )

        assert isinstance(self.ground_truth, torch.Tensor), \
            "ground_truth must be torch.Tensor"

        # self.real_params = (
        #     real_params
        #     if real_params is not None
        #     else TRUE_PARAMS[self.name]["params"]
        # )

        self.chlor_a_hat = None

    def evaluate(self, params):
        """
        params: (N,9)
        returns: (N,)
        """
        assert params.shape == torch.Tensor([9, 1]), f"AssertionError:shape should be [9, 1] but is {params.shape}" 

        fitness, self.chlor_a_hat = reflectance_fitness(
            params=params,
            ground_truth=self.ground_truth
        )

        return fitness

