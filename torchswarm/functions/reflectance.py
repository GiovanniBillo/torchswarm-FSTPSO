from .func_consts import PROBLEM_BOUNDS, TRUE_PARAMS

class CalculateReflectance(nn.Module):
    def __init__(self, params):
        """
        params: (N, 2, 2)
        """
        super().__init__()

        assert isinstance(params, torch.Tensor), \
            f"params must be torch.Tensor, got {type(params)}"
        assert params.ndim == 3 and params.shape[1:] == (2, 2), \
            f"params must have shape (N,2,2), got {params.shape}"

        self.params = params  # PSO updates externally

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

        bbp = self.params[:, 0]
        atot = self.params[:, 1]

        u_lambda = bbp/(atot + bbp) 
        
        g_0 = self.params[:,2]
        g_1 = self.params[:,3]

        r_rs = g_0 * u_lambda + g_1 * (u_lambda)**2 
        
        a1 = self.params[:, 4]
        a2 = self.params[:, 5]
        a3 = self.params[:, 6]
        a4 = self.params[:, 7]
        a5 = self.params[:, 8]

        chlor_a = a1 + a2*torch.log10(r_rs) + a3*torch.log10(r_rs) + a4*torch.log10(r_rs) + a5*torch.log10(r_rs) # from this paper: https://opg.optica.org/oe/fulltext.cfm?uri=oe-20-19-20920 
        out = chlor_a
        assert out.shape == state.shape, \
            f"ODE output must match state shape {state.shape}, got {out.shape}"

        return out

def solve_reflectance(params, y0, t):
    """
    params: (N, 2, 2)
    y0: (2,) or (1,2) or (N,2)
    t: (T,)
    returns: (T, N, 2)
    """
    assert isinstance(params, torch.Tensor), \
        f"params must be torch.Tensor, got {type(params)}"
    assert params.ndim == 3 and params.shape[1:] == (2, 2), \
        f"params must be (N,2,2), got {params.shape}"

    assert isinstance(t, torch.Tensor), \
        f"t must be torch.Tensor, got {type(t)}"
    assert t.ndim == 1, \
        f"t must be 1D tensor (T,), got {t.shape}"

    N = params.shape[0]

    assert isinstance(y0, torch.Tensor), \
        f"y0 must be torch.Tensor, got {type(y0)}"

    # --- normalize y0 ---
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

    ode_func = CalculateReflectance(params)
    sol = odeint(ode_func, y0, t)

    assert sol.ndim == 3 and sol.shape == (t.shape[0], N, 2), \
        f"solution must be (T,N,2), got {sol.shape}"

    return sol

# class solve_reflectance(Function):

#     def evaluate(self, pos):
        
#         gt = "Something you retrieve from the dataset itself" # i.e idk, initialize a particle swarm of the same size of the observations that were taken everyday and then compare them in order 
#         g_0 = 0.089
#         g_1 = 0.1245

#         bbp = pos[0]
#         atot = pos[1]
#         first = g_0*(bbp/(bbp + atot)) 
#         second = g_1*(bbp/(bbp + atot))**2
#         pred = first + second 

#         out = pred - gt 
#         return out.squeeze()

def reflectance_fitness(params, ground_truth):
    """
    params: (N, 2, 2)
    ground_truth: (T, 2)
    returns: (N,)
    """
    assert isinstance(ground_truth, torch.Tensor), \
        f"ground_truth must be torch.Tensor, got {type(ground_truth)}"
    assert ground_truth.ndim == 2 and ground_truth.shape[1] == 2, \
        f"ground_truth must be (T,2), got {ground_truth.shape}"

    sol = solve_reflectance(params, t)  # (T, N, 2)

    T, N, _ = sol.shape
    assert ground_truth.shape[0] == T, \
        f"ground_truth length {ground_truth.shape[0]} does not match solution time {T}"

    gt = ground_truth[:, None, :]  # (T,1,2)

    mse = torch.mean((sol - gt) ** 2, dim=(0, 2))  # (N,)

    assert mse.shape == (N,), \
        f"fitness must be (N,), got {mse.shape}"
    return mse, sol

class Reflectance(Function):

    def __init__(self, ground_truth=None, real_params=None):
        self.name = self.__class__.__name__
        self.bounds = PROBLEM_BOUNDS[self.name]
        gt = REAL_SOLUTIONS[self.name]

        self.ground_truth = (
            ground_truth if ground_truth is not None
            else gt[:, 0, :] 
        )

        print("GROUND TRUTH:", self.ground_truth)
        self.real_params = (
            real_params if real_params is not None
            else TRUE_PARAMS[self.name]
        )

        # self.initial_conditions = self.ground_truth[0].unsqueeze(0)
        self.initial_conditions = self.ground_truth[0]
        print("SHAPE OF INITIAL CONDITION:", self.initial_conditions)
        self.chlor_a_hat = None

    def evaluate(self, params):
        fit, self.chlor_a_hat = reflectance_fitness(
            params=params,
            y0=self.initial_conditions,
            ground_truth=self.ground_truth
        )
        return fit 
