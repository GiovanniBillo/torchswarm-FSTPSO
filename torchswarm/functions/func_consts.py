import torch
import numpy as np

from torchswarm.data.data_utils import get_bounds

NPZ_PATH = "torchswarm/data/all_reflectances.npz"
# NPZ_PATH = 'torchswarm/data/esa_oc_412_valid_samples.npz'

# -----------------------------
# Utilities
# -----------------------------
def load_npz_as_torch(
    npz_path: str,
    batch_size: int = 20_000,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    exclude_keys: set[str] | None = None,
    seed: int = 0,
):
    """
    Load an .npz and return:
      - all columns as torch tensors (full length)
      - a consistent random batch across all 1D columns
      - the batch indices

    This is robust to "whatever the names": it loads every npz key except excluded ones,
    and batches all 1D arrays that share the same length M.

    Returns:
      full_tensors: dict[str, torch.Tensor]
      batch_tensors: dict[str, torch.Tensor]
      idx: torch.LongTensor
      M: int
    """
    exclude_keys = exclude_keys or set()
    data = np.load(npz_path, allow_pickle=True)

    # Keys that are not numeric vectors, or you don't want batched
    default_exclude = {"vars", "rules", "source", "product", "file"}
    exclude_keys = set(exclude_keys) | default_exclude

    # Convert numpy arrays -> torch tensors where possible
    full_tensors: dict[str, torch.Tensor] = {}
    lengths: dict[str, int] = {}

    for k in data.files:
        if k in exclude_keys:
            continue

        arr = data[k]

        # Skip non-numeric / object arrays (e.g. metadata)
        if not hasattr(arr, "dtype") or arr.dtype == object:
            continue

        # We primarily support 1D numeric arrays (your "columns")
        if arr.ndim != 1:
            # Keep it if you want, but don't include in batching logic
            # (lat/lon are 1D in your pipeline anyway)
            full_tensors[k] = torch.as_tensor(arr, dtype=dtype, device=device)
            continue

        t = torch.as_tensor(arr, dtype=dtype, device=device)
        full_tensors[k] = t
        lengths[k] = t.shape[0]

    if not lengths:
        raise ValueError(
            f"No 1D numeric columns found in {npz_path}. "
            f"Available keys: {data.files}"
        )

    # Determine the "main" length M: the most common length across columns
    # (lat/lon and data columns should all match this)
    from collections import Counter
    length_counts = Counter(lengths.values())
    M = length_counts.most_common(1)[0][0]

    # Columns eligible for batching: 1D tensors with length M
    batch_keys = [k for k, L in lengths.items() if L == M]
    if not batch_keys:
        raise ValueError("No columns share a common length for batching.")

    if batch_size > M:
        raise ValueError(f"batch_size={batch_size} > M={M}")

    # Stable randomness if desired
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    idx = torch.randperm(M, generator=g)[:batch_size]

    batch_tensors = {k: full_tensors[k][idx] for k in batch_keys}

    return full_tensors, batch_tensors, idx, M


# -----------------------------
# Load everything + make a batch
# -----------------------------
batch_size = 5000 
full_cols, batch_cols, idx, M = load_npz_as_torch(
    NPZ_PATH,
    batch_size=batch_size,
    device="cpu",
    dtype=torch.float32,
    exclude_keys={"time_index"},  # add if present
    seed=0,
)

print("Total samples:", M)
print("Loaded columns (full):", sorted(full_cols.keys()))
print("Batched columns:", sorted(batch_cols.keys()))
print("Batch size:", idx.numel())

# Expose your old variables if you still want them explicitly:
# (Only if present in file)
rrs_batch   = batch_cols.get("rrs_412") or batch_cols.get("rrs-412")  # just in case
atot_batch  = batch_cols.get("atot_412")
bbp_batch   = batch_cols.get("bbp_412")
chlor_batch = batch_cols.get("chlor_a")

# -----------------------------
# Bounds + constants
# -----------------------------
PROBLEM_BOUNDS = {
    "LotkaVolterra": (0.0001, 0.2),
    # If you want bounds computed from the file automatically for all columns:
    # get_bounds should be updated to accept columns=None -> infer all numeric 1D keys
    # "Reflectance": get_bounds(NPZ_PATH, columns=None), # TODO
}

PROBLEM_BOUNDS["Reflectance"] = {
    "bbp":  (0.5, 2.0),     # scale
    "atot": (0.5, 2.0),     # scale
    "g0":   (0.05, 0.15),
    "g1":   (0.05, 0.20),
    "a1":   (-2.0, 2.0),
    "a2":   (-4.0, 4.0),
    "a3":   (-4.0, 4.0),
    "a4":   (-4.0, 4.0),
    "a5":   (-4.0, 4.0),
}

PROBLEM_BOUNDS["OCx"] = {
    "a0": (-2.0, 2.0),
    "a1": (-5.0, 5.0),
    "a2": (-5.0, 5.0),
    "a3": (-5.0, 5.0),
    "a4": (-5.0, 5.0),
}

TRUE_PARAMS = {
    "LotkaVolterra": torch.tensor([[0.1, 0.02], [0.01, 0.1]], dtype=torch.float32).unsqueeze(0),
    "Reflectance": torch.tensor([]),
}

initial_conditions = torch.tensor([30.0, 10.0], dtype=torch.float32)

# Your "real solutions" mapping:
# For reflectance/OC problems, ground truth is the batch chlorophyll vector
REAL_SOLUTIONS = {
    "LotkaVolterra": None,
    "Reflectance": batch_cols.get("chlor_a"),  # <-- batch GT ready for fitness
}

if __name__ == "__main__":
    print("REAL_SOLUTIONS keys:", REAL_SOLUTIONS.keys())
    print("PROBLEM_BOUNDS keys:", PROBLEM_BOUNDS.keys())

# import torch
# import numpy as np

# from torchswarm.data.data_utils import get_bounds 
# NPZ_PATH = 'torchswarm/data/all_reflectances.npz'
# # inverse_chlor_a_solution, inverse_chlor_a_problem_bounds = get_esa_metadata(wavelength=lam) 
# PROBLEM_BOUNDS = {
#         "LotkaVolterra": (0.0001, 0.2), ## how can one bound be if these represent growth rate, death rate and stuff like that??
#         "Reflectance":  get_bounds(NPZ_PATH, columns=['chlor_a', 'rrs_412', 'atot_412']) # returns a dictionary of tuples

#     # "Reflectance":      ['bounds for backscattering'], ['bounds for absorption'] # if any
#         }
# # PROBLEM_BOUNDS["Reflectance"] = {
# #     # scale factors applied to observed bbp_412 and atot_412
# #     "bbp":  (0.1, 10.0),     # bbp_scale
# #     "atot": (0.1, 10.0),     # atot_scale

# #     # r_rs = g0*u + g1*u^2
# #     "g0":   (0.05, 0.15),
# #     "g1":   (0.05, 0.20),

# #     # OCx polynomial coefficients (log10 space)
# #     "a1":   (-2.0,  2.0),
# #     "a2":   (-6.0,  6.0),
# #     "a3":   (-6.0,  6.0),
# #     "a4":   (-6.0,  6.0),
# #     "a5":   (-6.0,  6.0),
# # }
# PROBLEM_BOUNDS["Reflectance"] = {
#     "bbp":  (0.5, 2.0),     # scale
#     "atot": (0.5, 2.0),     # scale
#     "g0":   (0.05, 0.15),
#     "g1":   (0.05, 0.20),
#     "a1":   (-2.0, 2.0),
#     "a2":   (-4.0, 4.0),
#     "a3":   (-4.0, 4.0),
#     "a4":   (-4.0, 4.0),
#     "a5":   (-4.0, 4.0),
# }

# PROBLEM_BOUNDS["OCx"] = {
#     "a0": (-2.0, 2.0),
#     "a1": (-5.0, 5.0),
#     "a2": (-5.0, 5.0),
#     "a3": (-5.0, 5.0),
#     "a4": (-5.0, 5.0),
# }

# TRUE_PARAMS={"LotkaVolterra": torch.Tensor([[0.1, 0.02],[0.01, 0.1]]).unsqueeze(0), 
#              "Reflectance": torch.Tensor([])  
#              }
# # initial_conditions = torch.Tensor([30, 10]).unsqueeze(0)
# initial_conditions = torch.Tensor([30, 10])

# # solution = solve_lotka_volterra(TRUE_PARAMS['LotkaVolterra'], initial_conditions, t) 

# lv_solution_placeholder = None
# esa_data = np.load(NPZ_PATH) 
# # data = np.load("esa_oc_412_valid_samples.npz")

# rrs_all   = torch.tensor(esa_data["rrs_412"],   dtype=torch.float32)
# atot_all  = torch.tensor(esa_data["atot_412"],  dtype=torch.float32)
# chlor_all = torch.tensor(esa_data["chlor_a"],   dtype=torch.float32)
# bbp_all = torch.tensor(esa_data["bbp_412"],   dtype=torch.float32)

# M = chlor_all.shape[0]
# print("Total samples:", M)

# B = 20000  # start with 10k–50k

# idx = torch.randperm(M)[:B]

# rrs_batch   = rrs_all[idx]
# atot_batch  = atot_all[idx]
# chlor_batch = chlor_all[idx]
# bbp_batch = bbp_all[idx]

# solution_inverse_chlor_a = torch.from_numpy(esa_data['chlor_a']) 

# REAL_SOLUTIONS = {"LotkaVolterra": lv_solution_placeholder, 
#                   "Reflectance": solution_inverse_chlor_a
#                   } 

# if __name__=="__main__":
#     print(REAL_SOLUTIONS)
#     print(PROBLEM_BOUNDS)
