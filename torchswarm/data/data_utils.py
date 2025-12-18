import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import interactive
import numpy as np

URL = "https://data.ceda.ac.uk/.../ESACCI-OC-L3S-OC_PRODUCTS-....nc"

# chl_gt = get_chlorophyll_gt(
#     URL,
#     lat=45.39,
#     lon=13.34,
#     side=1.0
# )
import numpy as np

def get_bounds(npz_path, columns=None, verbose=True):
    """
    Load a .npz dataset and compute min/max bounds for selected columns.

    Parameters
    ----------
    npz_path : str
        Path to the .npz file.
    columns : list of str or None
        Keys inside the npz file to compute bounds for.
        If None, all array-like entries are used.
    verbose : bool
        Whether to print the bounds.

    Returns
    -------
    bounds : dict
        Dictionary {column: (min, max)}
    """

    data = np.load(npz_path)

    # If no columns specified, infer numeric arrays
    if columns is None:
        columns = [
            k for k in data.files
            if isinstance(data[k], np.ndarray)
            and data[k].ndim >= 1
            and data[k].size > 0
        ]

    bounds = {}

    if verbose:
        print("\n================ FINAL DATASET ================\n")
        print("Samples:", len(data[columns[0]]))

    for col in columns:
        arr = data[col]

        # Safety: ignore NaNs if present
        col_min = np.nanmin(arr)
        col_max = np.nanmax(arr)

        bounds[col] = (col_min, col_max)

        if verbose:
            print(f"{col} range: {col_min:.6g} → {col_max:.6g}")

    return bounds

def preprocess_data(arr):
    rrs_arr = rrs.values
    rrs_arr[np.isnan(rrs_arr)] = 0
    nonzero_rrs = rrs_arr != 0
    
     
def get_esa_metadata(wavelength='412', variables='chlor_a'):
    FILE = "ESACCI-OC-L3S-OC_PRODUCTS-MERGED-5D_DAILY_4km_GEO_PML_OCx_QAA-20220101-fv6.0.nc"
    ds = xr.open_dataset(FILE)
    print("\n================ DATASET STRUCTURE ================\n")
    print(ds)

# ----------------------------
# 2. Print available data variables
# ----------------------------

    print("\n================ AVAILABLE VARIABLES ================\n")
    for var in ds.data_vars:
        print(f"- {var} : {ds[var].dims}  {ds[var].shape}")
    
    # for i in variables: ... should probably generalize
    subset = ds[[f'Rrs_{wavelength}',f'atot_{wavelength}', 'chlor_a']]

    rrs = subset['Rrs_412'].isel(time=0).where(subset['Rrs_412'] != 'nan')
    atot = subset['atot_412'].isel(time=0).where(subset['atot_412'] != 'nan')
    chlor_a = subset['chlor_a'].isel(time=0).where(subset['chlor_a'] != 'nan')

    gt = preprocess_data(chlor_a) 
    
    # in gase we want to retrieve more than the ground truth...
    # final_valid = (nonzero_rrs & nonzero_atot & nonzero_chlor_a) 

    # # valid_rrs = rrs_arr[nonzero_rrs]
    # # valid_atot = atot_arr[nonzero_atot]

    # valid_rrs = rrs_arr[final_valid]
    # valid_atot = atot_arr[final_valid]
    # valid_chlor_a = chlor_a_arr[final_valid]

    return gt


# def get_chlorophyll_gt(
#     url,
#     lat,
#     lon,
#     side=1.0,
#     time_index=0,
# ):
#     """
#     Retrieve chlorophyll_a ground truth from ESA CCI via OPeNDAP.

#     Returns:
#         chlor_a (np.ndarray): 2D array (lat, lon)
#     """

#     ds = xr.open_dataset(url, engine="netcdf4")

#     chl = (
#         ds["chlor_a"]
#         .isel(time=time_index)
#         .sel(
#             lat=slice(lat - side, lat + side),
#             lon=slice(lon - side, lon + side),
#         )
#         .where(ds["chlor_a"] > 0)
#     )

#     return chl.values

