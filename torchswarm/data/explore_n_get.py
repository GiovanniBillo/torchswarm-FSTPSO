#!/usr/bin/env python3
"""
Extract valid ESA CCI Ocean Colour samples and save them
with lat-lon metadata in both NumPy (.npz) and CSV format.
"""

import xarray as xr
import numpy as np
import csv

# ------------------------------------------------------
# Configuration
# ------------------------------------------------------

FILE = "ESACCI-OC-L3S-OC_PRODUCTS-MERGED-5D_DAILY_4km_GEO_PML_OCx_QAA-20220101-fv6.0.nc"

NPZ_OUTPUT = "esa_oc_412_valid_samples.npz"
# CSV_OUTPUT = "esa_oc_412_valid_samples.csv"
# VAR_BOUNDS_OUTPUT = "esa_oc_412_var_bounds.csv" 
TIME_INDEX = 0   # 5D product -> single time slice

# ------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------

ds = xr.open_dataset(FILE)

print("\n================ DATASET STRUCTURE ================\n")
print(ds)

# ------------------------------------------------------
# 2. Select variables of interest
# ------------------------------------------------------

subset = ds[["Rrs_412", "atot_412", "chlor_a"]].isel(time=TIME_INDEX)

# ------------------------------------------------------
# 3. Convert to NumPy arrays
# ------------------------------------------------------

rrs_arr = subset["Rrs_412"].values
atot_arr = subset["atot_412"].values
chlor_a_arr = subset["chlor_a"].values

# ------------------------------------------------------
# 4. Build validity mask
# ------------------------------------------------------

valid_mask = (
    np.isfinite(rrs_arr) &
    np.isfinite(atot_arr) &
    np.isfinite(chlor_a_arr) &
    (rrs_arr > 0) &
    (atot_arr > 0) &
    (chlor_a_arr > 0)
)

print("\n================ VALIDITY CHECK ================\n")
print("Total pixels:", rrs_arr.size)
print("Valid pixels:", np.count_nonzero(valid_mask))

# ------------------------------------------------------
# 5. Extract valid samples
# ------------------------------------------------------

valid_rrs = rrs_arr[valid_mask]
valid_atot = atot_arr[valid_mask]
valid_chlor_a = chlor_a_arr[valid_mask]

# ------------------------------------------------------
# 6. Extract matching lat-lon metadata
# ------------------------------------------------------

lat_vals = subset["lat"].values
lon_vals = subset["lon"].values

lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)

valid_lat = lat_grid[valid_mask]
valid_lon = lon_grid[valid_mask]

# ------------------------------------------------------
# 7. Final sanity checks
# ------------------------------------------------------

assert len(valid_rrs) == len(valid_atot) == len(valid_chlor_a)
assert len(valid_rrs) == len(valid_lat) == len(valid_lon)

print("\n================ FINAL DATASET ================\n")
print("Samples:", len(valid_rrs))
print("Rrs_412 range:", valid_rrs.min(), "→", valid_rrs.max())
print("atot_412 range:", valid_atot.min(), "→", valid_atot.max())
print("chlor_a range:", valid_chlor_a.min(), "→", valid_chlor_a.max())

# ------------------------------------------------------
# 8. Save NumPy dataset (.npz)
# ------------------------------------------------------

np.savez_compressed(
    NPZ_OUTPUT,
    rrs_412=valid_rrs,
    atot_412=valid_atot,
    chlor_a=valid_chlor_a,
    lat=valid_lat,
    lon=valid_lon,
    source="ESA CCI Ocean Colour v6.0",
    product="OC_PRODUCTS",
    wavelength="412 nm",
)

print(f"\nSaved NumPy dataset to: {NPZ_OUTPUT}")

# ------------------------------------------------------
# 9. Save CSV (for inspection)
# ------------------------------------------------------

# with open(CSV_OUTPUT, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["lat", "lon", "Rrs_412", "atot_412", "chlor_a"])

#     for i in range(len(valid_rrs)):
#         writer.writerow([
#             valid_lat[i],
#             valid_lon[i],
#             valid_rrs[i],
#             valid_atot[i],
#             valid_chlor_a[i],
#         ])

# with open(VAR_BOUNDS_OUTPUT, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Rrs_412_min", "Rrs_412_min", "atot_412_min", "atot_412_min", "chlor_a_min", "chlor_a_max"])
#     writer.writerow(valid_rrs.min(), valid_rrs.max(),
#                     valid_atot.min(), valid_atot.max(),
#                     valid_chlor_a.min(), valid_chlor_a.max()) 

# print(f"Saved CSV dataset to: {CSV_OUTPUT}")
# print(f"Saved VAR BOUNDS CSV to: {VAR_BOUNDS_OUTPUT}")

