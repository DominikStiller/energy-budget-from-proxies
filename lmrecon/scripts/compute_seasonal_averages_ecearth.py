from __future__ import annotations

import sys

import dask
import numpy as np
import xarray as xr
from dask.distributed import Client

from lmrecon.grid import GLOBAL_GRID, Regridder
from lmrecon.io import VARS_AND_DIMS_TO_DROP
from lmrecon.logger import get_logger
from lmrecon.stats import average_seasonally
from lmrecon.units import convert_to_si_units
from lmrecon.util import filter_cf_valid, get_base_path, get_data_path, standardize_coordinate_names

logger = get_logger(__name__)


def _load_data():
    das = {}
    for variable in ["tas", "tos", "rsdt", "rlut", "rsut", "siconc", "ohc300", "ohc700"]:
        filename = variable
        if "ohc" in filename:
            filename = filename.replace("ohc", "thetaot")
        ds = xr.open_dataset(
            get_base_path()
            / f"datasets/simulations/past1000-EC-Earth3-Veg-LR/past1000-EC-Earth3-Veg-LR_{filename}.nc",
            chunks="auto",
        )

        ds = standardize_coordinate_names(ds)
        ds = ds.assign_coords(lat=filter_cf_valid(ds["lat"]), lon=filter_cf_valid(ds["lon"]))
        da = convert_to_si_units(ds[variable if "ohc" not in variable else "thetao"])
        da = Regridder(GLOBAL_GRID).regrid(da, ignore_degenerate=True, periodic=False)
        if "ohc" in variable:
            depth = 300 if variable == "ohc300" else 700
            rho = 1025  # kg/m^3
            cp = 3850  # J/(kg K)
            da = np.float32(rho * cp * depth) * da
        das[variable] = da.rename(variable)

    ds_merged = xr.merge(das.values()).drop_attrs()
    ds_merged = ds_merged.drop_vars(VARS_AND_DIMS_TO_DROP, errors="ignore").squeeze()
    return ds_merged


if __name__ == "__main__":
    experiment_path = get_data_path() / "cmip6" / "EC-Earth3-Veg-LR" / "past1000"
    output_path = experiment_path / "seasonal_averages.zarr"
    if output_path.exists():
        print(f"Output path {output_path} exists")
        sys.exit()

    client = Client(n_workers=dask.system.CPU_COUNT // 2, threads_per_worker=1)  # noqa: F841
    print(client.dashboard_link)

    ds = _load_data()

    if "rsdt" in ds.variables and "rsut" in ds.variables and "rlut" in ds.variables:
        ds["eei"] = ds["rsdt"] - ds["rsut"] - ds["rlut"]
    if "siconc" in ds.variables:
        ds["siconcn"] = ds["siconc"].where(ds.lat > 0)
        ds["siconcs"] = ds["siconc"].where(ds.lat < 0)

    logger.info("Averaging seasonally")
    ds = average_seasonally(ds)

    logger.info(f"Saving dataset to {output_path}")
    ds.chunk(chunks=dict(time=200)).to_zarr(output_path)

    logger.info("Computing of seasonal averages completed")
