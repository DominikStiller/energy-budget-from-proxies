from __future__ import annotations

import json
import sys

import dask
import numpy as np
import xarray as xr
from dask.distributed import Client

from lmrecon.logger import get_logger
from lmrecon.scripts.calibrate_indices import calibrate_indices
from lmrecon.stats import anomalize
from lmrecon.time import (
    use_decimal_year_time_coords,
    use_string_season_coords,
    use_tuple_time_coords,
)
from lmrecon.units import calculate_sistatistics
from lmrecon.util import get_data_path

logger = get_logger(__name__)


if __name__ == "__main__":
    client = Client(n_workers=dask.system.CPU_COUNT // 2, threads_per_worker=1)  # noqa: F841

    experiments = {
        "CESM2": "past1000_historical",
        "MPI-ESM1-2-LR": "past2k_historical",
        "MRI-ESM2-0": "past1000_historical",
        # "MIROC-ES2L": "past1000_historical",
    }

    base_path = get_data_path() / "cmip6"
    output_path = base_path / "mmm" / next(iter(experiments.values()))
    if output_path.exists():
        print(f"Output path {output_path} exists")
        sys.exit()

    logger.info("Loading input data")
    dss = [
        xr.open_zarr(base_path / model / exp / "seasonal_averages.zarr")
        for model, exp in experiments.items()
    ]
    dss = [xr.merge([ds, calculate_sistatistics(ds)]) for ds in dss]

    logger.info("Computing multi-model mean")
    # Use skipna so that only the union of ocean masks is used (even if some cells only have one model)
    mmm = use_decimal_year_time_coords(
        xr.concat([use_tuple_time_coords(ds) for ds in dss], dim="model", join="inner").mean(
            "model", skipna=True
        )
    )

    logger.info(f"Saving averages to {output_path}")
    mmm.chunk(chunks=dict(time=200)).to_zarr(output_path / "seasonal_averages.zarr")

    # Reload so we don't have to recompute mean
    mmm = xr.open_zarr(output_path / "seasonal_averages.zarr")

    logger.info("Computing climatologies and anomalies")
    mmm_anom, climatology_1961_1990 = anomalize(mmm, period=(1961, 1991), return_climatology=True)
    _, climatology_1979_2000 = anomalize(mmm, period=(1979, 2001), return_climatology=True)

    logger.info(f"Saving anomalies to {output_path}")
    mmm_anom.chunk(chunks=dict(time=200)).to_zarr(output_path / "seasonal_anomalies.zarr")

    if not (output_path / "climatology_1961-1990.nc").exists():
        use_string_season_coords(climatology_1961_1990).compute().to_netcdf(
            output_path / "climatology_1961-1990.nc"
        )

    if not (output_path / "climatology_1979-2000.nc").exists():
        use_string_season_coords(climatology_1979_2000).compute().to_netcdf(
            output_path / "climatology_1979-2000.nc"
        )

    logger.info("Saving land mask")
    np.isnan(mmm["tos"]).all("time").to_dataset(name="mask").to_netcdf(
        output_path.parent / "landmask.nc"
    )

    json.dump(
        {
            "experiments": experiments,
        },
        (output_path / "metadata.json").open("w"),
        indent=4,
    )

    logger.info("Computing of multi-model mean completed")

    calibrate_indices("mmm", "past1000_historical")
