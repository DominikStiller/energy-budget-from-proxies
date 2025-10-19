from __future__ import annotations

import sys

import dask
import numpy as np
import xarray as xr
from dask.distributed import Client

from lmrecon.logger import get_logger
from lmrecon.stats import anomalize
from lmrecon.time import use_string_season_coords
from lmrecon.util import get_data_path

logger = get_logger(__name__)


if __name__ == "__main__":
    model_id = sys.argv[1]
    experiment_id1 = sys.argv[2]
    experiment_id2 = sys.argv[3]

    model_path = get_data_path() / "cmip6" / model_id
    output_path = model_path / f"{experiment_id1}_{experiment_id2}"
    if output_path.exists():
        print(f"Output path {output_path} exists")
        sys.exit()

    client = Client(n_workers=dask.system.CPU_COUNT // 2, threads_per_worker=1)  # noqa: F841

    logger.info(f"Loading seasonal averages from {model_path}")
    ds1 = xr.open_zarr(model_path / experiment_id1 / "seasonal_averages.zarr")
    ds2 = xr.open_zarr(model_path / experiment_id2 / "seasonal_averages.zarr")

    # Remove overlap if exists, add a small epsilon to make start point exclusive
    ds2 = ds2.sel(time=slice(ds1.time[-1] + 1e-10, None))

    if model_id == "CESM2":
        # Match five-year means before and after stitching point by shifting ds1
        # All other models have historical run initialized from past1000
        offset_fields = ["ohc300", "ohc700", "rsdt"]
        offset = (
            ds2.isel(time=slice(None, 4 * 5)).mean("time")
            - ds1.isel(time=slice(-4 * 5, None)).mean("time")
        )[offset_fields].compute()
        ds1[offset_fields] = ds1[offset_fields] + offset[offset_fields]

    if (ds2.time[0] - ds1.time[-1]) > 0.51:
        # More than one season is missing
        raise ValueError("Two datasets are not contiguous")
    elif np.isclose(ds2.time[0] - ds1.time[-1], 0.5):
        # Single season is missing (e.g., DJF 1850)
        # Set DJF 1850 = mean of DJF 1849 and DJF 1851
        time1 = ds1.time[-4]
        time2 = ds2.time[3]
        ds_gap = ((ds1.sel(time=time1) + ds2.sel(time=time2)) / 2).assign_coords(
            time=[(time1 + time2) / 2]
        )
        dss = [ds1, ds_gap, ds2]
    else:
        dss = [ds1, ds2]

    ds = xr.concat(dss, dim="time", compat="identical").unify_chunks()

    logger.info(f"Saving averages to {output_path}")
    ds.chunk(chunks=dict(time=200)).to_zarr(output_path / "seasonal_averages.zarr", zarr_format=2)

    logger.info("Computing anomalies relative to 1961-1990")
    ds_anom, climatology = anomalize(ds, period=(1961, 1991), return_climatology=True)

    logger.info(f"Saving anomalies to {output_path}")
    ds_anom.chunk(chunks=dict(time=200)).to_zarr(
        output_path / "seasonal_anomalies.zarr", zarr_format=2
    )

    if not (output_path / "climatology.nc").exists():
        use_string_season_coords(climatology).compute().to_netcdf(output_path / "climatology.nc")

    logger.info("Computing of combined seasonal averages and anomalies completed")
