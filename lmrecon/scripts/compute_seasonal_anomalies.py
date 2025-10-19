from __future__ import annotations

import sys

import dask
import xarray as xr
from dask.distributed import Client

from lmrecon.logger import get_logger
from lmrecon.scripts.calibrate_indices import calibrate_indices
from lmrecon.stats import anomalize
from lmrecon.time import use_string_season_coords
from lmrecon.util import get_data_path

logger = get_logger(__name__)


if __name__ == "__main__":
    model_id = sys.argv[1]
    experiment_id = sys.argv[2]

    experiment_path = get_data_path() / "cmip6" / model_id / experiment_id
    output_path = experiment_path / "seasonal_anomalies.zarr"
    if output_path.exists():
        print(f"Output path {output_path} exists")
        sys.exit()

    client = Client(n_workers=dask.system.CPU_COUNT // 2, threads_per_worker=1)  # noqa: F841

    logger.info(f"Loading seasonal averages from {experiment_path}")
    ds = xr.open_zarr(experiment_path / "seasonal_averages.zarr")

    logger.info("Computing anomalies")
    ds_anom, climatology = anomalize(ds, return_climatology=True)

    logger.info(f"Saving dataset to {output_path}")
    ds_anom.chunk(chunks=dict(time=200)).to_zarr(output_path)

    if not (experiment_path / "climatology.nc").exists():
        # Needs compute() due to some bug
        use_string_season_coords(climatology).compute().to_netcdf(
            experiment_path / "climatology.nc"
        )

    logger.info("Computing of seasonal anomalies completed")

    calibrate_indices(model_id, experiment_id)
