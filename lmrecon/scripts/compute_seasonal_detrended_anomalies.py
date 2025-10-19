from __future__ import annotations

import sys

import dask
import xarray as xr
from dask.distributed import Client

from lmrecon.logger import get_logger
from lmrecon.stats import detrend_polynomial
from lmrecon.util import get_data_path

logger = get_logger(__name__)


if __name__ == "__main__":
    model_id = sys.argv[1]
    experiment_id = sys.argv[2]

    experiment_path = get_data_path() / "cmip6" / model_id / experiment_id
    output_path = experiment_path / "seasonal_anomalies_detrended.zarr"
    if output_path.exists():
        print(f"Output path {output_path} exists")
        sys.exit()

    client = Client(n_workers=dask.system.CPU_COUNT // 2, threads_per_worker=1)  # noqa: F841

    logger.info(f"Loading seasonal anomalies from {experiment_path}")
    ds_anom = xr.open_zarr(experiment_path / "seasonal_anomalies.zarr")

    logger.info("Detrending")
    ds_detrend = detrend_polynomial(ds_anom, by_season=True)

    logger.info(f"Saving dataset to {output_path}")
    ds_detrend.chunk(chunks=dict(time=200)).to_zarr(output_path, zarr_format=2)

    logger.info("Computing of seasonal detrended anomalies completed")
