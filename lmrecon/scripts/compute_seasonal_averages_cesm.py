from __future__ import annotations

import sys
from pathlib import Path

import dask
from dask.distributed import Client

from lmrecon.io import CESMTimeseriesLoader
from lmrecon.logger import get_logger
from lmrecon.stats import average_seasonally
from lmrecon.util import get_data_path

logger = get_logger(__name__)


if __name__ == "__main__":
    case_output_path = Path(sys.argv[1])
    variables = sys.argv[2].split(",")

    experiment_path = get_data_path() / "cmip6" / "CESM2" / case_output_path.name
    output_path = experiment_path / "seasonal_averages.zarr"
    if output_path.exists():
        print(f"Output path {output_path} exists")
        sys.exit()

    client = Client(n_workers=dask.system.CPU_COUNT // 2, threads_per_worker=1)  # noqa: F841

    ds = CESMTimeseriesLoader(case_output_path, variables).load_dataset()

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
