from __future__ import annotations

import sys

import dask
from dask.distributed import Client

from lmrecon.io import IntakeESMLoader, get_catalog_location
from lmrecon.logger import get_logger
from lmrecon.stats import average_seasonally
from lmrecon.util import get_data_path

logger = get_logger(__name__)


if __name__ == "__main__":
    model_id = sys.argv[1]
    experiment_id = sys.argv[2]
    catalog_location = get_catalog_location(sys.argv[3] if len(sys.argv) > 3 else "lmrecon")
    member_id = sys.argv[4] if len(sys.argv) > 4 else None

    experiment_path = get_data_path() / "cmip6" / model_id / experiment_id
    output_path = experiment_path / "seasonal_averages.zarr"
    if output_path.exists():
        print(f"Output path {output_path} exists")
        sys.exit()

    client = Client(n_workers=dask.system.CPU_COUNT // 2, threads_per_worker=1)  # noqa: F841

    ds = IntakeESMLoader(model_id, experiment_id, catalog_location=catalog_location).load_dataset(
        member_id=member_id
    )

    if "rsdt" in ds.variables and "rsut" in ds.variables and "rlut" in ds.variables:
        ds["eei"] = ds["rsdt"] - ds["rsut"] - ds["rlut"]
    if "siconc" in ds.variables:
        ds["siconcn"] = ds["siconc"].where(ds.lat > 0)
        ds["siconcs"] = ds["siconc"].where(ds.lat < 0)

    logger.info("Averaging seasonally")
    ds = average_seasonally(ds)

    if model_id == "MPI-ESM1-2-LR" and experiment_id == "past2k":
        ds["time"] = ds.time - 7000

    logger.info(f"Saving dataset to {output_path}")
    ds.chunk(chunks=dict(time=200)).to_zarr(output_path)

    logger.info("Computing of seasonal averages completed")
