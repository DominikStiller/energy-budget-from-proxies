from __future__ import annotations

import json
import sys

import dask
import numpy as np
import xarray as xr
from dask.distributed import Client

from lmrecon.io import save_mfdataset
from lmrecon.logger import get_logger
from lmrecon.mapper import PhysicalSpaceForecastSpaceMapper
from lmrecon.stats import area_weighted_rmse
from lmrecon.util import get_data_path

logger = get_logger(__name__)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Must specify dataset path and mapper id")

    client = Client(n_workers=dask.system.CPU_COUNT // 4, threads_per_worker=1)  # noqa: F841

    mapper_id = sys.argv[1]
    ds_path = get_data_path() / sys.argv[2]
    assert ds_path.suffix == ".zarr"
    output_directory = ds_path.with_name(f"{ds_path.stem}-truncated-{mapper_id}").resolve()

    if output_directory.exists():
        raise ValueError(f"Output directory {output_directory} exists")

    logger.info(f"Loading data from {ds_path}")
    ds = xr.open_zarr(ds_path)
    mapper = PhysicalSpaceForecastSpaceMapper.load(
        get_data_path() / "mapper" / mapper_id / "mapper.pkl"
    )

    logger.info("Truncating")
    ds_truncated = mapper.truncate_dataset(ds).astype(np.float32).persist()

    logger.info("Calculating RMSE")
    rmse = area_weighted_rmse(ds, ds_truncated).compute()

    save_mfdataset(ds_truncated, output_directory, add_timestamp=False)
    json.dump(
        {
            "mapper_id": mapper_id,
            "dataset": str(ds_path),
            "rmse": {field: rmse[field].item() for field in rmse.variables},
        },
        (output_directory / "metadata.json").open("w"),
        indent=4,
    )

    logger.info("Computing of truncated dataset completed")
