from __future__ import annotations

import sys

import xarray as xr

from lmrecon.indices import PDOIndex
from lmrecon.logger import get_logger
from lmrecon.util import get_data_path

logger = get_logger(__name__)


def calibrate_indices(model_id, experiment_id):
    experiment_path = get_data_path() / "cmip6" / model_id / experiment_id

    logger.info(f"Loading seasonal anomalies from {experiment_path}")
    ds = xr.open_zarr(experiment_path / "seasonal_anomalies.zarr")

    if "tos" in ds:
        pdo_index = PDOIndex()
        pdo_index.fit(ds["tos"])
        pdo_index.save(experiment_path)

    logger.info("Calibration of indices completed")


if __name__ == "__main__":
    model_id = sys.argv[1]
    experiment_id = sys.argv[2]
    calibrate_indices(model_id, experiment_id)
