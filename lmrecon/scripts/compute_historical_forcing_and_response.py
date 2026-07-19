from __future__ import annotations

import sys

import xarray as xr
from dask.distributed import Client

from lmrecon.io import IntakeESMLoader
from lmrecon.logger import get_logger
from lmrecon.stats import anomalize, average_seasonally
from lmrecon.util import get_data_path

logger = get_logger(__name__)


if __name__ == "__main__":
    model_id = sys.argv[1]
    experiment_id = sys.argv[2]
    members = sys.argv[3].split(",")

    experiment_path = get_data_path() / "cmip6" / model_id / experiment_id
    input_path = experiment_path / "seasonal_anomalies.zarr"
    output_path = experiment_path / "seasonal_anomalies_with_F_R.zarr"
    rad_fields = ["eei", "rsut", "rlut"]

    client = Client(n_workers=16, threads_per_worker=1)  # noqa: F841

    logger.info(f"Loading seasonal averages from {experiment_path}")
    ds_hist = xr.open_zarr(input_path)

    experiment_id_clim = "piClim-histnat" if experiment_id == "hist-nat" else "piClim-histall"
    logger.info(f"Loading {experiment_id_clim} data")
    ds_histall = xr.concat(
        [
            IntakeESMLoader(model_id, experiment_id_clim).load_dataset(member_id=member)
            for member in members
        ],
        dim="ens",
    ).mean("ens")
    ds_histall["eei"] = ds_histall["rsdt"] - ds_histall["rsut"] - ds_histall["rlut"]
    ds_histall = average_seasonally(ds_histall[rad_fields])
    ds_histall, ds_hist = xr.align(ds_histall, ds_hist, join="inner")

    logger.info("Loading piClim-control data")
    ds_control = IntakeESMLoader(model_id, "piClim-control").load_dataset()
    ds_control["eei"] = ds_control["rsdt"] - ds_control["rsut"] - ds_control["rlut"]
    ds_control = average_seasonally(ds_control[rad_fields])

    _, climatology = anomalize(ds_control, return_climatology=True)

    logger.info("Computing forcing and response")
    ds_F = anomalize(ds_histall, climatology=climatology)
    ds_R = ds_hist - ds_F

    ds_F = ds_F.rename({f: f"F_{f}" for f in rad_fields})
    ds_R = ds_R.rename({f: f"R_{f}" for f in rad_fields})

    logger.info(f"Saving dataset to {output_path}")
    xr.merge([ds_hist, ds_F, ds_R]).chunk(chunks=dict(time=200)).to_zarr(output_path)

    logger.info("Computing of forcing and response completed")
