from __future__ import annotations

import json
from pathlib import Path

import dask
import xarray as xr
from dask.distributed import Client

from lmrecon.io import save_mfdataset
from lmrecon.logger import get_logger
from lmrecon.mapper import PhysicalSpaceForecastSpaceMapper
from lmrecon.scripts.train_lim import train_lim
from lmrecon.util import get_data_path, get_timestamp, stack_state, to_cf_order, to_math_order

logger = get_logger(__name__)

if __name__ == "__main__":
    client = Client(n_workers=dask.system.CPU_COUNT // 2, threads_per_worker=1)  # noqa: F841

    # model, experiment = "MPI-ESM1-2-LR", "past2k"
    # model, experiment = "CESM2-WACCM-FV2", "past1000"
    # model, experiment = "MRI-ESM2-0", "past1000"
    # model, experiment = "EC-Earth3-Veg-LR", "past1000"
    model, experiment = "MIROC-ES2L", "past1000"

    # model, experiment = "MPI-ESM1-2-LR", "past2k_historical"
    # model, experiment = "CESM2-WACCM-FV2", "past1000_historical"
    # model, experiment = "MRI-ESM2-0", "past1000_historical"
    # model, experiment = "EC-Earth3-Veg-LR", "past1000_historical"
    # model, experiment = "MIROC-ES2L", "past1000_historical"

    # model, experiment = "CESM2-WACCM-FV2", "piControl"

    ds_path = Path() / "cmip6" / model / experiment / "seasonal_anomalies_detrended.zarr"
    # ds_path = Path() / "cmip6" / model / experiment / "seasonal_anomalies.zarr"
    # ds_path = Path() / "cmip6" / "NorESM2-LM" / "hist-nat" / "seasonal_anomalies_with_F_R.zarr"
    # ds_path = Path() / "cmip6" / "CanESM5" / "hist-nat" / "seasonal_anomalies_with_F_R.zarr"

    # Production
    k = {
        "tas": 20,
        "tos": 20,
        "eei": 15,
        "rsut": 15,
        "rlut": 10,
        "ohc300": 15,
        "siconcn": 10,
        "siconcs": 10,
    }
    separate_global_mean = ["tas", "tos", "eei", "rsut", "rlut", "ohc300"]
    # separate_global_mean = ["tas", "tos", "F_eei", "F_rsut", "F_rlut", "ohc300"]
    # separate_global_mean = None
    save_anomalies = True

    # F/R test
    # k = {
    #     "tas": 20,
    #     "tos": 20,
    #     "F_eei": 15,
    #     "R_eei": 15,
    #     "ohc300": 15,
    # }
    # separate_global_mean = ["tas", "tos", "F_eei", "R_eei", "ohc300"]
    # # separate_global_mean = None
    # save_anomalies = True

    # Determine retained variance
    # k = dict.fromkeys(["tas", "tos", "eei", "rsut", "rlut", "ohc300", "siconcn", "siconcs"], 50)
    # separate_global_mean = ["tas", "tos", "eei", "rsut", "rlut", "ohc300"]
    # save_anomalies = False

    logger.info(f"Loading physical dataset ({ds_path})")
    ds = xr.open_zarr(get_data_path() / ds_path)
    if isinstance(k, dict):
        ds = ds[list(k.keys())]
    ds = to_math_order(stack_state(ds))

    mapper = PhysicalSpaceForecastSpaceMapper(k, separate_global_mean)
    mapper.fit(ds)

    mapper_id = get_timestamp()
    directory = get_data_path() / "mapper" / mapper_id
    mapper.save(directory)
    if save_anomalies:
        save_mfdataset(
            to_cf_order(mapper.forward(ds)).to_dataset(name="data"),
            directory / "seasonal_anomalies",
            add_timestamp=False,
        )
        train_lim(mapper_id)

    json.dump(
        {
            "physical_dataset": str(ds_path),
            "k": k,
            "separate_global_mean": separate_global_mean,
        },
        (directory / "metadata.json").open("w"),
        indent=4,
    )

    logger.info(f"Computing of training data completed (mapper_id {directory.name})")
