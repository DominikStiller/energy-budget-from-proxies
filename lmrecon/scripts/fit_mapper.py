from __future__ import annotations

import json
from pathlib import Path

import dask
import xarray as xr
from dask.distributed import Client

from lmrecon.io import save_mfdataset
from lmrecon.logger import get_logger
from lmrecon.mapper import PhysicalSpaceForecastSpaceMapper
from lmrecon.util import get_data_path, get_timestamp, stack_state, to_cf_order, to_math_order

logger = get_logger(__name__)

if __name__ == "__main__":
    client = Client(n_workers=dask.system.CPU_COUNT, threads_per_worker=1)  # noqa: F841

    model, experiment = "MPI-ESM1-2-LR", "past2k"
    # model, experiment = "CESM2", "past1000"
    # model, experiment = "MRI-ESM2-0", "past1000"
    # model, experiment = "MIROC-ES2L", "past1000"
    ds_path = Path() / "cmip6" / model / experiment / "seasonal_anomalies_detrended.zarr"

    # Individual EOFs
    k = 30
    l = 0
    k_direct = {
        "tas": 20,
        "tos": 20,
        "eei": 30,
        "rsut": 15,
        "rlut": 10,
        "ohc300": 15,
        "siconcn": 10,
        "siconcs": 10,
        # "cldhigh": 10,
        # "cldlow": 20,
        # "clwvi": 15,
    }
    standardize_by_season = False
    separate_global_mean = False
    save_anomalies = True

    # # Individual EOFs to determine retained variance
    # k = 50
    # l = 0
    # k_direct = {
    #     "tas": 50,
    #     "tos": 50,
    #     "rsut": 50,
    #     "rlut": 50,
    #     "ohc300": 50,
    #     "ohc700": 50,
    #     "eei": 50,
    #     "siconcn": 50,
    #     "siconcs": 50,
    #     # "cldhigh": 50,
    #     # "cldlow": 50,
    #     # "clwvi": 50,
    # }
    # standardize_by_season = False
    # separate_global_mean = True
    # save_anomalies = False

    state_fields = [
        "tas",
        "tos",
        "eei",
        "rsut",
        "rlut",
        "ohc300",
        # "ohc700",
        "siconcn",
        "siconcs",
        # "cldhigh",
        # "cldlow",
        # "clwvi",
    ]

    logger.info(f"Loading physical dataset ({ds_path})")
    ds = xr.open_zarr(get_data_path() / ds_path)[state_fields]

    mapper = PhysicalSpaceForecastSpaceMapper(
        k, l, k_direct, standardize_by_season, separate_global_mean
    )
    ds_eof = mapper.fit_and_forward(to_math_order(stack_state(ds)))

    directory = get_data_path() / "mapper" / get_timestamp()
    mapper.save(directory)
    if save_anomalies:
        save_mfdataset(
            to_cf_order(ds_eof).to_dataset(name="data"),
            directory / "seasonal_anomalies",
            add_timestamp=False,
        )

    json.dump(
        {
            "physical_dataset": str(ds_path),
            "k": k,
            "l": l,
            "k_direct": k_direct,
            "standardize_by_season": standardize_by_season,
            "separate_global_mean": separate_global_mean,
        },
        (directory / "metadata.json").open("w"),
        indent=4,
    )

    logger.info(f"Computing of training data completed (mapper_id {directory.name})")
