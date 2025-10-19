"""
Computes forecasts and the corresponding verification data for a number of cases.
"""

from __future__ import annotations

import concurrent.futures
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from lmrecon.io import open_mfdataset
from lmrecon.lim import LIM
from lmrecon.logger import get_logger, logging_disabled
from lmrecon.mapper import PhysicalSpaceForecastSpaceMapper
from lmrecon.util import get_data_path, get_host, stack_state, to_math_order, unstack_state

logger = get_logger(__name__)


def run_forecast(i):
    idx = sample[i]
    time = data_training.time[idx].item()
    print(f"Case {i + 1}/{n_cases} (time {time:.2f})")

    with logging_disabled():
        initial_reduced = mapper.forward(
            to_math_order(stack_state(data_true.isel(time=idx).compute()))
        )
        if deterministic:
            fc_reduced = lim.forecast_deterministic(initial_reduced, n_steps)
            fc_physical = mapper.backward(fc_reduced)
        else:
            # initial_physical = create_initial_ensemble_from_perturbations(data_true.sel(time=year), n_ens, year_start=year, std=0.05)
            # initial_reduced = mapper.forward(to_math_order(stack_state(initial_physical)))
            n_int_steps_per_tau = 1440 // 4
            fc_reduced = lim.forecast_stochastic(
                initial_reduced, n_steps, n_int_steps_per_tau, n_ens, progressbar=False
            )
            fc_physical = mapper.backward(fc_reduced.stack(dict(sample=["time", "ens"]))).unstack(
                "sample"
            )

        fc_physical = (
            unstack_state(fc_physical)
            .expand_dims("case")
            .assign_coords(case=[i], time=np.arange(n_steps + 1) * lim.tau)
        )

    true = (
        data_true.isel(time=range(idx, idx + n_steps + 1))
        .expand_dims("case")
        .assign_coords(case=[i], time=np.arange(n_steps + 1) * lim.tau)
    )
    true_truncated = (
        data_true_truncated.isel(time=range(idx, idx + n_steps + 1))
        .expand_dims("case")
        .assign_coords(case=[i], time=np.arange(n_steps + 1) * lim.tau)
    )

    fc_physical[fields_to_save].astype(np.float32).to_netcdf(
        output_directory / f"forecast-{i}.nc", engine="h5netcdf"
    )
    true[fields_to_save].astype(np.float32).to_netcdf(
        output_directory / f"verification-{i}.nc", engine="h5netcdf"
    )
    true_truncated[fields_to_save].astype(np.float32).to_netcdf(
        output_directory / f"verification-trunc-{i}.nc", engine="h5netcdf"
    )


if __name__ == "__main__":
    lim_id = sys.argv[1]
    n_cases = 50
    n_ens = 50
    n_steps = 4
    deterministic = False
    fields_to_save = [
        "tas",
        "rsut",
        "rlut",
    ]

    lim_path = Path(get_data_path() / "lim" / lim_id)
    lim = LIM.load(lim_path / "lim.pkl")

    with (lim_path / "metadata.json").open("r") as f:
        metadata = json.load(f)
    mapper_path = get_data_path() / "mapper" / Path(metadata["mapper_id"])

    data_training = open_mfdataset(mapper_path / "seasonal_anomalies")["data"]
    with (mapper_path / "metadata.json").open("r") as f:
        true_data_path = get_data_path() / json.load(f)["physical_dataset"]

    data_true = xr.open_zarr(get_data_path() / true_data_path)
    data_true_truncated = open_mfdataset(
        next(iter(true_data_path.parent.glob(f"{true_data_path.stem}-truncated-*")))
    )

    mapper = PhysicalSpaceForecastSpaceMapper.load(mapper_path / "mapper.pkl")

    n_cases = n_cases or (len(data_training.time) - 1)
    sample = np.random.default_rng(89155354).choice(
        np.arange(len(data_training.time) - 1), n_cases, replace=False
    )

    output_directory = lim_path / "verification"
    output_directory.mkdir(parents=True)

    host = get_host()
    if host == "enkf":
        # 2 workers with 8 cores on enkf -> 4 cores per worker
        max_workers = 2
    elif host == "derecho":
        # 32 workers with 128 cores on Derecho -> 4 cores per worker
        max_workers = 32
    else:
        max_workers = None

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        concurrent.futures.wait([executor.submit(run_forecast, i) for i in range(n_cases)])

    logger.info("Computing of verification forecasts completed")
