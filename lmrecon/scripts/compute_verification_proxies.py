from __future__ import annotations

import json
import pickle
import sys

import cf_xarray as cfxr
import dask
import numpy as np
import xarray as xr
from dask.distributed import Client
from loky import get_reusable_executor
from tqdm import tqdm

from lmrecon.io import open_mfdataset
from lmrecon.logger import get_logger, logging_disabled
from lmrecon.mapper import PhysicalSpaceForecastSpaceMapper
from lmrecon.stats import annualize_seasonal_data
from lmrecon.time import (
    map_decimal_to_season,
    use_decimal_year_time_coords,
    use_tuple_time_coords,
)
from lmrecon.util import get_data_path, get_state_index, to_math_order

logger = get_logger(__name__)


def _calculate_proxy_estimate_seasonal(state, psms):
    psm1 = next(iter(psms.values()))
    state_index = get_state_index(state.coords.indexes["state"], psm1.field, psm1.lat, psm1.lon)
    state_at_proxy_location = state.isel(state=state_index)
    proxy_estimate = []
    # Could also do this vectorized per season
    for i, t in enumerate(state.time):
        season = map_decimal_to_season(t)
        if season in psms:
            proxy_estimate.append(psms[season].forward(state_at_proxy_location.isel(time=i)))
        else:
            proxy_estimate.append(np.atleast_2d(np.nan))
    proxy_estimate = np.squeeze(np.hstack(proxy_estimate))

    return xr.DataArray(
        proxy_estimate, dims=["time"], coords=dict(time=state_at_proxy_location.time)
    )


def _calculate_proxy_estimate_annual(state, psm):
    state_index = get_state_index(state.coords.indexes["state"], psm.field, psm.lat, psm.lon)
    state_at_proxy_location = annualize_seasonal_data(
        state.isel(state=state_index), psm.seasonality
    )
    proxy_estimate = np.squeeze(psm.forward(state_at_proxy_location))

    return xr.DataArray(
        proxy_estimate, dims=["time"], coords=dict(time=state_at_proxy_location.time)
    )


def _collect_proxy_data(state, pdb, psms):
    pids_annual = []
    proxies_annual = []
    estimates_annual = []
    pids_seasonal = []
    proxies_seasonal = []
    estimates_seasonal = []

    def _process_proxy(pid_pobj):
        pid, pobj = pid_pobj
        psm_or_dict = psms[pid]
        is_seasonally_resolved = isinstance(psm_or_dict, dict)

        da_proxy = xr.DataArray(pobj.value, coords=dict(time=pobj.time)).sel(
            time=slice(state.time[0], state.time[-1])
        )

        if is_seasonally_resolved:
            da_estimate = _calculate_proxy_estimate_seasonal(state, psm_or_dict)
            return (
                "seasonal",
                pid,
                use_tuple_time_coords(da_proxy),
                use_tuple_time_coords(da_estimate),
            )
        else:
            da_estimate = _calculate_proxy_estimate_annual(state, psm_or_dict)
            return ("annual", pid, da_proxy, da_estimate)

    with get_reusable_executor() as executor:
        results = list(
            tqdm(
                executor.map(_process_proxy, pdb.records.items()),
                total=len(pdb.records),
                unit="proxies",
            )
        )

    for result in results:
        if result[0] == "seasonal":
            _, pid, da_proxy, da_estimate = result
            pids_seasonal.append(pid)
            proxies_seasonal.append(da_proxy)
            estimates_seasonal.append(da_estimate)
        else:
            _, pid, da_proxy, da_estimate = result
            pids_annual.append(pid)
            proxies_annual.append(da_proxy)
            estimates_annual.append(da_estimate)

    data_seasonal = xr.merge(
        [
            xr.concat(proxies_seasonal, dim="pid").rename("true"),
            xr.concat(estimates_seasonal, dim="pid").rename("estimated"),
        ]
    ).assign_coords(pid=pids_seasonal)
    data_annual = xr.merge(
        [
            xr.concat(proxies_annual, dim="pid").rename("true"),
            xr.concat(estimates_annual, dim="pid").rename("estimated"),
        ]
    ).assign_coords(pid=pids_annual)

    return use_decimal_year_time_coords(data_seasonal), data_annual


def _load_posterior(rundir):
    """
    Map reduced state to physical state.
    """
    state_reduced = cfxr.decode_compress_to_multi_index(
        open_mfdataset(
            rundir / "posterior",
            parallel=True,
            chunks=dict(time=80),
            combine="nested",
            concat_dim="time",
        )
    )["data"].persist()

    with (rundir / "metadata.json").open("r") as f:
        metadata = json.load(f)
        mapper_path = get_data_path() / "mapper" / metadata["mapper_id"]

    mapper = PhysicalSpaceForecastSpaceMapper.load(mapper_path / "mapper.pkl")

    logger.info("Mapping reduced reconstruction to physical space")
    with logging_disabled():
        return mapper.backward(to_math_order(state_reduced.mean("ens")))


def _compute_state_at_proxy_locations(state, psms):
    # Precompute required states for efficiency
    state_indices = []
    for psm in psms.values():
        if isinstance(psm, dict):
            psm = next(iter(psm.values()))
        state_indices.append(
            get_state_index(state.coords.indexes["state"], psm.field, psm.lat, psm.lon)
        )
    return state.isel(state=list(set(state_indices))).compute()


def compute_verification_proxies(rundir):
    logger.info("Starting independent proxy verification")

    client = Client(n_workers=dask.system.CPU_COUNT, threads_per_worker=1)  # noqa: F841

    with (rundir / "metadata.json").open("r") as f:
        metadata = json.load(f)

    psms = pickle.load((get_data_path() / metadata["obs_dataset"] / "psms.pkl").open("rb"))

    logger.info("Mapping required state variables")
    state = _compute_state_at_proxy_locations(_load_posterior(rundir), psms)

    logger.info("Forwarding assimilated proxy system models")
    pdb_assimilated = pickle.load((rundir / "pdb_assimilated.pkl").open("rb"))
    data_seasonal, data_annual = _collect_proxy_data(state, pdb_assimilated, psms)
    data_seasonal.to_netcdf(rundir / "verification_seasonal_assimilated.nc")
    data_annual.to_netcdf(rundir / "verification_annual_assimilated.nc")

    logger.info("Forwarding withheld proxy system models")
    pdb_withheld = pickle.load((rundir / "pdb_withheld.pkl").open("rb"))
    data_seasonal, data_annual = _collect_proxy_data(state, pdb_withheld, psms)
    data_seasonal.to_netcdf(rundir / "verification_seasonal_withheld.nc")
    data_annual.to_netcdf(rundir / "verification_annual_withheld.nc")

    logger.info("Verification proxy computation completed")


if __name__ == "__main__":
    compute_verification_proxies(get_data_path() / "reconstructions" / sys.argv[1])
