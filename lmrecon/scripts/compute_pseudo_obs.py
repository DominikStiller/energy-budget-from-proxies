from __future__ import annotations

import itertools
import json
import pickle
import sys
from pathlib import Path

import cfr
import numpy as np
import xarray as xr
from tqdm import tqdm

from lmrecon.logger import get_logger
from lmrecon.mapper import PhysicalSpaceForecastSpaceMapper
from lmrecon.psm import PSM, IdentityPSM
from lmrecon.stats import annualize_seasonal_data
from lmrecon.time import use_decimal_year_time_coords, use_tuple_time_coords
from lmrecon.util import (
    get_data_path,
    get_timestamp,
)

logger = get_logger(__name__)

# Fix for a Dask bug, possibly https://github.com/dask/distributed/issues/8378
sys.setrecursionlimit(3000)


def generate_pseudoobs_grid(sigma_obs=1.33, multiplicity=1):
    """
    Generate pseudoobs on regular grid.
    """
    spacing = int(type[5:])
    # Symmetric about equator
    lats = np.concatenate([-np.arange(0, 91, spacing)[::-1][:-1], np.arange(0, 91, spacing)])
    lons = np.arange(0, 361, spacing)
    locations = list(itertools.product(lats, lons))
    locations = np.repeat(locations, multiplicity, axis=0)
    np.random.default_rng(48761555).shuffle(locations)

    records = {}
    logger.info(f"Selecting data from original grid, downsampled to {spacing}° grid")
    for i, (lat, lon) in tqdm(list(enumerate(locations))):
        pid = f"P{i + 1}"
        tas = da_truth["tas"].sel(lat=lat, lon=lon, method="nearest")
        lat, lon = (
            tas.lat.item(),
            tas.lon.item(),
        )  # Use grid lat/lon because prior may otherwise be nan
        if np.all(~np.isfinite(tas)):
            continue

        tas += noise_rng.normal(0, sigma_obs, tas.shape)
        records[pid] = cfr.ProxyRecord(
            pid=pid,
            lat=lat,
            lon=lon,
            elev=0,
            time=tas.time.values,
            value=tas.values,
            value_name="TAS",
            value_unit="K",
            ptype="tas",
        )

    psms = {pid: IdentityPSM("tas", lat, lon, sigma_obs, None) for pid in records}

    return records, psms, {"sigma_obs": sigma_obs, "multiplicity": multiplicity}


def _calibrate_pseudo_psm(
    da_proxy,
    da_truth_for_proxy,
    da_truth_truncated_for_proxy,
    proxy,
    psm_original,
    noise_rng,
    scale_snr,
) -> tuple[cfr.ProxyRecord, PSM]:
    if da_proxy.time[0].item() < da_truth_for_proxy.time[0].item():
        logger.warning(
            f"Proxy data start before simulation data for {proxy.pid}, pseudoproxy will be incomplete "
            f" (proxy starts {da_proxy.time[0].item()}, simulation starts {da_truth_for_proxy.time[0].item()})"
        )

    da_truth_for_proxy, da_proxy = xr.align(
        da_truth_for_proxy.dropna("time"),
        da_proxy.dropna("time"),
        join="inner",
    )

    if len(da_truth_for_proxy) == 0:
        logger.warning(f"No overlap between simulation and proxy data for {proxy.pid}, skipping")
        return None, None

    # Generate pseudoproxy
    pp_value = np.squeeze(psm_original.forward(da_truth_for_proxy.values))
    std_noise = pp_value.std() / psm_original.SNR / scale_snr

    pproxy = proxy.copy()
    pproxy.pid = f"pseudo_{proxy.pid}"
    pproxy.value = pp_value + noise_rng.normal(0, std_noise, pp_value.shape)
    pproxy.time = use_decimal_year_time_coords(da_truth_for_proxy).time.values

    # The original PSM and the one calibrated on the pseudoproxy should be identical up to sample
    # error. In practice, the pseudo-PSM SNR is much lower when calibrating against the truncated
    # truth (but not for untruncated). Therefore, use original PSM.
    ppsm = psm_original.copy()
    ppsm.pid = pproxy.pid
    # da_pproxy = xr.DataArray(pproxy.value, dims=["time"], coords=dict(time=pproxy.time))
    # if has_tuple_timedim(da_truth_truncated_for_proxy):
    #     # Seasonal
    #     da_pproxy = use_tuple_time_coords(da_pproxy, force_seasonal=True)
    # da_truth_truncated_for_proxy, da_pproxy = xr.align(
    #     da_truth_truncated_for_proxy.dropna("time"),
    #     da_pproxy.dropna("time"),
    #     join="inner",
    # )
    # if len(da_truth_truncated_for_proxy) == 0:
    #     logger.warning(
    #         f"No overlap between truncated simulation and proxy data for {proxy.pid}, skipping"
    #     )
    #     return None, None

    # ppsm = LinearPSM(
    #     pproxy.pid, psm_original.field, psm_original.lat, psm_original.lon, psm_original.seasonality
    # )
    # ppsm.calibrate(da_truth_truncated_for_proxy.values, da_pproxy.values)
    # if not math.isclose(psm_original.SNR * scale_snr, ppsm.SNR, rel_tol=0.2):
    #     logger.warning(
    #         f"SNR between real and pseudo PSM differs for {proxy.pid}: "
    #         f"real = {psm_original.SNR:.4f}, pseudo = {ppsm.SNR:.4f}"
    #     )

    return pproxy, ppsm


def generate_pseudoobs_from_real(calibrated_obs_dataset: str, scale_snr: float = 1):
    """
    Generate pseudoobs by replicating a real proxy network in terms of seasonality and SNR.
    """
    pdb_original: cfr.ProxyDatabase = pickle.load(
        (get_data_path() / "obs" / calibrated_obs_dataset / "pdb.pkl").open("rb")
    )
    psms_original: dict[str, PSM] = pickle.load(
        (get_data_path() / "obs" / calibrated_obs_dataset / "psms.pkl").open("rb")
    )
    with (get_data_path() / "obs" / calibrated_obs_dataset / "metadata.json").open() as f:
        mapper_id = json.load(f)["mapper_id"]
        mapper = PhysicalSpaceForecastSpaceMapper.load(
            get_data_path() / "mapper" / mapper_id / "mapper.pkl"
        )

    logger.info("Truncating truth data")
    da_truth_truncated = mapper.truncate_dataset(
        xr.merge(
            [
                # GISTEMP, trimmed to 1900-
                da_truth["tas"].sel(time=slice(1900, 2000)),
                # ERSST
                da_truth["tos"].sel(time=slice(1854, 2000)),
            ]
        )
    ).compute()

    records = {}
    psms = {}

    for pid, proxy in tqdm(list(pdb_original.records.items())[129:]):
        psm_original = psms_original[pid]

        da_proxy = xr.DataArray(proxy.value, dims=["time"], coords=dict(time=proxy.time))
        pproxy = None

        if isinstance(psm_original, dict):
            field = next(iter(psm_original.values())).field
            da_truth_for_proxy = use_tuple_time_coords(da_truth[field])
            da_truth_truncated_for_proxy = use_tuple_time_coords(da_truth_truncated[field])
            da_proxy = use_tuple_time_coords(da_proxy)
            pproxies = []
            ppsm = {}
            for season in psm_original:
                pproxy_season, ppsm_season = _calibrate_pseudo_psm(
                    # Select like this so that tuple timestamp is retained
                    da_proxy.sel(time=da_proxy.season == season),
                    da_truth_for_proxy.sel(
                        time=da_truth_for_proxy.season == season,
                        lat=psm_original[season].lat,
                        lon=psm_original[season].lon,
                    ),
                    da_truth_truncated_for_proxy.sel(
                        time=da_truth_truncated_for_proxy.season == season,
                        lat=psm_original[season].lat,
                        lon=psm_original[season].lon,
                    ),
                    proxy,
                    psm_original[season],
                    noise_rng,
                    scale_snr,
                )
                if pproxy_season is not None:
                    # Pseudoproxy can be None for coastal corals that had TOS calibration data in
                    # the original dataset but are part of the land in the pseudoproxy truth dataset
                    # or the truncation for the truth
                    pproxies.append(pproxy_season)
                    ppsm[season] = ppsm_season
            if pproxies:
                pproxy = pproxies[0].concat(pproxies[1:])
        else:
            assert np.isclose(proxy.dt, 1)
            da_truth_for_proxy = annualize_seasonal_data(
                da_truth[psm_original.field].sel(lat=psm_original.lat, lon=psm_original.lon),
                psm_original.seasonality,
            )
            da_truth_truncated_for_proxy = annualize_seasonal_data(
                da_truth_truncated[psm_original.field].sel(
                    lat=psm_original.lat, lon=psm_original.lon
                ),
                psm_original.seasonality,
            )
            pproxy, ppsm = _calibrate_pseudo_psm(
                da_proxy,
                da_truth_for_proxy,
                da_truth_truncated_for_proxy,
                proxy,
                psm_original,
                noise_rng,
                scale_snr,
            )
        if pproxy:
            records[pproxy.pid] = pproxy
            psms[pproxy.pid] = ppsm

    return (
        records,
        psms,
        {
            "calibrated_obs_dataset": calibrated_obs_dataset,
            "mapper_id": mapper_id,
            "scale_snr": scale_snr,
        },
    )


def generate_pseudoobs_from_real_idealized(calibrated_obs_dataset, multiplicity=200, std_noise=1):
    """
    Generate pseudoobs by replicating a sub-sample of a real proxy network but with seasonal resolution,
    constant replication (no seasonal variations), fixed noise level, and identity PSM.
    """
    pdb_original: cfr.ProxyDatabase = pickle.load(
        (get_data_path() / "obs" / calibrated_obs_dataset / "pdb.pkl").open("rb")
    )
    psms_original: dict[str, PSM] = pickle.load(
        (get_data_path() / "obs" / calibrated_obs_dataset / "psms.pkl").open("rb")
    )
    with (get_data_path() / "obs" / calibrated_obs_dataset / "metadata.json").open() as f:
        mapper_id = json.load(f)["mapper_id"]

    records = list(pdb_original.records.items())
    np.random.default_rng(544502).shuffle(records)
    pdb_original = cfr.ProxyDatabase(dict(records[:multiplicity]))

    records = {}
    psms = {}
    for pid, proxy in tqdm(pdb_original.records.items()):
        pproxy = proxy.copy()
        pproxy.pid = f"pseudo_{proxy.pid}"

        psm_original = psms_original[pid]
        field = psm_original.field
        grid_lat, grid_lon = psm_original.lat, psm_original.lon

        da_truth_for_proxy = (
            use_tuple_time_coords(da_truth[field]).sel(lat=grid_lat, lon=grid_lon).dropna("time")
        )

        pp_value = da_truth_for_proxy.values
        pproxy.value = pp_value + noise_rng.normal(0, std_noise, pp_value.shape)
        pproxy.time = use_decimal_year_time_coords(da_truth_for_proxy).time.values

        ppsm = IdentityPSM(field, grid_lat, grid_lon, std_noise, None)

        records[pproxy.pid] = pproxy
        psms[pproxy.pid] = ppsm

    return (
        records,
        psms,
        {
            "calibrated_obs_dataset": calibrated_obs_dataset,
            "mapper_id": mapper_id,
            "std_noise": std_noise,
            "multiplicity": multiplicity,
        },
    )


if __name__ == "__main__":
    type = "real"
    # type = "real_ideal"
    # type = "grid_8"

    # true_dataset = Path() / "cmip6/MPI-ESM1-2-LR/past2k_historical/seasonal_anomalies.zarr"
    # true_dataset = Path() / "cmip6/CESM2/past1000_historical/seasonal_anomalies.zarr"
    # true_dataset = Path() / "cmip6/MRI-ESM2-0/past1000_historical/seasonal_anomalies.zarr"
    true_dataset = Path() / "cmip6/MIROC-ES2L/past1000_historical/seasonal_anomalies.zarr"
    # true_dataset = Path() / "cmip6/CESM2/piControl/seasonal_anomalies.zarr"

    logger.info(f"Loading seasonal anomalies from {true_dataset}")
    da_truth = xr.open_zarr(get_data_path() / true_dataset)[["tas", "tos"]].load()

    if "CESM2/piControl" in str(true_dataset):
        # For CESM piControl (years 1-1200): shift to 800-2000
        da_truth = da_truth.assign_coords(time=da_truth["time"] + 800)

    noise_rng = np.random.default_rng(52565641)

    if type == "real":
        records, psms, metadata = generate_pseudoobs_from_real(sys.argv[1], scale_snr=1)
    elif type == "real_ideal":
        records, psms, metadata = generate_pseudoobs_from_real_idealized(
            sys.argv[1],
            std_noise=np.sqrt(0.4),
            multiplicity=500,
        )
    elif type.startswith("grid_"):
        records, psms, metadata = generate_pseudoobs_grid(sigma_obs=np.sqrt(0.4))
    else:
        raise ValueError("Invalid type")

    pdb = cfr.ProxyDatabase(records)

    directory = get_data_path() / "pseudoobs" / get_timestamp()
    directory.mkdir(parents=True)
    logger.info(f"Saving proxy database to {directory}/pdb.pkl")
    pickle.dump(pdb, (directory / "pdb.pkl").open("wb"))

    logger.info(f"Saving PSMs to {directory}/psms.pkl")
    pickle.dump(
        psms,
        (directory / "psms.pkl").open("wb"),
    )

    json.dump(
        {
            "true_dataset": str(true_dataset),
            "type": type,
        }
        | metadata,
        (directory / "metadata.json").open("w"),
        indent=4,
    )

    logger.info("Computing of pseudo-observations completed")
