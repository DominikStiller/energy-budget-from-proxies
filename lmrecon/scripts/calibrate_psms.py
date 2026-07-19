from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from cfr import ProxyDatabase
from matplotlib import pyplot as plt
from tqdm import tqdm

from lmrecon.datasets import load_ersstv6, load_gistemp
from lmrecon.logger import get_logger
from lmrecon.mapper import PhysicalSpaceForecastSpaceMapper
from lmrecon.plotting import save_plot
from lmrecon.psm import PSM, LinearPSM
from lmrecon.stats import annualize_seasonal_data, average_seasonally
from lmrecon.time import (
    Season,
    use_tuple_time_coords,
)
from lmrecon.util import (
    get_base_path,
    get_closest_gridpoint_with_data,
    get_data_path,
    get_spherical_distance,
    get_timestamp,
)

logger = get_logger(__name__)


MINIMUM_CALIBRATION_OVERLAP = 25
MINIMUM_CORR_THRESHOLD = 0.10
MAXIMUM_ANNUAL_ERROR_ACOR_THRESHOLD = 0.9
MAXIMUM_CALIBRATION_DISTANCE = 500  # km

SEASONALITY_CANDIDATES = [
    [Season.DJF, Season.MAM, Season.JJA, Season.SON],
    [Season.SON, Season.DJF, Season.MAM, Season.JJA],
    [Season.JJA, Season.SON, Season.DJF, Season.MAM],
    [Season.MAM, Season.JJA, Season.SON, Season.DJF],
    [Season.JJA],
    [Season.MAM, Season.JJA],
    [Season.JJA, Season.SON],
    [Season.DJF],
    [Season.SON, Season.DJF],
    [Season.DJF, Season.MAM],
]


def get_calibration_field_for_proxy_type(ptype: str) -> str:
    # Marine proxies are assimilated as SST, all others as TAS
    if ptype.startswith(("marine", "coral", "bivalve")):
        return "tos"
    else:
        return "tas"


def seasonalize_subseasonal_proxies(pdb):
    """Average all proxies with temporal resolution below seasonal to seasonal resolution"""
    seasonalized_proxies = {}
    for key, proxy in tqdm(pdb.records.items()):
        # Some timestamps are for beginning, but more for center of interval
        # -> don't shift time axis to center interval
        # Would require detection of which convention is used
        da = xr.DataArray(proxy.value, coords=dict(time=proxy.time))
        da_seasonal = average_seasonally(da, weight_months=False)

        proxy = proxy.copy()
        proxy.time = da_seasonal.time.data
        proxy.value = da_seasonal.data
        proxy.dt = 1 / 4

        seasonalized_proxies[key] = proxy

    return ProxyDatabase(seasonalized_proxies)


def annualize_biannual_proxies(pdb):
    """Average all proxies with temporal resolution of biannual to annual resolution"""
    annual_proxies = {}
    for key, proxy in tqdm(pdb.records.items()):
        da = xr.DataArray(proxy.value, coords=dict(time=proxy.time))

        # Ensure only full years are included, otherwise the annual average has a strong seasonal signal
        groupby = da.groupby(da.time.astype(int))
        groups = list(groupby.groups.values())
        da_annual = groupby.mean()
        if len(groups[0]) < 2:
            da_annual = da_annual.isel(time=slice(1, None))
        if len(groups[-1]) < 2:
            da_annual = da_annual.isel(time=slice(None, -1))

        proxy = proxy.copy()
        proxy.time = da_annual.time.data
        proxy.value = da_annual.data
        proxy.dt = 1

        annual_proxies[key] = proxy

    return ProxyDatabase(annual_proxies)


def ensure_annual_resolution(pdb):
    """
    Ensure that all nominally annual-resolution proxies only have one data point per year.
    Some annual proxies have two values in some years, or have non-integer timestamps. This also
    averages proxies with nominally two values per year (biannual).
    """
    annual_proxies = {}
    for key, proxy in tqdm(pdb.records.items()):
        da = xr.DataArray(proxy.value, coords=dict(time=proxy.time))
        da_annual = da.groupby(da.time.astype(int)).mean()

        proxy = proxy.copy()
        proxy.time = da_annual.time.data
        proxy.value = da_annual.data
        proxy.dt = 1

        annual_proxies[key] = proxy

    return ProxyDatabase(annual_proxies)


def _get_calibration_gridpoint(proxy, ds_field) -> tuple[float, float] | None:
    """Find the nearest grid point with data, returning None if it is too far away."""
    grid_lat, grid_lon = get_closest_gridpoint_with_data(proxy.lat, proxy.lon, ds_field)
    distance = get_spherical_distance(proxy.lat, proxy.lon, grid_lat, grid_lon)
    if distance > MAXIMUM_CALIBRATION_DISTANCE:
        # May happen for some Antarctic proxies (if calibration data is not infilled) or for
        # Red Sea corals (if land in mapper and thus doesn't have tos)
        logger.info(
            f"Removing {proxy.pid} ({proxy.ptype}): nearest calibration data point is "
            f"{distance:.0f} km away (> {MAXIMUM_CALIBRATION_DISTANCE} km) "
            f"(proxy: lat={proxy.lat:.2f}, lon={proxy.lon:.2f}, "
            f"calib. data: lat={grid_lat:.2f}, lon={grid_lon:.2f})"
        )
        return None
    return grid_lat, grid_lon


def _check_calibration_quality(psm: LinearPSM, pid: str, field: str, context: str = "") -> bool:
    """Return True if the calibrated PSM passes correlation and error autocorrelation checks."""
    suffix = f" for {pid} (field {field}{context})"
    if np.abs(psm.corr) < MINIMUM_CORR_THRESHOLD:
        logger.info(
            f"Insufficient correlation (|{psm.corr:.3f}| < {MINIMUM_CORR_THRESHOLD:.3f}){suffix}"
        )
        return False
    if np.abs(psm.annual_error_acor) > MAXIMUM_ANNUAL_ERROR_ACOR_THRESHOLD:
        logger.info(
            f"Excessive error autocorrelation (|{psm.annual_error_acor:.3f}| > "
            f"{MAXIMUM_ANNUAL_ERROR_ACOR_THRESHOLD:.3f}){suffix}"
        )
        return False
    return True


def calibrate_seasonal_psms(
    pdb_seasonal, ds_calib_seasonal, output_directory
) -> tuple[dict[str, dict[Season, PSM]], ProxyDatabase]:
    psms: dict[str, dict[Season, PSM]] = {}
    proxies_to_remove = []

    for pid, proxy in tqdm(pdb_seasonal.records.items()):
        da_proxy = use_tuple_time_coords(xr.DataArray(proxy.value, coords=dict(time=proxy.time)))

        field = get_calibration_field_for_proxy_type(proxy.ptype)
        # Find data for closest grid point at overlapping times
        # Need to convert decimal years to year + season label to prevent issues with floating point alignment
        result = _get_calibration_gridpoint(proxy, ds_calib_seasonal[field])
        if result is None:
            proxies_to_remove.append(proxy)
            continue
        grid_lat, grid_lon = result

        psms[pid] = {}
        for season in [Season.DJF, Season.MAM, Season.JJA, Season.SON]:
            da_calib_for_proxy, proxy_values_for_calib = xr.align(
                ds_calib_seasonal[field]
                .sel(lat=grid_lat, lon=grid_lon, season=season)
                .dropna("year"),
                da_proxy.sel(season=season).dropna("year"),
                join="inner",
            )

            if len(da_calib_for_proxy) < MINIMUM_CALIBRATION_OVERLAP:
                # Should not remove directly from records since records is being iterated over
                logger.info(
                    f"Insufficient calibration data ({len(da_calib_for_proxy)} samples) for {pid} (field {field})"
                )
                continue

            psm = LinearPSM(pid, field, grid_lat, grid_lon, season)
            psm.calibrate(
                da_calib_for_proxy.data, proxy_values_for_calib.data, da_calib_for_proxy.year.data
            )

            if not _check_calibration_quality(psm, pid, field):
                continue

            psms[pid][season] = psm

            # Plot diagnostics
            # da_calib_for_proxy = use_decimal_year_time_coords(da_calib_for_proxy)
            # fig, ax = plt.subplots()
            # ax.plot(da_calib_for_proxy.time, proxy_values_for_calib, label="Real proxy")
            # ax.plot(da_calib_for_proxy.time, psm.forward(da_calib_for_proxy)[0, :], label="PSM")
            # ax.set_xlabel("Year CE")
            # ax.set_ylabel(f"{proxy.value_name} ({proxy.value_unit})")
            # ax.legend()
            # ax.set_title(f"{proxy.pid} ({proxy.ptype}, SNR = {psm.SNR:.2f})")
            # save_plot(output_directory / "plots" / "proxies", pid)

        if not psms[pid]:
            # Unsuccessful calibration for any season
            proxies_to_remove.append(proxy)
            del psms[pid]

    return psms, pdb_seasonal - proxies_to_remove


def calibrate_annual_psms(
    pdb_annual, ds_calib_seasonal, output_directory
) -> tuple[dict[str, PSM], ProxyDatabase]:
    psms: dict[str, PSM] = {}
    proxies_to_remove = []

    for pid, proxy in tqdm(pdb_annual.records.items()):
        da_proxy = xr.DataArray(proxy.value, coords=dict(time=proxy.time))

        field = get_calibration_field_for_proxy_type(proxy.ptype)
        result = _get_calibration_gridpoint(proxy, ds_calib_seasonal[field])
        if result is None:
            proxies_to_remove.append(proxy)
            continue
        grid_lat, grid_lon = result

        psm_candidates = []
        for seasonality in SEASONALITY_CANDIDATES:
            da_calib_annual = annualize_seasonal_data(ds_calib_seasonal[field], seasonality)
            da_calib_for_proxy, proxy_values_for_calib = xr.align(
                da_calib_annual.sel(lat=grid_lat, lon=grid_lon).dropna("time"),
                da_proxy.dropna("time"),
                join="inner",
            )

            szn = " ".join(Season.to_str_list(seasonality))
            if len(da_calib_for_proxy) < MINIMUM_CALIBRATION_OVERLAP:
                logger.info(
                    f"Insufficient calibration data (less than {MINIMUM_CALIBRATION_OVERLAP} samples) for {pid} (field {field}, seasonality = {szn})"
                )
                continue

            psm = LinearPSM(pid, field, grid_lat, grid_lon, seasonality)
            psm.calibrate(
                da_calib_for_proxy.data, proxy_values_for_calib.data, da_calib_for_proxy.time.data
            )

            if not _check_calibration_quality(psm, pid, field, f", seasonality = {szn}"):
                continue

            psm_candidates.append(psm)

        if not psm_candidates:
            # Insufficient calibration data or correlation
            proxies_to_remove.append(proxy)
            continue

        # Select PSM with lowest BIC
        idx_opt = np.argmin([psm.BIC for psm in psm_candidates])
        psm_opt = psm_candidates[idx_opt]
        psms[pid] = psm_opt

        # Plot diagnostics
        # fig, ax = plt.subplots()
        # ax.plot(da_calib_for_proxy.time, proxy_values_for_calib, label="Real proxy")
        # for psm in psm_candidates:
        #     da_calib_annual = annualize_seasonal_data(ds_calib_seasonal[psm.field], psm.seasonality)
        #     da_calib_for_proxy, proxy_values_for_calib = xr.align(
        #         da_calib_annual.sel(lat=grid_lat, lon=grid_lon).dropna("time"),
        #         da_proxy.dropna("time"),
        #         join="inner",
        #     )

        #     szn = " ".join(Season.to_str_list(psm.seasonality))
        #     ax.plot(
        #         da_calib_for_proxy.time,
        #         psm.forward(da_calib_for_proxy)[0, :],
        #         label=f"PSM ({psm.field}, {szn}, BIC={psm.BIC:.1f}, SNR={psm.SNR:.2f})",
        #         alpha=1 if psm == psm_opt else 0.7,
        #         ls="-" if psm == psm_opt else ":"
        #     )
        # ax.set_xlabel("Time CE")
        # ax.set_ylabel(f"{proxy.value_name} ({proxy.value_unit})")
        # ax.legend(bbox_to_anchor=(0.5, -0.65), loc="lower center", ncols=2)
        # ax.set_title(f"{proxy.pid} ({proxy.ptype})")
        # save_plot(output_directory / "plots" / "proxies", pid)

    return psms, pdb_annual - proxies_to_remove


def plot_snrs(psms, output_directory):
    fig, ax = plt.subplots()

    snr = []
    for psm_or_dict in psms.values():
        if isinstance(psm_or_dict, dict):
            snr.append(np.median([psm.SNR for psm in psm_or_dict.values()]))
        else:
            snr.append(psm_or_dict.SNR)

    ax.hist(snr, bins=30)
    ax.set_xlabel("SNR")
    ax.set_ylabel("# Proxies")
    save_plot(output_directory / "plots", "snr_histogram")


if __name__ == "__main__":
    logger.setLevel("WARN")

    mapper_id = sys.argv[1]

    output_directory = get_data_path() / "obs" / get_timestamp()

    print("Loading mapper for truncation")
    mapper = PhysicalSpaceForecastSpaceMapper.load(
        get_data_path() / "mapper" / mapper_id / "mapper.pkl"
    )

    print("Loading proxies")
    pdb_path = Path("datasets/proxies/combined_temperature.pkl")
    pdb = pickle.load((get_base_path() / pdb_path).open("rb"))
    # Remove BCE years since they are confusing (1 BCE is encoded as -0)
    pdb = pdb.slice((1, np.inf))

    # Filter out proxies that don't overlap with the calibration data -> helps to speed up PSM calibration
    pdb = ProxyDatabase(
        {
            pid: pobj
            for pid, pobj in pdb.records.items()
            if len(pobj.time) > 0 and pobj.time[-1] > 1850
        }
    )

    pdb.plot()
    save_plot(output_directory / "plots", "proxies_full")

    # Standardize proxies so that we can compare calibration slopes
    pdb = pdb.standardize(ref_period=None)

    assert np.all([p.time_unit == "yr" for p in pdb.records.values()])
    n_subannual_incompatible = len(
        (pdb.filter(by="dt", keys=(0.26, 0.49)) + pdb.filter(by="dt", keys=(0.51, 0.99))).records
    )
    if n_subannual_incompatible > 0:
        logger.warning(
            f"Discarding {n_subannual_incompatible} proxies since they have an incompatible sub-annual resolution"
        )

    pdb_multiannual = pdb.filter(by="dt", keys=(1.01, np.inf))
    logger.warning(
        f"Discarding {len(pdb_multiannual.records)} proxies since they have multiannual resolution"
    )
    pdb_multiannual.plot()

    print("Converting subseasonal proxies to seasonal averages")
    pdb_subseasonal = pdb.filter(by="dt", keys=(0, 0.24))
    pdb_seasonal = seasonalize_subseasonal_proxies(pdb_subseasonal) + pdb.filter(
        by="dt", keys=(0.24, 0.26)
    )
    pdb_seasonal.plot()
    save_plot(output_directory / "plots", "proxies_seasonal")

    pdb_biannual = pdb.filter(by="dt", keys=(0.49, 0.51))
    pdb_annual = ensure_annual_resolution(
        pdb.filter(by="dt", keys=(0.99, 1.01))
    ) + annualize_biannual_proxies(pdb_biannual)
    pdb_annual.plot()
    save_plot(output_directory / "plots", "proxies_annual")
    save_plot(output_directory / "plots", "proxies_multiannual")

    print("Loading and truncating calibration datasets (GISTEMP and ERSSTv6)")
    # Truncate datasets to include representativeness error in PSM variance
    ds_calib_seasonal = use_tuple_time_coords(
        mapper.truncate_dataset(
            # These are already 1961-1990 anomalies
            xr.merge(
                [
                    load_gistemp().sel(time=slice(1900, 2001)),
                    load_ersstv6().sel(time=slice(None, 2001)),
                ]
            )
        )
    ).compute()

    print("Calibrating seasonal proxy PSMs")
    psms_seasonal, pdb_seasonal = calibrate_seasonal_psms(
        pdb_seasonal, ds_calib_seasonal, output_directory
    )

    print("Calibrating annual proxy PSMs")
    psms_annual, pdb_annual = calibrate_annual_psms(pdb_annual, ds_calib_seasonal, output_directory)

    print(f"Saving proxy database to {output_directory}/pdb.pkl")
    pickle.dump(pdb_seasonal + pdb_annual, (output_directory / "pdb.pkl").open("wb"))

    print(f"Saving PSMs to {output_directory}/psms.pkl")
    psms = psms_seasonal | psms_annual
    pickle.dump(psms, (output_directory / "psms.pkl").open("wb"))

    plot_snrs(psms, output_directory)

    json.dump(
        {
            "mapper_id": mapper_id,
            "pdb_path": str(pdb_path),
            "minimum_calibration_overlap": MINIMUM_CALIBRATION_OVERLAP,
            "minimum_corr_threshold": MINIMUM_CORR_THRESHOLD,
            "maximum_annual_error_acor_threshold": MAXIMUM_ANNUAL_ERROR_ACOR_THRESHOLD,
            "maximum_calibration_distance": MAXIMUM_CALIBRATION_DISTANCE,
        },
        (output_directory / "metadata.json").open("w"),
        indent=4,
    )

    print("Calibration of PSMs completed")
