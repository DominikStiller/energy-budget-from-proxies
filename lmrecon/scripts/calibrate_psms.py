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

from lmrecon.datasets import load_ersst, load_gistemp
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
    get_timestamp,
)

logger = get_logger(__name__)


MINIMUM_CALIBRATION_OVERLAP = 25
MINIMUM_CORR_THRESHOLD = 0.05
MAXIMUM_ANNUAL_ERROR_ACOR_THRESHOLD = 0.9

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


def get_candidate_fields_for_proxy_type(ptype: str) -> list[str]:
    # Marine proxies may be assimilated as SST or TAS, all others just as TAS
    if ptype.startswith(("marine", "coral", "bivalve")):
        return ["tas", "tos"]
    else:
        return ["tas"]


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


def ensure_annual_resolution(pdb):
    """
    Ensure that all nominally annual-resolition proxies only have one data point per year.
    Some annual proxies have two values in some years, or have non-integer timestamps.
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


def calibrate_seasonal_psms(
    pdb_seasonal, ds_calib_seasonal, output_directory
) -> tuple[dict[str, dict[Season, PSM]], ProxyDatabase]:
    psms: dict[str, dict[Season, PSM]] = {}
    proxies_to_remove = []

    for pid, proxy in tqdm(pdb_seasonal.records.items()):
        da_proxy = use_tuple_time_coords(xr.DataArray(proxy.value, coords=dict(time=proxy.time)))
        psms[pid] = {}

        for season in [Season.DJF, Season.MAM, Season.JJA, Season.SON]:
            psm_candidates = []

            for field in get_candidate_fields_for_proxy_type(proxy.ptype):
                # Find data for closest grid point at overlapping times
                # Need to convert decimal years to year + season label to prevent issues with floating point alignment
                grid_lat, grid_lon = get_closest_gridpoint_with_data(
                    proxy.lat, proxy.lon, ds_calib_seasonal[field]
                )
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
                psm.calibrate(da_calib_for_proxy.data, proxy_values_for_calib.data)

                if np.abs(psm.corr) < MINIMUM_CORR_THRESHOLD:
                    logger.info(
                        f"Insufficient correlation (|{psm.corr:.3f}| < {MINIMUM_CORR_THRESHOLD:.3f}) for {pid} (field {field})"
                    )
                    continue

                if np.abs(psm.annual_error_acor) > MAXIMUM_ANNUAL_ERROR_ACOR_THRESHOLD:
                    logger.info(
                        f"Excessive error autocorrelation (|{psm.annual_error_acor:.3f}| > {MAXIMUM_ANNUAL_ERROR_ACOR_THRESHOLD:.3f}) for {pid} (field {field})"
                    )
                    continue

                psm_candidates.append(psm)

            if not psm_candidates:
                # Insufficient calibration data or correlation
                continue

            # Select PSM with lowest BIC
            idx_opt = np.argmin([psm.BIC for psm in psm_candidates])
            psm_opt = psm_candidates[idx_opt]
            psms[pid][season] = psm_opt

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

        psm_candidates = []
        for field in get_candidate_fields_for_proxy_type(proxy.ptype):
            grid_lat, grid_lon = get_closest_gridpoint_with_data(
                proxy.lat, proxy.lon, ds_calib_seasonal[field]
            )

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
                psm.calibrate(da_calib_for_proxy.data, proxy_values_for_calib.data)

                if np.abs(psm.corr) < MINIMUM_CORR_THRESHOLD:
                    logger.info(
                        f"Insufficient correlation (|{psm.corr:.3f}| < {MINIMUM_CORR_THRESHOLD:.3f}) for {pid} (field {field}, seasonality = {szn})"
                    )
                    continue

                if np.abs(psm.annual_error_acor) > MAXIMUM_ANNUAL_ERROR_ACOR_THRESHOLD:
                    logger.info(
                        f"Excessive error autocorrelation (|{psm.annual_error_acor:.3f}| > {MAXIMUM_ANNUAL_ERROR_ACOR_THRESHOLD:.3f}) for {pid} (field {field}, seasonality = {szn})"
                    )
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
    mapper_id = sys.argv[1]

    output_directory = get_data_path() / "obs" / get_timestamp()

    logger.info("Loading mapper for truncation")
    mapper = PhysicalSpaceForecastSpaceMapper.load(
        get_data_path() / "mapper" / mapper_id / "mapper.pkl"
    )

    logger.info("Loading proxies")
    pdb_path = Path("datasets/proxies/combined.pkl")
    pdb = pickle.load((get_base_path() / pdb_path).open("rb"))
    # Remove BCE years since they are confusing (1 BCE is encoded as -0)
    pdb = pdb.slice((1, 2020))

    pdb.plot()
    save_plot(output_directory / "plots", "proxies_full")

    assert np.all([p.time_unit == "yr" for p in pdb.records.values()])
    n_seasonal_to_annual = len(pdb.filter(by="dt", keys=(0.26, 0.99)).records)
    if n_seasonal_to_annual > 0:
        logger.warning(
            f"Discarding {n_seasonal_to_annual} proxies since they have a resolution between seasonal and annual"
        )

    pdb_multiannual = pdb.filter(by="dt", keys=(1.01, np.inf))
    logger.warning(
        f"Discarding {len(pdb_multiannual.records)} proxies since they have multiannual resolution"
    )
    pdb_multiannual.plot()

    logger.info("Converting subseasonal proxies to seasonal averages")
    pdb_subseasonal = pdb.filter(by="dt", keys=(0, 0.24))
    pdb_seasonal = seasonalize_subseasonal_proxies(pdb_subseasonal) + pdb.filter(
        by="dt", keys=(0.24, 0.26)
    )
    pdb_seasonal.plot()
    save_plot(output_directory / "plots", "proxies_seasonal")

    pdb_annual = ensure_annual_resolution(pdb.filter(by="dt", keys=(0.99, 1.01)))
    pdb_annual.plot()
    save_plot(output_directory / "plots", "proxies_annual")
    save_plot(output_directory / "plots", "proxies_multiannual")

    logger.info("Loading and truncating calibration datasets (GISTEMP and ERSST)")
    # Truncate datasets to include representativeness error in PSM variance
    ds_calib_seasonal = use_tuple_time_coords(
        mapper.truncate_dataset(
            # These data are already 1961-1990 anomalies
            xr.merge([load_gistemp().sel(time=slice(1900, None)), load_ersst()])
        )
    ).compute()

    logger.info("Calibrating seasonal proxy PSMs")
    psms_seasonal, pdb_seasonal = calibrate_seasonal_psms(
        pdb_seasonal, ds_calib_seasonal, output_directory
    )

    logger.info("Calibrating annual proxy PSMs")
    psms_annual, pdb_annual = calibrate_annual_psms(pdb_annual, ds_calib_seasonal, output_directory)

    logger.info(f"Saving proxy database to {output_directory}/pdb.pkl")
    pickle.dump(pdb_seasonal + pdb_annual, (output_directory / "pdb.pkl").open("wb"))

    logger.info(f"Saving PSMs to {output_directory}/psms.pkl")
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
        },
        (output_directory / "metadata.json").open("w"),
        indent=4,
    )

    logger.info("Calibration of PSMs completed")
