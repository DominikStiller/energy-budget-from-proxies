from __future__ import annotations

import pickle

import numpy as np
import xarray as xr
from cfr import ProxyDatabase
from tqdm import tqdm

from lmrecon.datasets import load_gistemp, load_pdsi
from lmrecon.logger import get_logger
from lmrecon.psm import LinearPSM
from lmrecon.scripts.calibrate_psms import SEASONALITY_CANDIDATES
from lmrecon.stats import annualize_seasonal_data
from lmrecon.util import (
    get_base_path,
    get_closest_gridpoint_with_data,
    is_coordinate_inside_box,
    remove_empty_proxies,
)

logger = get_logger(__name__)


def filter_drought_sensitive_trw(pdb, margin_percent=10):
    """Remove drought-sensitive TRW proxies, which may have spurious correlations with temperature"""
    # Use untruncated versions for both since we cannot truncate PDSI
    da_tas = load_gistemp()["tas"].sel(time=slice(2001)).compute()
    da_pdsi = load_pdsi()["pdsi"].sel(time=slice(2001)).compute()
    da_tas, da_pdsi = xr.align(da_tas, da_pdsi)

    pdb_trw = pdb.filter("ptype", "tree.TRW")
    drought_sensitive_proxies = set()
    for proxy in tqdm(pdb_trw.records.values()):
        da_proxy = xr.DataArray(proxy.value, coords=dict(time=proxy.time))
        grid_lat_tas, grid_lon_tas = get_closest_gridpoint_with_data(proxy.lat, proxy.lon, da_tas)
        # Closest non-nan grid point can be different for PDSI
        grid_lat_pdsi, grid_lon_pdsi = get_closest_gridpoint_with_data(
            proxy.lat, proxy.lon, da_pdsi
        )

        psm_candidates_tas = []
        psm_candidates_pdsi = []
        for seasonality in SEASONALITY_CANDIDATES:
            da_tas_annualized = annualize_seasonal_data(
                da_tas.sel(lat=grid_lat_tas, lon=grid_lon_tas), seasonality
            )
            da_calib_for_proxy, proxy_values_for_calib = xr.align(
                da_tas_annualized.dropna("time"),
                da_proxy.dropna("time"),
                join="inner",
            )
            calib_time = da_calib_for_proxy.time.data
            da_calib_for_proxy = da_calib_for_proxy.data / da_calib_for_proxy.std().compute().item()
            proxy_values_for_calib = (
                proxy_values_for_calib.data / proxy_values_for_calib.std().compute().item()
            )
            psm_tas = LinearPSM("", "tas", grid_lat_tas, grid_lon_tas, seasonality)
            psm_tas.calibrate(da_calib_for_proxy, proxy_values_for_calib, calib_time)
            psm_candidates_tas.append(psm_tas)

            da_pdsi_annualized = annualize_seasonal_data(
                da_pdsi.sel(lat=grid_lat_pdsi, lon=grid_lon_pdsi), seasonality
            )
            da_calib_for_proxy, proxy_values_for_calib = xr.align(
                da_pdsi_annualized.dropna("time"),
                da_proxy.dropna("time"),
                join="inner",
            )
            calib_time = da_calib_for_proxy.time.data
            da_calib_for_proxy = da_calib_for_proxy.data / da_calib_for_proxy.std().compute().item()
            proxy_values_for_calib = (
                proxy_values_for_calib.data / proxy_values_for_calib.std().compute().item()
            )
            psm_pdsi = LinearPSM("", "tas", grid_lat_pdsi, grid_lon_pdsi, seasonality)
            psm_pdsi.calibrate(da_calib_for_proxy, proxy_values_for_calib, calib_time)
            psm_candidates_pdsi.append(psm_pdsi)

        tas_opt = psm_candidates_tas[np.argmin([psm.BIC for psm in psm_candidates_tas])]
        pdsi_opt = psm_candidates_pdsi[np.argmin([psm.BIC for psm in psm_candidates_pdsi])]

        if np.abs(pdsi_opt.corr) * (1 + margin_percent / 100) > np.abs(tas_opt.corr):
            drought_sensitive_proxies.add(proxy)

    logger.warning(f"Discarding {len(drought_sensitive_proxies)} drought-sensitive TRW proxies")
    return pdb - drought_sensitive_proxies


def filter_paired_d18O_corals(pdb):
    corals = pdb.filter("ptype", "coral").records
    paired_d18O_corals = set()
    for pid in corals:
        if pid.startswith("ch2k") and pid.endswith("SrCa"):
            pid_d18O = pid.replace("SrCa", "d18O")
            if pid_d18O in corals:
                paired_d18O_corals.add(corals[pid_d18O])

    return pdb - paired_d18O_corals


def restrict_proxies(pdb):
    records = pdb.records
    for pid, proxy in records.items():
        if proxy.ptype == "coral.d18O" and is_coordinate_inside_box(
            proxy.lat, proxy.lon, (-10, 10), (160, 210)
        ):
            # Coral d18O in tropical central Pacific only before 1970
            # This avoids strong salinity-driven trend post-1970 during calibration and assimilation
            # See Tierney et al. (2015; doi:10.1002/2014PA002717) and Nurhati et al. (2009; doi:10.1029/2009GL040270)
            records[pid] = proxy.slice((1, 1970))
    return ProxyDatabase(records)


if __name__ == "__main__":
    pdb_path = get_base_path() / "datasets" / "proxies"
    pdb: ProxyDatabase = pickle.load((pdb_path / "combined.pkl").open("rb"))

    # Remove proxies that are not temperature-sensitive
    # Jess: "lake scanning of pigments is not at all a T proxy — that's a lake record from
    # South America that is notorious for propagating into reconstructions and producing weird trends"
    pdb = pdb - pdb.filter(by="ptype", keys="lake.reflectance")
    # Gemini says it's not a good temperature proxy
    pdb = pdb - pdb.filter(by="ptype", keys="lake.accumulation")
    # Jess: "coral calcification is not a good T proxy"
    pdb = pdb - pdb.filter(by="ptype", keys="coral.calc")
    # Ant_027 is annually resolved but may have too strong influence if not properly assimilated
    # (accounting for its diffusive nature)
    pdb = pdb - pdb.filter(by="ptype", keys="borehole")
    # More precipitation than temperature; also issues with Afr_012
    pdb = pdb - pdb.filter(by="ptype", keys="speleothem.d18O")

    logger.info("Removing drought-sensitive TRW proxies")
    pdb = filter_drought_sensitive_trw(pdb)
    logger.info("Removing paired d18O coral proxies")
    pdb = filter_paired_d18O_corals(pdb)
    pdb = restrict_proxies(pdb)
    pdb = remove_empty_proxies(pdb)

    logger.info(f"Saving proxy database to {pdb_path / 'combined_temperature.pkl'}")
    pickle.dump(pdb, (pdb_path / "combined_temperature.pkl").open("wb"))
