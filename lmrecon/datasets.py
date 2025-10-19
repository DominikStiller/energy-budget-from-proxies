from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from lmrecon.constants import SECONDS_PER_YEAR
from lmrecon.stats import anomalize, average_annually, average_seasonally
from lmrecon.time import (
    round_to_nearest_season,
    use_decimal_year_time_coords,
)
from lmrecon.units import (
    calculate_sistatistic,
    calculate_sistatistics,
)
from lmrecon.util import get_base_path

# Explicitly export load_x methods since I often use star imports for lmrecon.datasets
__all__ = [
    "load_andrews2022_feedbacks",
    "load_anna",
    "load_ar6_erf",
    "load_brennan22",
    "load_cdr_sic",
    "load_ceresgm",
    "load_cheng2017",
    "load_cmip6_amip_bcs",
    "load_cobesst",
    "load_combined_eruptions",
    "load_cooper25",
    "load_dalaiden2023",
    "load_deepc",
    "load_era5",
    "load_erbe_wfov",
    "load_erbe_wfov_global_mean",
    "load_ersst",
    "load_evolv2k",
    "load_gebbie2019",
    "load_gistemp",
    "load_hadcrut",
    "load_haden4",
    "load_hadisst",
    "load_iapv4",
    "load_lmronline",
    "load_lmronline_global_mean",
    "load_meng25",
    "load_modera",
    "load_moderac",
    "load_pages19",
    "load_ps2004",
    "load_satire_solar",
    "load_siindex",
    "load_smith2025_eei",
    "load_walsh19",
    "load_wu2025",
    "load_zanna",
]


def load_lmronline(
    version="production_ccsm4_pagesv2_wCoral_gis_seaslinobj_online_20m20sOHC_past1000",
):
    ds_lmronline = xr.open_dataset(
        get_base_path()
        / f"datasets/reconstructions/LMRonline/condensed_reconstructions/{version}/spatial_ens_mean.nc"
    )
    ds_lmronline = ds_lmronline.rename(
        {
            "tas_sfc": "tas",
            "tos_sfc": "tos",
            "rlut_toa": "rlut",
            "rsut_toa": "rsut",
            "ohc_0-700m": "ohc700",
        }
    )[["tas", "tos", "rsut", "rlut", "ohc700"]]
    ds_lmronline["eei"] = -(ds_lmronline["rsut"] + ds_lmronline["rlut"])
    # J -> W
    ds_lmronline["ddtohc700"] = ds_lmronline["ohc700"].differentiate("time") / SECONDS_PER_YEAR
    ds_lmronline = anomalize(ds_lmronline, (1961, 1991))
    return ds_lmronline


def load_lmronline_global_mean(
    version="production_ccsm4_pagesv2_wCoral_gis_seaslinobj_online_20m20sOHC_past1000",
):
    ds_lmronline = xr.merge(
        [
            xr.open_dataset(
                get_base_path()
                / f"datasets/reconstructions/LMRonline/condensed_reconstructions/{version}/{file}_indices.nc"
            )["glob_mean"].rename(var)
            for var, file in zip(
                ["tas", "tos", "rsut", "rlut", "ohc700"],
                ["tas_sfc", "tos_sfc", "rsut_toa", "rlut_toa", "ohc_0-700m"],
            )
        ]
    )
    ds_lmronline = ds_lmronline.rename(year="time")
    ds_lmronline = ds_lmronline.stack(dict(ens=("mc_iteration", "ensemble_member")))
    ds_lmronline = ds_lmronline.assign_coords(dict(ens=range(len(ds_lmronline["ens"]))))
    ds_lmronline["eei"] = -(ds_lmronline["rsut"] + ds_lmronline["rlut"])
    # J -> W
    ds_lmronline["ddtohc700"] = ds_lmronline["ohc700"].differentiate("time") / SECONDS_PER_YEAR
    ds_lmronline = anomalize(ds_lmronline, (1961, 1991))
    return ds_lmronline


def load_anna():
    ds_anna = xr.merge(
        [
            xr.open_dataset(
                get_base_path()
                / "datasets/reconstructions/anna/Spatial_ESRF_1850_2020_Loc_time_Annual_HadGEM3_rlut_toa_Amon_r0.4_nens150loc_rad25000_dlat10.nc"
            )["LMR"].rename("rlut"),
            xr.open_dataset(
                get_base_path()
                / "datasets/reconstructions/anna/Spatial_ESRF_1850_2020_Loc_time_Annual_HadGEM3_rsut_toa_Amon_r0.4_nens150loc_rad25000_dlat10.nc"
            )["LMR"].rename("rsut"),
        ]
    )
    ds_anna = ds_anna.assign_coords(time=ds_anna.time.dt.year.data)
    ds_anna["eei"] = -(ds_anna["rsut"] + ds_anna["rlut"])
    ds_anna = anomalize(ds_anna, (1961, 1991))
    return ds_anna


def load_era5(anomalies=True):
    # ds = xr.open_dataset(get_base_path() / "datasets/temperature/ERA5/data.nc")
    # ds = ds.rename(valid_time="time", latitude="lat", longitude="lon", t2m="tas")
    # ds = ds.drop_vars(["expver", "number"])
    # ds = xe.Regridder(ds, GLOBAL_GRID, method="conservative", periodic=True)(ds)
    # ds = average_seasonally(ds)
    # ds.to_netcdf(get_base_path() / "datasets/temperature/ERA5/data_seasonal.nc")

    # All above
    ds = xr.open_dataset(get_base_path() / "datasets/temperature/ERA5/data_seasonal.nc")
    if anomalies:
        ds = anomalize(ds, (1961, 1991))
    return ds


def load_gistemp(anomalies=True):
    # ds_gistemp = xr.open_dataset(get_base_path() / "datasets/temperature/GISTEMP/gistemp1200_GHCNv4_ERSSTv5.nc")
    # ds_gistemp = ds_gistemp.rename(tempanomaly="tas").drop_vars("time_bnds")
    # ds_gistemp = ds_gistemp.assign_coords(dict(lon=ds_gistemp["lon"] % 360)).sortby("lon")
    # ds_gistemp = average_seasonally(ds_gistemp)
    # # GISTEMP has longitude grid shifted by 1 deg, need to regrid for truncation; error is minimal
    # ds_gistemp = xe.Regridder(ds_gistemp, GLOBAL_GRID, "bilinear", periodic=True)(
    #     ds_gistemp, keep_attrs=True, skipna=True, na_thres=0.75
    # )

    # All above
    ds_gistemp = xr.open_dataset(
        get_base_path() / "datasets/temperature/GISTEMP/gistemp_seasonal.nc"
    ).astype(np.float32)
    if anomalies:
        ds_gistemp = anomalize(ds_gistemp, (1961, 1991))
    return ds_gistemp


def load_ersst(anomalies=True):
    # ds_ersst = xr.open_dataset(get_base_path() / "datasets/temperature/ERSST5/ersstv5.nc")
    # ds_ersst = ds_ersst.rename(sst="tos")[["tos"]]
    # ds_ersst = average_seasonally(ds_ersst)
    # # ERSST has latitude grid shifted by 1 deg, need to regrid for truncation
    # # Bilinear since it is already 2 deg
    # ds_ersst = xe.Regridder(ds_ersst, GLOBAL_GRID, "bilinear", periodic=True)(
    #     ds_ersst, keep_attrs=True, skipna=True, na_thres=0.75
    # ).astype(np.float32)
    # landmask = xr.open_dataset(get_data_path() / "cmip6" / "mmm" / "landmask.nc")["mask"]
    # ds_ersst = ds_ersst.where(~landmask)
    # ds_ersst.to_netcdf(get_base_path() / "datasets/temperature/ERSST5/ersstv5_seasonal.nc")

    # _, climatology_1961_1990 = anomalize(ds_ersst, period=(1961, 1991), return_climatology=True)
    # use_string_season_coords(climatology_1961_1990).compute().to_netcdf(
    #     get_base_path() / "datasets/temperature/ERSST5" / "climatology_1961-1990.nc"
    # )

    # pdo_index = PDOIndex()
    # pdo_index.fit(anomalize(ds_ersst["tos"], (1961, 1991)))
    # pdo_index.save(get_base_path() / "datasets/temperature/ERSST5")

    # All above
    ds_ersst = xr.open_dataset(get_base_path() / "datasets/temperature/ERSST5/ersstv5_seasonal.nc")
    if anomalies:
        ds_ersst = anomalize(ds_ersst, (1961, 1991))
    return ds_ersst


def load_cobesst(anomalies=True):
    # ds = xr.open_dataset(get_base_path() / "datasets/temperature/COBE-SST2/sst.mon.mean.nc")
    # ds = ds.rename(sst="tos")
    # ds = Regridder().regrid(ds, method="conservative").astype(np.float32)
    # landmask = xr.open_dataset(get_data_path() / "cmip6" / "mmm" / "landmask.nc")["mask"]
    # ds = ds.where(~landmask)
    # ds = average_seasonally(ds)
    # ds.to_netcdf(get_base_path() / "datasets/temperature/COBE-SST2/cobesst2_seasonal.nc")

    ds = xr.open_dataset(get_base_path() / "datasets/temperature/COBE-SST2/cobesst2_seasonal.nc")
    if anomalies:
        ds = anomalize(ds, (1961, 1991))
    return ds


def load_hadisst(anomalies=True):
    # ds_hadisst = xr.open_dataset(get_base_path() / "datasets/temperature/HadISST/HadISST_sst.nc")[["sst"]]
    # ds_hadisst = ds_hadisst.rename(latitude="lat", longitude="lon", sst="tos")
    # ds_hadisst = ds_hadisst.where(ds_hadisst["tos"] > -999)
    # ds_hadisst = ds_hadisst.assign_coords(dict(lon=ds_hadisst["lon"] % 360)).sortby(["lat", "lon"])
    # ds_hadisst = average_seasonally(ds_hadisst)
    # ds_hadisst = Regridder().regrid(ds_hadisst, method="conservative").astype(np.float32)
    # landmask = xr.open_dataset(get_data_path() / "cmip6" / "mmm" / "landmask.nc")["mask"]
    # ds_hadisst = ds_hadisst.where(~landmask)
    # ds_hadisst.to_netcdf(get_base_path() / "datasets/temperature/HadISST/hadisst_seasonal.nc")

    ds_hadisst = xr.open_dataset(
        get_base_path() / "datasets/temperature/HadISST/hadisst_seasonal.nc"
    )
    if anomalies:
        ds_hadisst = anomalize(ds_hadisst, (1961, 1991))
    return ds_hadisst


def load_hadcrut():
    ds_hadcrut = xr.open_dataset(
        get_base_path()
        / "datasets/temperature/HadCRUT5/HadCRUT.5.0.2.0.analysis.anomalies.ensemble_mean.nc"
    )[["tas_mean"]]
    ds_hadcrut = ds_hadcrut.rename(latitude="lat", longitude="lon", tas_mean="tas").drop_vars(
        "realization"
    )
    ds_hadcrut = ds_hadcrut.assign_coords(dict(lon=ds_hadcrut["lon"] % 360)).sortby("lon")
    ds_hadcrut = anomalize(average_seasonally(ds_hadcrut), (1961, 1991))
    return ds_hadcrut


def load_zanna():
    ds_zanna = xr.open_dataset(get_base_path() / "datasets/reconstructions/Zanna2019_OHC.nc")[
        ["OHC_300m", "OHC_700m", "OHC_2000m", "OHC_full_depth"]
    ]
    ds_zanna = ds_zanna.rename(
        {
            "time (starting 1870)": "time",
            "OHC_300m": "ohc300",
            "OHC_700m": "ohc700",
            "OHC_2000m": "ohc2000",
            "OHC_full_depth": "ohc",
        }
    )
    ds_zanna = ds_zanna.assign_coords(dict(time=1870 + ds_zanna["time"]))
    ds_zanna = ds_zanna * 1e21  # ZJ -> J
    ds_zanna = anomalize(ds_zanna, (1961, 1990))
    return ds_zanna


def load_wu2025():
    # Data for black line in Fig. 1c in Wu et al. (2025; https://zenodo.org/records/15395211)
    # ds = xr.open_dataset(get_base_path() / "datasets/ocean/Wu2025/GF_OHC_estimate.nc")
    # ds = ds.assign_coords(time=ds.time.dt.year)
    # ds = ds.rename(ENSEMBLE="ens", dep="lev", dep_bnds="lev_bnds")
    # ds = ds.rename(gohr="ddtohc2000", gohc="ohc2000", theta_e="thetao")
    # ds["ohc2000"] = ds["ohc2000"] * 1e21
    # ds["ohc300"] = convert_thetao_to_ohc(ds["thetao"], ds["lev"], ds["lev_bnds"], 300)

    # ds2 = xr.open_dataset("/Users/dstiller/data/lmrecon/datasets/ocean/Wu2025/global_energy_budget.nc")
    # ds2 = ds2.assign_coords(time=ds2.time.dt.year).dropna("time")
    # ds["eei"] = ds2["Nm"]
    # ds["eei_stde"] = ds2["Nd"]
    # ds.to_netcdf(get_base_path() / "datasets/ocean/Wu2025/Wu2025.nc")
    ds = xr.open_dataset(get_base_path() / "datasets/ocean/Wu2025/Wu2025.nc")
    ds["ohc300"] = anomalize(ds["ohc300"], (1961, 1990))
    ds["ohc2000"] = anomalize(ds["ohc2000"], (1961, 1990))
    return ds


def load_gebbie2019():
    # https://www.ncei.noaa.gov/pub/data/paleo/gcmoutput/gebbie2019/
    # ds = xr.open_dataset(get_base_path() / "datasets/ocean/Gebbie2019/Theta_OPT-0015.nc")
    # ds = ds.rename(theta="thetao", depth="lev", latitude="lat", longitude="lon", year="time")[["thetao"]]
    # ds["tos"] = ds["thetao"].sel(lev=0)
    # ds["thetao"] = ds["thetao"].assign_attrs(units="K")
    # ds["lev"] = ds["lev"].assign_attrs(units="m")
    # lev = ds["lev"].copy()
    # # 0 m and 10 m temperature is identical -> treat as one layer
    # # Interpolate 5-year values to annual
    # ds = ds.isel(lev=slice(1, None)).sortby("time").interp(time=np.arange(850, 2001))
    # lev_bnds = xr.DataArray(np.concat([[lev.values[:-1]], [lev.values[1:]]]), dims=["bnds", "lev"])
    # ds["ohc300"] = convert_thetao_to_ohc(ds["thetao"], ds["lev"], lev_bnds, depth=300)
    # ds["ohc700"] = convert_thetao_to_ohc(ds["thetao"], ds["lev"], lev_bnds, depth=700)
    # ds["ohc"] = convert_thetao_to_ohc(ds["thetao"], ds["lev"], lev_bnds, depth=10000)
    # ds.to_netcdf(get_base_path() / "datasets/ocean/Gebbie2019/OPT-0015.nc")
    ds = xr.open_dataset(get_base_path() / "datasets/ocean/Gebbie2019/OPT-0015.nc")
    return ds


def load_cheng2017():
    # http://www.ocean.iap.ac.cn/ftp/cheng/IAP_Ocean_heat_content_0_2000m/
    # ds = open_mfdataset(get_base_path() / "datasets/ocean/IAP_Ocean_heat_content_0_2000m/individual")
    # ds = ds[["OHC300", "OHC2000"]].rename(OHC300="ohc300", OHC2000="ohc2000")
    # ds = ds.where(ds < 0.9e30)
    # ds = ds.assign_coords(time=pd.to_datetime(ds.time.values, format="%Y%m"))
    # ds = average_seasonally(ds)
    # ds.to_netcdf(get_base_path() / "datasets/ocean/IAP_Ocean_heat_content_0_2000m/OHC_seasonal.nc")
    ds = xr.open_dataset(
        get_base_path() / "datasets/ocean/IAP_Ocean_heat_content_0_2000m/OHC_seasonal.nc"
    )
    ds = anomalize(ds, (1961, 1991))
    return ds


def load_iapv4():
    # http://www.ocean.iap.ac.cn/ftp/cheng/IAPv4.2_Ocean_heat_content_0_6000m/
    # ds = open_mfdataset(get_base_path() / "datasets/ocean/IAPv4.2_Ocean_heat_content_0_6000m/individual")
    # ds[["OHC300", "OHC2000", "OHC6000"]].rename(OHC300="ohc300", OHC2000="ohc2000", OHC6000="ohc6000")
    # ds = ds.where(ds < 0.9e30)
    # ds = ds.assign_coords(time=pd.to_datetime(ds.time.values, format="%Y%m"))
    # ds = average_seasonally(ds)
    # ds.to_netcdf(get_base_path() / "datasets/ocean/IAPv4.2_Ocean_heat_content_0_6000m/OHC_seasonal.nc")
    ds = xr.open_dataset(
        get_base_path() / "datasets/ocean/IAPv4.2_Ocean_heat_content_0_6000m/OHC_seasonal.nc"
    )
    ds = anomalize(ds, (1961, 1991))
    return ds


def load_smith2025_eei():
    # https://doi.org/10.5281/zenodo.15639576
    ds = pd.read_csv(
        get_base_path()
        / "datasets"
        / "ClimateIndicator-Smith2025/data/earth_energy_imbalance/earth_energy_imbalance.csv"
    )
    ds["time"] = ds["time"].astype(int)
    ds = ds.set_index("time").to_xarray()
    ds["ohc2000"] = ds["ocean_0-700m"] + ds["ocean_700-2000m"]
    ds = ds.rename({"ocean_0-700m": "ohc700", "ocean_full-depth": "ohc", "total": "ehc"})[
        ["ohc700", "ohc2000", "ohc", "ehc"]
    ]
    ds = ds * 1e21  # ZJ -> J
    return ds


def load_modera():
    ds_modera = xr.open_dataset(
        get_base_path()
        / "datasets/reconstructions/ModE-RA/ModE-RA_ensmean_temp2_anom_wrt_1901-2000_1421-2008_mon.nc"
    )
    ds_modera = ds_modera.rename(latitude="lat", longitude="lon", temp2="tas")
    ds_modera = average_seasonally(ds_modera)
    ds_modera = anomalize(ds_modera, (1961, 1991))
    return ds_modera


def load_moderac():
    ds_modera = xr.open_dataset(
        get_base_path()
        / "datasets/reconstructions/ModE-RA/ModE-RAclim_ensmean_temp2_anom_1421-2008_mon.nc"
    )
    ds_modera = ds_modera.rename(latitude="lat", longitude="lon", temp2="tas")
    ds_modera = average_seasonally(ds_modera)
    ds_modera = anomalize(ds_modera, (1961, 1991))
    return ds_modera


def load_pages19():
    ds_pages19 = xr.open_dataset(
        get_base_path() / "datasets/reconstructions/pages2k_ngeo19_recons.nc"
    )
    ds_pages19 = ds_pages19.drop_vars("LMRv2.1").rename(year="time").sel(time=slice(850, 2000))
    ds_pages19 = anomalize(ds_pages19, (1961, 1991))
    # Merge ensemble members from all methods
    ds_pages19 = (
        ds_pages19.rename(ens="ens_orig")
        .to_array("method")
        .stack(ens=["ens_orig", "method"])
        .to_dataset(name="tas")
    )
    return ds_pages19


def load_brennan22(prior="CCSM4"):
    # Original reconstruction relative to 1979-1999 based on satellite climo
    # But the SIC data come as absolute values
    ds_brennan = xr.open_dataset(
        get_base_path()
        / f"datasets/reconstructions/brennan22/Brennan_and_Hakim_2021_{prior}_model_prior_reconstructions_nh.nc"
    )[["sic_lalo_ensemble_mean", "si_extent_anomalies", "si_area_anomalies"]]
    ds_brennan = ds_brennan.rename_vars(
        {
            "sic_lalo_ensemble_mean": "siconc",
            "si_extent_anomalies": "siextentn",
            "si_area_anomalies": "siarean",
        }
    )
    ds_brennan["siconc"] /= 100
    ds_brennan["siarean"] = anomalize(ds_brennan["siarean"] * 1e6, (1961, 1991))
    ds_brennan["siextentn"] = anomalize(ds_brennan["siextentn"] * 1e6, (1961, 1991))
    ds_brennan = ds_brennan.stack(ens=["nens", "nit"])
    ds_brennan = ds_brennan.assign_coords(ens=np.arange(len(ds_brennan.ens)))
    return ds_brennan


def load_cooper25():
    cooper2025_base = get_base_path() / "datasets/reconstructions/cooper2025"

    # ds_tas = xr.open_dataset(cooper2025_base / "tasanom_ensmean_v0.nc").drop_vars(["height", "wts"])
    # ds_tos = xr.open_dataset(cooper2025_base / "sstanom_ensmean_v0.nc").drop_vars(["wts", "mask"])
    # ds_cooper_temp = average_seasonally(xr.merge([ds_tas, ds_tos]))
    # ds_cooper_temp = Regridder().regrid(ds_cooper_temp)
    # ds_cooper_temp.to_netcdf(cooper2025_base / "cooper25_temp_seasonal.nc")

    # def _load_sic(region, unadjusted):
    #     x = xr.open_dataset(cooper2025_base / f"sic{region}hanom_ensmean_v0.nc").drop_vars(
    #         ["wts"]
    #     ).reset_coords("mask")
    #     if unadjusted:
    #         x = x[["anom_unadjusted"]].rename(anom_unadjusted="siconc")
    #     else:
    #         x = x[["anom"]].rename(anom="siconc")
    #     if region == "n":
    #         region = slice(0, 90)
    #     else:
    #         region = slice(-90, 0)
    #     x = Regridder(GLOBAL_GRID.sel(lat=region)).regrid(x, method="conservative")
    #     del x["lat"].attrs["bounds"]
    #     del x["lon"].attrs["bounds"]
    #     return x

    # ds_cooper_sic_anom = average_seasonally(
    #     xr.concat([_load_sic("s", False), _load_sic("n", False)], dim="lat")
    # ).astype(np.float32)
    # ds_cooper_sic_anom.to_netcdf(cooper2025_base / "cooper25_sic_anom_seasonal.nc")

    # ds_cooper_sic_abs = xr.open_dataset(cooper2025_base / "limDA_ensmean_v0_sstice_ts.nc")
    # ds_cooper_sic_abs = ds_cooper_sic_abs[["ice_cov_prediddle"]].rename(ice_cov_prediddle="siconc")
    # ds_cooper_sic_abs = Regridder().regrid(ds_cooper_sic_abs)
    # ds_cooper_sic_abs = ds_cooper_sic_abs.where(~np.isnan(ds_cooper_sic_anom).any("time"))
    # ds_cooper_sic_abs = average_seasonally(ds_cooper_sic_abs)
    # ds_cooper_sic_abs.to_netcdf(cooper2025_base / "cooper25_sic_abs_seasonal.nc")

    # All above
    ds_cooper_temp = xr.open_dataset(cooper2025_base / "cooper25_temp_seasonal.nc")
    ds_cooper_sic = xr.open_dataset(cooper2025_base / "cooper25_sic_abs_seasonal.nc")
    ds_cooper_temp = anomalize(ds_cooper_temp, (1961, 1991))
    ds_cooper = xr.merge([ds_cooper_temp, ds_cooper_sic, calculate_sistatistics(ds_cooper_sic)])
    return ds_cooper


def load_siindex():
    # Relative to 1979-2000!
    def _load(region):
        x = pd.concat(
            [
                pd.read_csv(
                    get_base_path()
                    / f"datasets/seaice/nsidc_siindex/{region}/monthly/data/{region[0].upper()}_{i:02}_extent_v3.0.csv"
                )
                for i in range(1, 13)
            ]
        )
        x.columns = x.columns.str.strip()
        x = x.rename(
            columns=dict(mo="month", extent=f"siextent{region[0]}", area=f"siarea{region[0]}")
        )
        x = x.drop(columns=["region", "data-type"])
        x["time"] = pd.to_datetime(x[["year", "month"]].assign(day=1))
        x = x.drop(columns=["year", "month"]).set_index("time")
        x = x.replace(-9999, np.nan)
        return x.to_xarray()

    ds_siindex = xr.merge([_load("north"), _load("south")]).sortby("time")
    ds_siindex = anomalize(average_seasonally(ds_siindex), (1979, 2001))
    return ds_siindex


def load_cdr_sic():
    # Absolute!
    # def _load(region):
    #     grid = xr.open_dataset(get_base_path() / f"datasets/seaice/nsidc_sic_cdr/ancillary/G02202-ancillary-ps{region[0]}25-v05r00.nc")
    #     grid = grid.rename({"latitude": "lat", "longitude": "lon"})[["lat", "lon"]]

    #     x = xr.open_dataset(get_base_path() / f"datasets/seaice/nsidc_sic_cdr/{region}/aggregate/sic_ps{region[0]}25_197811-202406_v05r00.nc")
    #     x = x[["cdr_seaice_conc_monthly"]].rename_vars({"cdr_seaice_conc_monthly": "siconc"})
    #     x = x.assign_coords(lat=grid.lat, lon=grid.lon)
    #     x["mask"] = ~np.isnan(x["siconc"]).all("time")
    #     if region == "north":
    #         region = slice(0, 90)
    #     else:
    #         region = slice(-90, 0)
    #     # periodic=True messes with Antarctic sea ice (no SIC between 100E-260E)
    #     x = xe.Regridder(x, GLOBAL_GRID.sel(lat=region), "bilinear", periodic=False)(x)
    #     del x["lat"].attrs["bounds"]
    #     del x["lon"].attrs["bounds"]
    #     x = x.where(x["mask"])
    #     return x.drop_vars("mask")
    # ds_sic = average_seasonally(xr.concat([_load("south"), _load("north")], dim="lat")).astype(np.float32)
    # ds_sic.to_netcdf(get_base_path() / "datasets/seaice/nsidc_sic_cdr/nsidc_sic_cdr_seasonal.nc")

    # _, climatology_1979_2000 = anomalize(ds_sic, period=(1979, 2001), return_climatology=True)
    # use_string_season_coords(climatology_1979_2000).compute().to_netcdf(
    #     get_base_path() / "datasets/seaice/nsidc_sic_cdr/climatology_1979-2000.nc"
    # )

    # All above
    ds_sic = xr.open_dataset(
        get_base_path() / "datasets/seaice/nsidc_sic_cdr/nsidc_sic_cdr_seasonal.nc"
    )
    return xr.merge([ds_sic, calculate_sistatistics(ds_sic)])


def load_walsh19():
    # x = xr.open_dataset(get_base_path() / "datasets/seaice/walsh2019/G10010_sibt1850_v2.0.nc")
    # x = x[["seaice_conc"]].rename(seaice_conc="siconc")
    # x = x.where(x["siconc"] <= 110) / 100  # 120 is land mask
    # x["mask"] = ~np.isnan(x["siconc"]).all("time")
    # x = xe.Regridder(x, GLOBAL_GRID.sel(lat=slice(0, 90)), "bilinear", periodic=False)(x).astype(np.float32)
    # x = x.where(x["mask"])
    # x = x.drop_vars("mask")
    # x = average_seasonally(x)
    # x.to_netcdf(get_base_path() / "datasets/seaice/walsh2019/walsh2019_seasonal.nc")
    ds_sic = xr.open_dataset(get_base_path() / "datasets/seaice/walsh2019/walsh2019_seasonal.nc")
    return xr.merge(
        [
            ds_sic,
            calculate_sistatistic(ds_sic, "siarean"),
            calculate_sistatistic(ds_sic, "siextentn"),
        ]
    )


def load_erbe_wfov():
    files = Path(get_base_path() / "datasets/radiation/ERBS_WFOV/").glob(
        "*/ERBE_S10N_WFOV_SF_ERBS_Regional_Edition4.1.72DayAnnual_*.nc"
    )
    dss = []
    for f in sorted(files):
        ds = xr.open_dataset(f, decode_cf=False)
        ds = ds.sel(time=ds.time >= 0)
        ds = xr.decode_cf(ds)
        dss.append(ds)
    ds = xr.concat(dss, "time")

    ds = ds.rename(
        TOA_SW_FLUX="rsut",
        TOA_LW_FLUX="rlut",
        TOA_INCOMING_SOLAR="rsdt",
        TOA_NET_FLUX="eei",
    )[["rsut", "rlut", "rsdt", "eei"]]

    ds = use_decimal_year_time_coords(ds)
    ds = ds.assign_coords(time=ds["time"].astype(int))
    # Cannot use 1961-1990 average since data start after
    # Use 1999 instead of 2000 since time is int
    ds = anomalize(ds, (1985, 1999))
    return ds


def load_erbe_wfov_global_mean():
    ds = xr.open_dataset(
        get_base_path()
        / "datasets/radiation/ERBS_WFOV/ERBE_S10N_WFOV_SF_ERBS_AreaAverageTimeSeries_Edition4.1.72DayAnnualGlobalFilled.nc"
    )
    ds = ds.rename(
        TOA_SW_FLUX="rsut",
        TOA_LW_FLUX="rlut",
        TOA_INCOMING_SOLAR="rsdt",
        TOA_NET_FLUX="eei",
    )[["rsut", "rlut", "rsdt", "eei"]].isel(column_name=0)

    ds = ds.assign_coords(time=ds["time"].astype(int))
    # Cannot use 1961-1990 average since data start after
    ds = anomalize(ds, (1985, 1999))
    return ds


def load_deepc(annual=False):
    # ds = xr.merge([
    #     xr.open_dataset(get_base_path() / "datasets/radiation/DEEP-C/DEEP-C_TOA_NET_v5_1985-2020.nc"),
    #     xr.open_dataset(get_base_path() / "datasets/radiation/DEEP-C/DEEP-C_TOA_ASR_v5_1985-2020.nc"),
    #     xr.open_dataset(get_base_path() / "datasets/radiation/DEEP-C/DEEP-C_TOA_OLR_v5_1985-2020.nc"),
    # ])
    # ds["rsut"] = 1361/4 - ds["ASR"]
    # ds["OLR"] = -ds["OLR"]  # OLR is positive downward??
    # ds = ds.rename(dict(NET="eei", OLR="rlut")).drop("ASR")
    # ds = Regridder().regrid(ds, method="conservative")

    # All above
    ds = xr.open_dataset(get_base_path() / "datasets/radiation/DEEP-C/DEEP-C_seasonal.nc")
    ds = average_annually(ds, remove_incomplete=False) if annual else average_seasonally(ds)
    ds = anomalize(ds, (1985, 1999))
    return ds


def load_ceres():
    # For global means, use load_ceresgm()
    ds = xr.open_dataset(get_base_path() / "datasets/radiation/CERES/CERES_EBAF-TOA_Ed4.2.1.nc")
    ds = ds.rename(
        toa_sw_all_mon="rsut",
        toa_lw_all_mon="rlut",
        toa_net_all_mon="eei",
        solar_mon="rsdt",
        cldarea_total_daynight_mon="clt",
    )[["eei", "rsut", "rlut", "rsdt", "clt"]]
    return ds


def load_ceresgm():
    # Global mean uses geodetic weights (accounting for non-sphericity of earth)
    # -> use these instead of calculating global mean myself from gridded CERES data
    # See https://ceres.larc.nasa.gov/data/general-product-info/#geodetic-zone-weights-information
    ds = xr.open_dataset(
        get_base_path() / "datasets/radiation/CERES/CERES_EBAF-TOA_Ed4.2.1-globalmean.nc"
    )
    ds = ds.rename(
        gtoa_sw_all_mon="rsut",
        gtoa_lw_all_mon="rlut",
        gtoa_net_all_mon="eei",
        gsolar_mon="rsdt",
        gcldarea_total_daynight_mon="clt",
    )[["eei", "rsut", "rlut", "rsdt", "clt"]]
    return ds


def load_haden4():
    # client = Client(n_workers=dask.system.CPU_COUNT // 2, threads_per_worker=1)
    # ds = open_mfdataset(get_base_path() / "datasets/ocean/HadEN4-g10")
    # ds = ds.rename(depth="lev")
    # assert ds["temperature"].attrs["units"] == "kelvin"
    # ds["temperature"] = ds["temperature"].assign_attrs(units="K")
    # assert ds["lev"].attrs["units"] == "metres"
    # ds["lev"] = ds["lev"].assign_attrs(units="m")
    # ds = xr.merge([
    #     convert_thetao_to_ohc(ds["temperature"], ds["depth_bnds"], 300),
    #     convert_thetao_to_ohc(ds["temperature"], ds["depth_bnds"], 700),
    # ])
    # ds = average_seasonally(ds)
    # ds = Regridder().regrid(ds, method="conservative")
    # All Above
    ds = xr.open_dataset(get_base_path() / "datasets/ocean/HadEN4-g10_ohc_seasonal.nc")
    ds = anomalize(ds, (1961, 1991))
    return ds


def load_ps2004():
    # Pollack & Smerdon (2004) borehole NH(!!) temperatures
    df = pd.read_csv(
        get_base_path() / "datasets/temperature/fig6.10_ipccar4_wg1_2007.unsmoothed.txt",
        skiprows=14217,
        nrows=501,
        delim_whitespace=True,
        names=["time", "tas"],
        index_col="time",
    )
    ds = df.to_xarray()
    ds = anomalize(ds, (1961, 1991))
    return ds


def load_satire_solar():
    # tsi = xr.open_dataset(
    #     get_base_path() / "datasets/solar/SSI_14C_cycle_yearly_cmip_v20160613_fc.nc",
    #     decode_times=False,
    # )
    # tsi = (tsi["ssi"] * tsi["wavelength_bin"]).sum("wavelength")
    # tsi = average_annually(tsi, weight_months=False).to_dataset(name="tsi")
    # All above
    return xr.open_dataset(get_base_path() / "datasets/solar/SSI_14C_fc_annual.nc")


def load_meng25(prior="CCSM4"):
    ds = xr.open_mfdataset(
        str(
            get_base_path()
            / f"datasets/reconstructions/LMRseasonal/{prior}/ensemble_mean/*_mean.nc"
        )
    )
    ds = use_decimal_year_time_coords(ds)
    ds = ds.assign_coords(time=round_to_nearest_season(ds.time)).sel(time=slice(850, None))
    ds = ds.drop_vars(
        ["Nino1+2", "Nino3", "Nino4", "IOD", "IOBW", "GMOHC", "GMT", "NH-Temp", "SH-Temp"],
        errors="ignore",
    )
    if "siextentn" in ds:
        ds = ds.rename_vars(dict(Nino34="nino34", SIE="siextentn"))
        ds["siextentn"] = ds["siextentn"] / 1e6  # m^2 -> km^2
    ds = anomalize(ds, (1961, 1991))
    return ds


def load_evolv2k():
    df = pd.read_csv(
        get_base_path() / "datasets/volcanic/Sigl-Toohey_2024_eVolv2k_v4.tab", sep="\t", skiprows=48
    )
    df.columns = [
        "year",
        "yearCE",
        "month",
        "day",
        "lat",
        "so4Greenland",
        "so4Antarctic",
        "vssi",
        "sigma_vssi",
        "asy",
        "loc",
        "tephra",
        "ref",
    ]
    df = df.drop(["yearCE", "so4Greenland", "so4Antarctic", "tephra", "ref"], axis=1).iloc[::-1]
    df = df[(df["year"] >= 1)]
    # pd.to_datetime doesn't work with s resolution
    df["time"] = df.apply(lambda r: f"{r.year:04}-{r.month:02}-{r.day:02}", axis=1).astype(
        "datetime64[s]"
    )
    df = df.set_index("time").drop(["year", "month", "day"], axis=1)
    return df.to_xarray()


def load_combined_eruptions(vssi_min=0.5):
    ds_evolv2k = load_evolv2k()
    ds_evolv2k = ds_evolv2k.drop_vars(["sigma_vssi", "asy"])
    ds_evolv2k = ds_evolv2k.sel(time=slice(np.datetime64("0850"), None))
    ds_evolv2k = ds_evolv2k.sel(time=ds_evolv2k["vssi"] >= vssi_min)

    ds = xr.open_dataset(
        "/Users/dstiller/data/esgf/input4MIPs/CMIP7/CMIP/uoexeter/UOEXETER-CMIP-2-2-1/atmos/day/utsvolcemis/gn/v20250521/utsvolcemis_input4MIPs_emissions_CMIP_UOEXETER-CMIP-2-2-1_gn_17500101-20231120.nc"
    )
    ds = ds.sel(eruption_number=ds.time >= pd.to_datetime("1900"))
    ds["vssi"] = ds["utsvolcemis"] * 32.065 / 64.066 / 1e9
    ds = ds.drop_vars(["utsvolcemis", "time_bnds", "depth", "height"])
    # Filter small eruptions
    ds = ds.sel(eruption_number=ds["vssi"] >= vssi_min)
    ds = ds.set_xindex("time").drop_vars("eruption_number").rename(eruption_number="time")
    ds["time"] = ds["time"].astype("datetime64[s]")
    ds = ds.reset_coords(["lat", "lon"])

    return use_decimal_year_time_coords(
        xr.concat(
            [ds_evolv2k, ds],
            dim="time",
        )
    )


def load_ar6_erf():
    x = (
        xr.concat(
            [
                pd.read_csv(
                    get_base_path() / "datasets/radiation/AR6/AR6_ERF_1750-2019.csv",
                    index_col="year",
                ).to_xarray(),
                pd.read_csv(
                    get_base_path() / "datasets/radiation/AR6/AR6_ERF_1750-2019_pc05.csv",
                    index_col="year",
                ).to_xarray(),
                pd.read_csv(
                    get_base_path() / "datasets/radiation/AR6/AR6_ERF_1750-2019_pc95.csv",
                    index_col="year",
                ).to_xarray(),
            ],
            dim="value",
        )
        .rename(year="time")
        .assign_coords(value=["mean", "p05", "p95"])
    )
    return x


def load_andrews2022_feedbacks():
    # Only use CMIP6 models
    models = [
        "CESM2",
        "CanESM5",
        "CNRM-CM6-1",
        "HadGEM3-GC31-LL",
        "IPSL-CM6A-LR",
        "MIROC6",
        "MRI-ESM2-0",
    ]
    ds = xr.concat(
        [
            xr.open_dataset(
                get_base_path()
                / f"datasets/radiation/Andrews2022/amip-piForcing_{model}_GlobalAnnualEnsembleMean.nc"
            )
            .drop_vars(
                ["time", "lat", "lon", "lat_bnds", "lon_bnds", "time_bnds", "height"],
                errors="ignore",
            )
            .rename(year="time")
            .set_xindex("time")
            for model in models
        ],
        dim="model",
    ).assign_coords(model=models)
    ds = ds.rename(dt="tas", dn="eei")[["tas", "eei"]]
    return ds


def load_cmip6_amip_bcs(seasonal=True):
    # Loads the pre-diddle BCs (monthly averages, not mid-monthly values)
    # from lmrecon.util import get_esgf_path

    # base = get_esgf_path() / "input4MIPs/CMIP6/CMIP/PCMDI/PCMDI-AMIP-1-1-6"
    # ds = xr.merge(
    #     [
    #         xr.open_dataset(
    #             base
    #             / "ocean/mon/tos/gn/v20191121/tos_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-6_gn_187001-201812.nc"
    #         ),
    #         xr.open_dataset(
    #             base
    #             / "seaIce/mon/siconc/gn/v20191121/siconc_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-6_gn_187001-201812.nc"
    #         ),
    #         xr.open_dataset(
    #             base
    #             / "ocean/fx/sftof/gn/v20191121/sftof_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-6_gn.nc"
    #         ),
    #     ]
    # )
    # ds = ds.where(ds["sftof"] > 50)[["tos", "siconc"]]
    # if seasonal:
    #     ds = xe.Regridder(ds, GLOBAL_GRID, method="conservative", periodic=True)(ds)
    #     ds = average_seasonally(ds)
    #     ds.to_netcdf(get_base_path() / "datasets/ocean/amip_seasonal.nc")
    # else:
    #     ds = xe.Regridder(ds, load_mask_ocean(), method="conservative", periodic=True)(ds)
    #     ds.to_netcdf(get_base_path() / "datasets/ocean/amip_monthly.nc")

    if seasonal:
        ds = xr.open_dataset(get_base_path() / "datasets/ocean/amip_seasonal.nc")
        return anomalize(ds, (1961, 1991))
    else:
        ds = xr.open_dataset(get_base_path() / "datasets/ocean/amip_monthly.nc")
        return ds


def load_dalaiden2023():
    ds_dalaiden2023 = xr.open_dataset(
        "/Users/dstiller/data/lmrecon/datasets/seaice/Dalaiden2023/Dalaiden2023_sic-antarctic.nc",
        decode_times=False,
    )
    ds_dalaiden2023 = ds_dalaiden2023.assign_coords(time=np.arange(1700, 2001))
    ds_dalaiden2023 = ds_dalaiden2023.rename({"SIC": "siconc"})[["siconc"]]
    ds_dalaiden2023["siconc"] = ds_dalaiden2023["siconc"] / 100
    ds_dalaiden2023 = xr.merge([ds_dalaiden2023, calculate_sistatistics(ds_dalaiden2023)])
    return ds_dalaiden2023
