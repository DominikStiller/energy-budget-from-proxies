from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from lmrecon.time import use_monthly_npdatetime_time_coords


def load_cesm_timeseries(
    path,
    variables=[
        "TREFHT",
        "FSNT",
        "FLNT",
        "SOLIN",
        "FSUTOA",
        "FLUT",
        "FLDT",
        "FSNTC",
        "FLNTC",
        "FSNS",
        "FSDS",
        "PRECC",
        "PRECL",
        "CLOUD",
        "CLDLOW",
        "CLDMED",
        "CLDHGH",
    ],
    component="atm",
    frequency="month_1",
    npdatetime_coords=True,
):
    files = []
    for var in variables:
        files.extend(Path(f"{path}/{component}/proc/tseries/{frequency}/").glob(f"*.{var}.*.nc"))
    ds = xr.open_mfdataset(
        sorted(files),
        parallel=True,
        data_vars="minimal",
        coords="minimal",
        compat="override",
        use_cftime=True,
        # CESM timeseries have size 1 netcdf chunks in time, which creates too many Dask tasks
        chunks=dict(time=-1),
    )
    ds = fix_cesm_timestamp(ds)
    if npdatetime_coords:
        ds = use_monthly_npdatetime_time_coords(ds)
    if "lev" in ds.dims and component == "atm":
        ds = ds.assign_coords(lev=ds["lev"] * 1e2)
    if component == "ocn":
        ds = apply_mask_ocean(ds)

    return ds


def find_cesm_history_files(path, component="atm", h="h0", year_start=None, year_end=None):
    all_files = sorted(Path(f"{path}/{component}/hist/").glob(f"*.{h}.*.nc"))
    if year_start is None:
        files = all_files
    else:
        files = []
        for f in all_files:
            year = int(f.name.split(".")[-2].split("-")[0])
            if year_start <= year <= year_end:
                files.append(f)

    ds = xr.open_mfdataset(files, coords="minimal", compat="override", parallel=True)
    return ds


def fix_cesm_timestamp(ds):
    """
    Fix CESM timestamp since by default it is for the first day of the next month.
    This method moves them to the middle of the averaging period.

    Args:
        ds: dataset with timestamps from CESM output

    Returns:
        dataset with timestamps shifted half a month backwards
    """
    if isinstance(ds, xr.Dataset) and (
        "time_bnds" in ds.variables or "time_bound" in ds.variables or "time_bounds" in ds.variables
    ):
        if "time_bnds" in ds.variables:
            time_bounds = ds["time_bnds"]
        elif "time_bound" in ds.variables:
            time_bounds = ds["time_bound"]
        else:
            time_bounds = ds["time_bounds"]

        # Assume timestamps are the same across ens dimension
        if "ens" in time_bounds.dims:
            time_bounds = time_bounds.sel(ens=0)
        time_bounds = time_bounds.values

        # POP2 TEMP's lower bound for the first timestep has hour=2 and millisecond=3 for some reason
        time_bounds[0, 0] = time_bounds[0, 0].replace(hour=0, minute=0, second=0, microsecond=0)

        period_middle = time_bounds[:, 0] + (time_bounds[:, 1] - time_bounds[:, 0]) / 2
        return ds.assign_coords(time=period_middle)
    else:
        print("time_bnds not available, falling back to naive method (assuming monthly data)")

        def _fix(time):
            time = time.item()
            if time.month == 1:
                time = time.replace(year=time.year - 1, month=12)
            else:
                time = time.replace(month=time.month - 1)

            day = 1 + time.daysinmonth // 2
            hour = 0 if time.daysinmonth % 2 == 0 else 12
            return time.replace(day=day, hour=hour)

        return ds.assign_coords(time=list(map(_fix, ds["time"])))


def apply_mask_ocean(ds):
    if "SST" in ds or "ICEFRAC" in ds:
        deg = np.median(np.diff(ds.lon))
        mask_ocean = load_mask_ocean() if deg > 1.5 else load_mask_ocean_1deg()
        assert np.isclose(ds.lat, mask_ocean.lat).all()
        assert np.isclose(ds.lon, mask_ocean.lon).all()
        # Alignment may fail otherwise due to floating point error
        mask_ocean = mask_ocean.assign_coords(lat=ds.lat, lon=ds.lon)
        if "SST" in ds:
            ds["SST"] = ds["SST"].where(mask_ocean)
        if "ICEFRAC" in ds:
            ds["ICEFRAC"] = ds["ICEFRAC"].where(mask_ocean)
    return ds


def load_mask_land():
    mask_land = xr.open_dataset(
        "/glade/campaign/cesm/cesmdata/cseg/inputdata/share/domains/domain.lnd.fv1.9x2.5_gx1v7.181205.nc"
    )["mask"]
    mask_coords = xr.open_dataset(
        "/glade/campaign/cesm/cesmdata/inputdata/atm/cam/topo/fv_1.9x2.5_nc3000_Nsw084_Nrs016_Co120_Fi001_ZR_GRNL_031819.nc"
    )
    mask_land = mask_land.rename(dict(xc="lon", yc="lat", nj="lat", ni="lon"))
    mask_land = mask_land.assign_coords(dict(lat=mask_coords.lat, lon=mask_coords.lon)).astype(bool)
    return mask_land


def load_mask_ocean():
    mask_ocean = xr.open_dataset(
        "/glade/campaign/cesm/cesmdata/cseg/inputdata/share/domains/domain.ocn.fv1.9x2.5_gx1v7.181205.nc"
    )
    mask_coords = xr.open_dataset(
        "/glade/campaign/cesm/cesmdata/inputdata/atm/cam/topo/fv_1.9x2.5_nc3000_Nsw084_Nrs016_Co120_Fi001_ZR_GRNL_031819.nc"
    )
    mask_ocean = mask_ocean["mask"].rename(dict(nj="lat", ni="lon", xc="lon", yc="lat"))
    mask_ocean = mask_ocean.assign_coords(dict(lat=mask_coords.lat, lon=mask_coords.lon)).astype(
        bool
    )
    return mask_ocean


def load_land_ocean_frac():
    mask_ocean = xr.open_dataset(
        "/glade/campaign/cesm/cesmdata/cseg/inputdata/share/domains/domain.ocn.fv1.9x2.5_gx1v7.181205.nc"
    )
    mask_coords = xr.open_dataset(
        "/glade/campaign/cesm/cesmdata/inputdata/atm/cam/topo/fv_1.9x2.5_nc3000_Nsw084_Nrs016_Co120_Fi001_ZR_GRNL_031819.nc"
    )
    oceanfrac = mask_ocean["frac"].rename(dict(nj="lat", ni="lon", xc="lon", yc="lat"))
    oceanfrac = oceanfrac.assign_coords(dict(lat=mask_coords.lat, lon=mask_coords.lon))
    oceanfrac = xr.merge([oceanfrac.rename("oceanfrac"), (1 - oceanfrac).rename("landfrac")])
    return oceanfrac


def load_mask_ocean_1deg():
    mask_ocean = xr.open_dataset(
        "/glade/campaign/cesm/cesmdata/cseg/inputdata/share/domains/domain.ocn.fv0.9x1.25_gx1v7.151020.nc"
    )
    mask_ocean = mask_ocean["mask"].rename(dict(nj="lat", ni="lon", xc="lon", yc="lat"))
    mask_ocean = mask_ocean.astype(bool)
    return mask_ocean


def add_derived_fields(ds: xr.Dataset) -> xr.Dataset:
    if "SOLIN" in ds and "FSNT" in ds:
        ds["FSUT"] = ds["SOLIN"] - ds["FSNT"]
    if "FSNT" in ds and "FLNT" in ds:
        ds["RESTOM"] = ds["FSNT"] - ds["FLNT"]
    if "FSUTOA" in ds:
        if "FLUTOA" in ds:
            ds["RESTOA"] = ds["SOLIN"] - ds["FSUTOA"] - ds["FLUTOA"]
        elif "FLUT" in ds:
            ds["RESTOA"] = ds["SOLIN"] - ds["FSUTOA"] - ds["FLUT"]
    if "FSNTC" in ds and "FLNTC" in ds:
        ds["RESTOMC"] = ds["FSNTC"] - ds["FLNTC"]
        ds["CRE"] = ds["RESTOM"] - ds["RESTOMC"]
        ds["SWCRE"] = ds["FSNT"] - ds["FSNTC"]
        ds["LWCRE"] = -(ds["FLNT"] - ds["FLNTC"])
    if "PRECC" in ds:
        ds["PREC"] = ds["PRECC"] + ds["PRECL"]
    if "FSNS" in ds and "FSDS" in ds:
        ds["ALBSFC"] = 1 - ds["FSNS"] / ds["FSDS"]
    return ds


def map_cesm_to_cf_field(field: str | list[str]) -> str | list[str]:
    if isinstance(field, list):
        return [map_cesm_to_cf_field(f) for f in field]
    return {
        "TREFHT": "tas",
        "SST": "tos",
        "RESTOM": "eei",
        "RESTOA": "eei",
        "FSUTOA": "rsut",
        "FSUT": "rsut",
        "FLUTOA": "rlut",
        "FSNTOA": "rsnt",
        "FLUT": "rlut",
        "SOLIN": "rsdt",
        "CLDLOW": "cll",
        "CLDMED": "clm",
        "CLDHGH": "clh",
        "ICEFRAC": "siconc",
    }.get(field, field)


def map_cf_to_cesm_field(field: str | list[str]) -> str | list[str]:
    if isinstance(field, list):
        return [map_cf_to_cesm_field(f) for f in field]
    return {
        "tas": "TREFHT",
        "tos": "SST",
        "eei": "RESTOM",
        "rsut": "FSUTOA",
        "rlut": "FLUTOA",
        "rsnt": "FSNTOA",
        "rsdt": "SOLIN",
        "cll": "CLDLOW",
        "clm": "CLDMED",
        "clh": "CLDHGH",
        "siconc": "ICEFRAC",
    }.get(field, field)
