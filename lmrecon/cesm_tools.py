from __future__ import annotations

from pathlib import Path

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
        "PRECC",
        "PRECL",
        "CLOUD",
        "CLDLOW",
        "CLDMED",
        "CLDHGH",
    ],
    component="atm",
    frequency="month_1",
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
    ds = use_monthly_npdatetime_time_coords(ds)
    ds = ds.assign_coords(lev=ds["lev"] * 1e2)
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


def add_derived_fields(ds: xr.Dataset) -> xr.Dataset:
    ds["FLDT"] = ds["FLNT"] - ds["FLUT"]
    ds["FSUT"] = ds["SOLIN"] - ds["FSNT"]
    ds["RESTOM"] = ds["FSNT"] - ds["FLNT"]
    ds["RESTOA"] = ds["SOLIN"] - ds["FSUTOA"] - ds["FLUT"]
    if "FSNTC" in ds:
        ds["RESTOMC"] = ds["FSNTC"] - ds["FLNTC"]
        ds["CRE"] = ds["RESTOM"] - ds["RESTOMC"]
    if "PRECC" in ds:
        ds["PREC"] = ds["PRECC"] + ds["PRECL"]
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
