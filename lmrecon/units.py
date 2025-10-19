from __future__ import annotations

import numpy as np
import pint_xarray  # noqa: F401
import xarray as xr
import xesmf as xe
from pint_xarray import unit_registry as ureg

from lmrecon.constants import EARTH_RADIUS_MEAN, SURFACE_AREA_EARTH, SURFACE_AREA_OCEAN
from lmrecon.logger import get_logger

logger = get_logger(__name__)


def convert_to_si_units(da: xr.DataArray):
    # Input may be an index which cannot be modified in place
    units = da.units

    with xr.set_options(keep_attrs=True):
        if units == "degC":
            da = da + 273.15
            da.attrs["units"] = "K"
        elif units == "0.001":
            da = da / 1000
            da.attrs["units"] = "1"
        elif units == "%":
            da = da / 100
            da.attrs["units"] = "1"
        elif units == "centimeters":
            da = da / 100
            da.attrs["units"] = "m"
        elif units == "hPa":
            da = da * 1e2
            da.attrs["units"] = "Pa"

    return da


def convert_thetaot_to_ohc(thetaot_avg: xr.DataArray, depth: float) -> xr.DataArray:
    """
    Calculates the ocean heat content from the average potential temperature in a given layer.

    Args:
        thetaot_avg: average potential temperature [K]
        depth: depth over which the given potential temperature is the average [m]

    Returns:
        Ocean heat content [J/m^2]
    """
    assert thetaot_avg.units == "K"

    rho = 1025  # kg/m^3
    cp = 3850  # J/(kg K)

    ohc = rho * cp * thetaot_avg * depth
    ohc = ohc.assign_attrs(dict(units="J m-2"))

    return ohc


def convert_thetao_to_ohc(
    thetao: xr.DataArray, lev: xr.DataArray, lev_bnds: xr.DataArray, depth: float
) -> xr.DataArray:
    """
    Calculates the ocean heat content from the 3D potential temperature field.

    Args:
        thetao: potential temperature [K]
        lev: levels [m]
        lev_bnds: level bounds [m]
        depth: depth over which to integrate [m]

    Returns:
        Ocean heat content [J/m^2]
    """
    assert thetao.units == "K"
    if "units" in lev_bnds.attrs:
        assert lev_bnds.units == "m"
    if "units" in lev.attrs:
        assert lev.units == "m"

    rho = 1025  # kg/m^3
    cp = 3850  # J/(kg K)

    # Weight by layer thickness
    weights = (lev_bnds.isel(bnds=1) - lev_bnds.isel(bnds=0)).astype(thetao.dtype)

    land_mask = np.isnan(thetao).all("lev")

    # Need to use float32(rho * cp), otherwise everything is cast to float64
    # Use where instead of slice in case the levels are time-varying
    ohc = np.float32(rho * cp) * thetao.where(lev <= depth).weighted(weights).sum("lev")
    ohc = ohc.where(~land_mask)
    ohc = ohc.assign_attrs(dict(units="J m-2")).rename(f"ohc{depth}")

    return ohc


def convert_ohc_per_ocean_area_to_total_ohc(ohc_per_area: xr.DataArray) -> xr.DataArray:
    """
    Converts the ocean heat content in J/m^2 to J by multiplying with Earth's ocean area
    """
    if ohc_per_area.pint.units:
        return (ohc_per_area * (SURFACE_AREA_OCEAN * ureg("m^2"))).pint.to("ZJ")
    else:
        return ohc_per_area * SURFACE_AREA_OCEAN


def convert_ohc_per_surface_area_to_total_ohc(ohc_per_area: xr.DataArray) -> xr.DataArray:
    """
    Converts the ocean heat content in J/m^2 to J by multiplying with Earth's surface area
    """
    if ohc_per_area.pint.units:
        return (ohc_per_area * (SURFACE_AREA_EARTH * ureg("1e6 km^2"))).pint.to("ZJ")
    else:
        return ohc_per_area * SURFACE_AREA_EARTH


def convert_total_ohc_to_ohc_per_surface_area(total_ohc: xr.DataArray) -> xr.DataArray:
    """
    Converts the total ocean heat content in J to J/m^2 defined over the whole surface area,
    not just the ocean
    """
    return total_ohc / SURFACE_AREA_EARTH


def calculate_sistatistic(da: xr.Dataset | xr.DataArray, field: str) -> xr.DataArray:
    """
    Calculate the sea ice extent/area based on the field name.

    Args:
        da: DataArray containing absolute sea ice concentration (not anomalies!)
        field: The name of the sea ice statistic (siarean, siareas, siextentn, or siextents)

    Returns:
        DataArray of sea ice extent/area
    """
    if isinstance(da, xr.Dataset):
        da = da["siconc"]
    match field:
        case "siarean":
            return calculate_siarea(da.where(da.lat > 0)).rename("siarean")
        case "siareas":
            return calculate_siarea(da.where(da.lat < 0)).rename("siareas")
        case "siextentn":
            return calculate_siextent(da.where(da.lat > 0)).rename("siextentn")
        case "siextents":
            return calculate_siextent(da.where(da.lat < 0)).rename("siextents")
        case _:
            raise ValueError("Invalid statistic")


def calculate_sistatistics(da: xr.Dataset | xr.DataArray) -> xr.DataArray:
    """
    Calculate sea ice extent/area.

    Args:
        da: DataArray containing absolute sea ice concentration (not anomalies!)

    Returns:
        DataArray of sea ice extent/area
    """
    return xr.merge(
        [
            calculate_sistatistic(da, field)
            for field in ["siarean", "siareas", "siextentn", "siextents"]
        ]
    )


def calculate_siextent(da: xr.DataArray) -> xr.DataArray:
    """
    Calculate the sea ice extent from the sea ice concentration field. The sea ice extent is defined
    as the area where the sea ice concentration is greater than 15%.

    Args:
        da: DataArray containing absolute sea ice concentration (not anomalies!)

    Returns:
        DataArray of sea ice extent
    """
    grid_area = xe.util.cell_area(da.to_dataset(name="ds"), earth_radius=EARTH_RADIUS_MEAN / 1e3)
    return grid_area.where(da >= 0.15).sum(["lat", "lon"]).rename("siextent")


def calculate_siarea(da: xr.DataArray) -> xr.DataArray:
    """
    Calculate the sea ice area from the sea ice concentration field.

    Args:
        da: DataArray containing absolute sea ice concentration (not anomalies!)

    Returns:
        DataArray of sea ice area
    """
    grid_area = xe.util.cell_area(da.to_dataset(name="ds"), earth_radius=EARTH_RADIUS_MEAN / 1e3)
    # NSIDC also filter cells by SIC >= 15% for SI area, not just extent
    # (https://nsidc.org/sites/default/files/g02135-v003-userguide_1_1.pdf, Section 3.1.6)
    # However, every other source does not threshold (e.g., Notz 2014) so we don't either
    return (da * grid_area).sum(["lat", "lon"]).rename("siarea")
