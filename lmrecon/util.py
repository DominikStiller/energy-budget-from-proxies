from __future__ import annotations

import contextlib
import itertools
import platform
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Self

import cftime
import dask.array
import numpy as np
import pyleoclim
import xarray as xr
from xarray import DataArray, Dataset

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from pandas import Index, MultiIndex
    from xarray.core.types import T_Xarray

from lmrecon.constants import EARTH_RADIUS_MEAN


def stack_state(ds: T_Xarray) -> DataArray:
    if isinstance(ds, DataArray):
        ds = ds.to_dataset(name="data")

    # Find possible sampling dimensions
    if not (sample_dims := list(set(ds.dims) & {"time", "ens", "case"})):
        sample_dims = []

    stacked = ds.to_stacked_array("state", sample_dims=sample_dims, variable_dim="field", name="")

    if "lat" in ds.dims and "lon" in ds.dims:
        # Force order
        stacked = stacked.reorder_levels(dict(state=["field", "lat", "lon"]))
    return stacked


def unstack_state(da: DataArray) -> Dataset:
    ds = da.to_unstacked_dataset("state")
    if "state" in ds.dims:
        ds = ds.unstack("state")
    if len(ds.data_vars) == 1:
        ds = ds[next(iter(ds.data_vars))]
    return ds


def to_cf_order(da: DataArray) -> DataArray:
    """Transpose DataArray dimensions to follow CF conventions (i.e., sampling dimension first)"""
    if "lat" in da.dims and "lon" in da.dims:
        return da.transpose("case", "time", "ens", "lat", "lon", ..., missing_dims="ignore")
    else:
        return da.transpose("case", "time", "ens", "state", ..., missing_dims="ignore")


def to_math_order(da: DataArray) -> DataArray:
    """Transpose DataArray dimensions to follow mathematical conventions (i.e., sampling dimension last)"""
    if "lat" in da.dims and "lon" in da.dims:
        return da.transpose("lat", "lon", "ens", "time", "case", ..., missing_dims="ignore")
    else:
        return da.transpose("state", "ens", "time", "case", ..., missing_dims="ignore")


def is_dask_array(arr: DataArray | ArrayLike):
    if isinstance(arr, dask.array.Array):
        return True
    elif isinstance(arr, np.ndarray):
        return False
    elif isinstance(arr, DataArray):
        return hasattr(arr.data, "dask")
    else:
        return False


def field_complement(ds: T_Xarray, other_fields: list[str]) -> list[str]:
    """
    Returns all field names that are in the dataset but not in other_fields

    Args:
        ds: the Dataset/DataArray with all fields
        other_fields: the fields to exclude

    Returns:
        the field name complement
    """
    if isinstance(ds, xr.DataArray):
        fields = np.unique(ds.field)
    else:
        fields = ds.keys()
    return list_complement(fields, other_fields)


def list_complement(elements: list, others: list) -> list:
    return list(set(elements) - set(others))


def area_weighted(ds: T_Xarray, square_weights=False) -> T_Xarray:
    """
    Weight the data by the grid cell area, i.e., the cosine of latitude, assuming a regular lat-lon grid.

    Args:
        ds: data
        square_weights: Whether weights should be squared, e.g., when calculating mean of variance

    Returns:
        Weighted data
    """
    weights = np.cos(np.radians(ds[get_position_coords(ds)[0]])).compute()
    if square_weights:
        weights **= 2
    return ds.weighted(weights)


def get_timestamp():
    return datetime.now().replace(microsecond=0).isoformat().replace(":", "-")


def get_host() -> str:
    hostname = platform.node()
    if hostname == "enkf":
        return "enkf"
    elif hostname.startswith(("casper", "crhtc")):
        return "casper"
    elif hostname.startswith(("derecho", "dec")):
        return "derecho"
    elif hostname.startswith(("mbp-dominik", "mac")):
        return "mbp-dominik"
    else:
        raise ValueError("Unknown host")


def get_base_path() -> Path:
    host = get_host()
    if host == "enkf":
        return Path("/home/enkf6/dstiller/")
    elif host in ["casper", "derecho"]:
        return Path("/glade/campaign/univ/uwas0141/")
    elif host == "mbp-dominik":
        return Path("/Users/dstiller/data/lmrecon/")


def get_data_path() -> Path:
    return get_base_path() / "lmrecon"


def get_esgf_path() -> Path:
    host = get_host()
    if host == "mbp-dominik":
        return Path("/Users/dstiller/data/esgf/")
    elif host in ["casper", "derecho"]:
        return Path("/glade/campaign/univ/uwas0141/")


def round_to(x, base, precision=1):
    return np.round(base * np.round(x / base), precision)


def get_position_dims(ds):
    # CAM and other models have lat, lon
    # POP has nlat, nlon
    lat_dim = None
    lon_dim = None
    for dim in ds.dims:
        if dim in ["lat", "nlat", "latitude"]:
            lat_dim = dim
        if dim in ["lon", "nlon", "longitude"]:
            lon_dim = dim
    return lat_dim, lon_dim


def get_position_coords(ds):
    lat_coord = None
    lon_coord = None
    for coord in ds.coords:
        if not lat_coord and coord in ["lat", "latitude", "TLAT", "ULAT"]:
            lat_coord = coord
        if not lon_coord and coord in ["lon", "longitude", "TLONG", "ULONG"]:
            lon_coord = coord
    return lat_coord, lon_coord


def get_spherical_distance(
    lat_base: float, lon_base: float, lat_others: ArrayLike, lon_others: ArrayLike
) -> ArrayLike:
    # Calculate great circle distance using Haversine formula
    # Could also use Vincenty's formula, which accounts for oblateness, but probably not necessary

    lat_base, lon_base, lat_others, lon_others = list(
        map(np.radians, [lat_base, lon_base, lat_others, lon_others])
    )

    dlat = lat_others - lat_base
    dlon = lon_others - lon_base

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_base) * np.cos(lat_others) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    d = EARTH_RADIUS_MEAN / 1e3 * c  # [km]
    assert np.all(d >= 0)
    return d


def get_closest_gridpoint(
    lat_base: float, lon_base: float, lat_others: ArrayLike | Index, lon_others: ArrayLike | Index
) -> tuple[float, float]:
    if lat_others.shape != lon_others.shape:
        # Create Cartesian grid from coordinates
        lat_others, lon_others = list(zip(*list(itertools.product(lat_others, lon_others))))
    distances = get_spherical_distance(lat_base, lon_base, lat_others, lon_others)
    idx_min = np.argmin(distances)
    return lat_others[idx_min], lon_others[idx_min]


def get_closest_gridpoint_with_data(
    lat_base: float, lon_base: float, da: xr.DataArray
) -> tuple[float, float] | None:
    """
    Return closest gridpoint that does not all have nans in da.

    Args:
        lat_base: Latitude for which to find closest gridpoints
        lon_base: Longitude for which to find closest gridpoints
        da: DataArray in which to search for gridpoint

    Returns:
        Coordinates of gridpoint with data that is closest to base
    """
    lat_others = da.lat.data
    lon_others = da.lon.data
    if lat_others.shape != lon_others.shape:
        # Create Cartesian grid from coordinates
        lat_others, lon_others = list(zip(*list(itertools.product(lat_others, lon_others))))
    distances = get_spherical_distance(lat_base, lon_base, lat_others, lon_others)
    idxs = np.argsort(distances)
    for idx in idxs:
        lat, lon = lat_others[idx], lon_others[idx]
        has_data = not np.isnan(da.sel(lat=lat, lon=lon)).all("time").compute().item()
        if has_data:
            return lat, lon


def add_cartesian_coords(da):
    """
    Add Cartesian coordinates in km with the origin at the intersection of the equator and the prime
    meridian. This is useful to calculate spatial frequency/wavenumber spectra.

    Args:
        da: xarray object with lat/lon coordinates in degrees

    Returns:
        xarray object with x/y coordinates in km
    """
    lat, lon = np.meshgrid(da.lat.data, da.lon.data, indexing="ij")
    x = EARTH_RADIUS_MEAN / 1e3 * 2 * np.pi * np.cos(np.radians(lat)) * lon / 360
    y = EARTH_RADIUS_MEAN / 1e3 * np.pi * lat / 180
    return da.assign_coords(x=(["lat", "lon"], x), y=(["lat", "lon"], y))


def has_float_timedim(ds: T_Xarray) -> bool:
    return isinstance(ds["time"].values.flat[0], float | np.floating)


def has_int_timedim(ds: T_Xarray) -> bool:
    return isinstance(ds["time"].values.flat[0], int | np.int64)


def has_tuple_timedim(ds: T_Xarray) -> bool:
    if "time" in ds.coords:
        return isinstance(ds["time"].values.flat[0], tuple)
    else:
        return "year" in ds.coords and "season" in ds.coords


def has_cftime_timedim(ds: T_Xarray) -> bool:
    return isinstance(ds["time"].values.flat[0], cftime.datetime)


def has_npdatetime_timedim(ds: T_Xarray) -> bool:
    return isinstance(ds["time"].values.flat[0], np.datetime64)


def get_state_index(coords: MultiIndex, field: str, lat: float, lon: float) -> int:
    """
    Determine the state vector index given a field and location. The location is chosen by proximity and does not need
    to be exactly in the coords.

    Args:
        coords: MultiIndex with order (field, lat, lon) as created by stack_state()
        field: field to select
        lat: latitude to select
        lon: longitude to select

    Returns:
        Index
    """
    return coords.get_loc(
        (
            field,
            *get_closest_gridpoint(
                lat, lon, coords.get_level_values(1), coords.get_level_values(2)
            ),
        )
    )


def filter_cf_valid(da: xr.DataArray) -> xr.DataArray:
    # Filter invalid values
    # Not supported by xarray (https://github.com/pydata/xarray/issues/8359)
    if "valid_min" in da.attrs:
        da = xr.where(da >= da.attrs["valid_min"], da, np.nan, keep_attrs=True)
    if "valid_max" in da.attrs:
        da = xr.where(da <= da.attrs["valid_max"], da, np.nan, keep_attrs=True)
    return da


def standardize_coordinate_names(da):
    if "latitude" in da.variables:
        da = da.rename(latitude="lat")
    if "longitude" in da.variables:
        da = da.rename(longitude="lon")
    return da


def is_ocean_field(field: str) -> bool:
    return field in ["tos", "siconc", "siconcn", "siconcs", "ohc300", "ohc700"]


def subsample_ensemble(
    ds: xr.Dataset | xr.DataArray, n: int, offset=0
) -> xr.Dataset | xr.DataArray:
    idx_sample = np.random.default_rng(42324235 + offset).choice(ds.ens, n, replace=False)
    return ds.sel(ens=sorted(idx_sample))


class NanMask:
    def __init__(self):
        self.nan_mask: ArrayLike | None = None

    def _validate_input(self, da: ArrayLike):
        if len(da.shape) > 2:
            raise ValueError("Must be one- or two-dimensional (state x sampling)")

    def fit(self, da: ArrayLike) -> Self:
        self._validate_input(da)
        self.nan_mask = np.array(np.isnan(da).any(axis=1))
        return self

    def forward(self, da: ArrayLike) -> ArrayLike:
        self._validate_input(da)
        if self.nan_mask is None:
            self.fit(da)
        return da[~self.nan_mask]

    def backward(self, da: ArrayLike) -> ArrayLike:
        self._validate_input(da)

        shape = (len(self.nan_mask), da.shape[1])
        if is_dask_array(da):
            # Do not set chunks here, rechunk at end
            # Assignment of chunked dask arrays with boolean masks is extremely slow for some reason
            decompressed = dask.array.empty(shape)
        else:
            decompressed = np.empty(shape)

        decompressed[~self.nan_mask] = da
        decompressed[self.nan_mask] = np.nan

        if is_dask_array(da):
            decompressed = decompressed.rechunk(da.chunksize)

        return decompressed


def create_pyleoclim_series(da: xr.DataArray) -> pyleoclim.Series | pyleoclim.EnsembleSeries:
    from lmrecon.time import use_decimal_year_time_coords

    da = use_decimal_year_time_coords(da)
    if "ens" in da.dims:
        return pyleoclim.EnsembleSeries(
            [create_pyleoclim_series(da.sel(ens=ens)) for ens in da.ens], label=da.name
        )
    else:
        return pyleoclim.Series(da.time, da, time_unit="Year (CE)", label=da.name, verbose=False)


@contextlib.contextmanager
def local_np_seed(seed):
    """
    Overwrite numpy RNG seed.

    From https://gist.github.com/VictorDarvariu/6cede9c79900c6215b5f848993d283c6

    Args:
        seed: Seed
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
