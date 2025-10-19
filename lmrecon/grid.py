from __future__ import annotations

import numpy as np
import regionmask
import spharm
import xarray as xr
import xesmf as xe
from xgcm import Grid

GLOBAL_GRID = xe.util.grid_global(2, 2, lon1=359, cf=True).drop_vars(["latitude_longitude"])

# Standard CMIP6 pressure levels
# https://cmip6dr.github.io/Data_Request_Home/Documents/CMIP6_pressure_levels.pdf
CMIP6_PLEV27 = (
    np.array(
        [
            1000,
            975,
            950,
            925,
            900,
            875,
            850,
            825,
            800,
            775,
            750,
            700,
            650,
            600,
            550,
            500,
            450,
            400,
            350,
            300,
            250,
            225,
            200,
            175,
            150,
            125,
            100,
        ]
    )
    * 1e2
)
CMIP6_PLEV19 = (
    np.array(
        [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 5, 1]
    )
    * 1e2
)
CMIP6_PLEV7c = np.array([900, 740, 620, 500, 375, 245, 90]) * 1e2


class Regridder:
    def __init__(self, target_grid=GLOBAL_GRID):
        self.regridders = {}  # realm -> regridder
        self.target_grid = target_grid

    def regrid(self, ds, realm="", method="bilinear", periodic=True, extrapolate=False):
        if realm not in self.regridders:
            # Since all fields of the same realm will have the same grid, we can reuse the regridder
            self.regridders[realm] = xe.Regridder(
                ds,
                self.target_grid,
                method,
                periodic=periodic,
                extrap_method="inverse_dist" if extrapolate else None,
            )
        # Adaptive masking (skipna=True) converts it to float64
        return self.regridders[realm](ds, keep_attrs=True, skipna=True, na_thres=0.75).astype(
            np.float32
        )


def mask_greenland_and_antarctica(da: xr.DataArray) -> xr.DataArray:
    # Select Greenland and Antarctica regions from IPCC AR6
    # See https://regionmask.readthedocs.io/en/latest/defined_scientific.html#land
    mask = (
        regionmask.defined_regions.ar6.land.mask_3D(GLOBAL_GRID)
        .sel(region=[0, 44, 45])
        .any(dim="region")
        .drop_vars("latitude_longitude")
    )
    return da.where(~mask)


def regrid_spherical_harmonics(da: xr.DataArray, ntrunc) -> xr.DataArray:
    """
    Truncate data on a regular grid in spherical harmonics space.

    See https://climatedataguide.ucar.edu/climate-tools/common-spectral-model-grid-resolutions
    for a summary of truncation levels.

    Adapted from https://github.com/frodre/LMROnline/blob/master/LMR_utils.py#L1113

    Args:
        da: array to truncate (shape lat x lon x sample)
        ntrunc: triangular truncation level

    Returns:
        Truncated array
    """
    if np.isnan(da).any():
        raise ValueError("Spherical harmonics regridding does not support nan")

    specob_old = spharm.Spharmt(len(da.lon), len(da.lat), gridtype="regular")

    nlat_new = (ntrunc + 1) + (ntrunc + 1) % 2
    nlon_new = int(nlat_new * 1.5)
    specob_new = spharm.Spharmt(nlon_new, nlat_new, gridtype="regular")

    # TODO maybe regrid in chunks
    x = spharm.regrid(specob_old, specob_new, da, ntrunc=ntrunc)

    include_poles = nlat_new % 2 != 0
    new_coords = da.coords.copy()
    new_coords["lat"], new_coords["lon"] = generate_regular_grid(
        nlat_new, nlon_new, include_endpts=include_poles
    )
    return xr.DataArray(x, dims=da.dims, coords=new_coords, name=None)


def generate_regular_grid(nlats, nlons, lat_bnd=(-90, 90), lon_bnd=(0, 360), include_endpts=False):
    """
    Generate regularly spaced latitude and longitude arrays where each point
    is the center of the respective grid cell.

    Adapted from https://github.com/frodre/LMROnline/blob/master/LMR_utils.py#L1242

    Parameters
    ----------
    nlats: int
        Number of latitude points
    nlons: int
        Number of longitude points
    lat_bnd: tuple(float), optional
        Bounding latitudes for gridcell edges (not centers).  Accepts values
        in range of [-90, 90].
    lon_bnd: tuple(float), optional
        Bounding longitudes for gridcell edges (not centers).  Accepts values
        in range of [-180, 360].
    include_endpts: bool
        Include the poles in the latitude array.

    Returns
    -------
    lat_center_2d:
        Array of central latitide points (nlat x nlon)
    lon_center_2d:
        Array of central longitude points (nlat x nlon)
    """
    if len(lat_bnd) != 2 or len(lon_bnd) != 2:
        raise ValueError("Bound tuples must be of length 2")
    if np.any(np.diff(lat_bnd) < 0) or np.any(np.diff(lon_bnd) < 0):
        raise ValueError("Lower bounds must be less than upper bounds.")
    if np.any(abs(np.array(lat_bnd)) > 90):
        raise ValueError("Latitude bounds must be between -90 and 90")
    if np.any(abs(np.diff(lon_bnd)) > 360):
        raise ValueError("Longitude bound difference must not exceed 360")
    if np.any(np.array(lon_bnd) < -180) or np.any(np.array(lon_bnd) > 360):
        raise ValueError("Longitude bounds must be between -180 and 360")

    lon_center = np.linspace(lon_bnd[0], lon_bnd[1], nlons, endpoint=False)

    if include_endpts:
        lat_center = np.linspace(lat_bnd[0], lat_bnd[1], nlats)
    else:
        tmp = np.linspace(lat_bnd[0], lat_bnd[1], nlats + 1)
        lat_center = (tmp[:-1] + tmp[1:]) / 2.0

    return lat_center, lon_center


def get_pressure_from_hybrid_coordinate(ds: xr.Dataset, dim="lev"):
    plev = None
    if "formula" in ds[dim].attrs:
        match ds[dim].attrs["formula"]:
            case "p = a*p0 + b*ps":
                # MRI-ESM
                plev = ds["a"] * ds["p0"] + ds["b"] * ds["ps"]
            case "p = ap + b*ps":
                # MPI-ESM
                plev = ds["ap"] + ds["b"] * ds["ps"]
    elif "long_name" in ds[dim].attrs:  # noqa: SIM102
        if ds[dim].attrs["long_name"] == "hybrid level at midpoints (1000*(A+B))":
            plev = 1000 * (ds["hyam"] + ds["hybm"])
        elif ds[dim].attrs["long_name"] == "hybrid level at interfaces (1000*(A+B))":
            plev = 1000 * (ds["hyai"] + ds["hybi"])

    if plev is None:
        raise ValueError("Unrecognized formula for sigma to pressure coordinate")

    return plev.rename("lev")


def regrid_hybrid_to_pressure(
    ds: xr.Dataset, field: str, lev_target: list[float] = CMIP6_PLEV19
) -> xr.DataArray:
    if "lev" not in ds[field].dims:
        return ds[field]

    assert ds["lev"].attrs["standard_name"] == "atmosphere_hybrid_sigma_pressure_coordinate"
    plev = get_pressure_from_hybrid_coordinate(ds)

    grid = Grid(ds, coords={"lev": {"center": "lev"}}, periodic=False, autoparse_metadata=False)
    da_pressure = grid.transform(
        ds[field], "lev", np.array(lev_target), target_data=plev, method="log"
    )

    if "units" in ds[field].attrs:
        da_pressure.attrs["units"] = ds[field].attrs["units"]
    return da_pressure


def get_z_from_hybrid_coordinate(ds: xr.Dataset, bounds=False, eta=None):
    """
    Get the vertical coordinates from a hybrid coordinate formula.

    eta (sea surface height) is the only time-varying term and can massively inflate the dataset
    even though it is a small term. For some applications (like integrating the ocean heat content
    over hundreds of meters), using a precomputed average eta is much more efficient.

    Args:
        ds: Dataset containing formula terms
        bounds: Whether to convert level bounds or levels themselves. Defaults to False.
        eta: Precomputed eta.

    Returns:
        Vertical coordinate
    """
    z = None

    lev = "lev"
    zlev = "zlev"
    sigma = "sigma"
    if bounds:
        lev += "_bnds"
        zlev += "_bnds"
        sigma += "_bnds"

    if "formula" in ds[lev].attrs:
        match ds[lev].attrs["formula"]:
            # MIROC?????
            case "for k <= nsigma: z(n,k,j,i) = eta(n,j,i) + sigma(k)*(min(depth_c,depth(j,i))+eta(n,j,i)) ; for k > nsigma: z(n,k,j,i) = zlev(k)":
                # https://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#_ocean_sigma_over_z_coordinate
                # Fixed levels above depth_c (for MIROC: 50 m), below terrain-following sigma
                # depth_c has index nsigma within depth

                # Assume nsigma and depth_c are identical across files
                nsigma = ds["nsigma"].item()
                depth_c = ds["depth_c"].item()
                if eta is None:
                    eta = ds["eta"]

                z_above = ds[zlev].isel(lev=slice(0, nsigma))
                z_below = eta + ds[sigma].isel(lev=slice(nsigma, None)) * (
                    np.minimum(depth_c, ds["depth"]) + eta
                )
                z = -xr.concat([z_above, z_below], dim="lev")

    if z is None:
        raise ValueError("Unrecognized formula for sigma to vertical coordinate")

    return z.rename(lev)


def mask_poles(da: xr.DataArray) -> xr.DataArray:
    return da.where(abs(da.lat) <= 88)
