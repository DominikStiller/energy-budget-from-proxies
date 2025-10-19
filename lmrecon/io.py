from __future__ import annotations

import dataclasses
import glob
import warnings
from pathlib import Path

import intake
import numpy as np
import xarray as xr
from xarray import SerializationWarning

from lmrecon.cesm_tools import fix_cesm_timestamp
from lmrecon.clouds import (
    get_high_cloud_fraction,
    get_low_cloud_fraction,
)
from lmrecon.grid import (
    GLOBAL_GRID,
    Regridder,
    get_z_from_hybrid_coordinate,
    regrid_hybrid_to_pressure,
)
from lmrecon.logger import get_logger
from lmrecon.units import (
    convert_thetao_to_ohc,
    convert_to_si_units,
)
from lmrecon.util import (
    filter_cf_valid,
    get_base_path,
    get_data_path,
    get_timestamp,
    has_cftime_timedim,
    standardize_coordinate_names,
)

logger = get_logger(__name__)

warnings.filterwarnings("ignore", category=SerializationWarning)


@dataclasses.dataclass
class CMIP6Variable:
    name: str  # will be renamed to this, possible after postprocessing
    id: str  # id in CMIP dataset
    table: str

    @staticmethod
    def from_list(vars_str: list[str]) -> list[CMIP6Variable]:
        vars = []
        for var in vars_str:
            if var in CMIP_VARIABLES:
                vars.append(CMIP_VARIABLES.get(var))
            else:
                raise ValueError(f"Unknown variable name {var}")
        return vars


CMIP_VARIABLES = {
    "clwvi": CMIP6Variable("clwvi", "clwvi", "Amon"),
    "rsdt": CMIP6Variable("rsdt", "rsdt", "Amon"),
    "rsut": CMIP6Variable("rsut", "rsut", "Amon"),
    "rlut": CMIP6Variable("rlut", "rlut", "Amon"),
    "cldlow": CMIP6Variable("cldlow", "cl", "Amon"),
    "cldhigh": CMIP6Variable("cldhigh", "cl", "Amon"),
    "tas": CMIP6Variable("tas", "tas", "Amon"),
    "tos": CMIP6Variable("tos", "tos", "Omon"),
    "siconc": CMIP6Variable("siconc", "siconc", "SImon"),
    "ohc300": CMIP6Variable("ohc300", "thetao", "Omon"),
    "ohc700": CMIP6Variable("ohc700", "thetao", "Omon"),
}


@dataclasses.dataclass
class CESMVariable:
    name: str  # will be renamed to this, possible after postprocessing
    id: str  # CESM variable
    component: str

    @staticmethod
    def from_list(vars_str: list[str]) -> list[CESMVariable]:
        vars = []
        for var in vars_str:
            if var in CESM_VARIABLES:
                vars.append(CESM_VARIABLES.get(var))
            else:
                raise ValueError(f"Unknown variable name {var}")
        return vars


# See https://github.com/NCAR/PyConform/blob/master/examples/CESM/CMIP6/CESM_MastList.def
CESM_VARIABLES = {
    # "clwvi": CESMVariable("clwvi", "TGCLDLWP", "atm"),
    "rsdt": CESMVariable("rsdt", "SOLIN", "atm"),
    "rsut": CESMVariable("rsut", "FSUTOA", "atm"),
    "rlut": CESMVariable("rlut", "FLUT", "atm"),
    # We might have different low/high thresholds, so cannot use CLDLOW/CLDHGH
    "cldlow": CESMVariable("cldlow", "CLOUD", "atm"),
    "cldhigh": CESMVariable("cldhigh", "CLOUD", "atm"),
    "tas": CESMVariable("tas", "TREFHT", "atm"),
    "tos": CESMVariable("tos", "SST", "ocn"),
    "siconc": CESMVariable("siconc", "aice", "ice"),
    "ohc300": CESMVariable("ohc300", "TEMP", "ocn"),
    "ohc700": CESMVariable("ohc700", "TEMP", "ocn"),
}


VARS_AND_DIMS_TO_DROP = [
    "dcpp_init_year",
    "member_id",
    "time_bnds",
    "lat_bnds",
    "lon_bnds",
    "nsigma",
    "depth_c",
    "height",
    "depth",
    "type",
    "p0",
    "vertices",
    "vertices_latitude",
    "vertices_longitude",
]


def get_catalog_location(type="lmrecon") -> str:
    match type:
        case "lmrecon":
            return str(get_base_path() / "CMIP6" / "catalog.json")
        case "glade":
            return str(get_base_path() / "CMIP6" / "catalog_glade.json")
        case "glade_official":
            return "/glade/collections/cmip/catalog/intake-esm-datastore/catalogs/glade-cmip6.json"
        case _:
            raise ValueError(f"Unknown type {type}")


class IntakeESMLoader:
    def __init__(
        self,
        model_id: str,
        experiment_id: str,
        variables: list[str] | None = None,
        catalog_location: str | None = None,
    ):
        self.model_id = model_id
        self.experiment_id = experiment_id
        self.regridder = Regridder(GLOBAL_GRID)
        self.catalog_location = catalog_location or get_catalog_location()
        self.variables = variables

        logger.debug(f"Opening catalog {self.catalog_location}")
        self.cat = intake.open_esm_datastore(self.catalog_location)
        self._get_variables()

    def _get_variables(self):
        """Get intersection of available and supported variables"""
        if not self.variables:
            all_datasets = (
                self.cat.search(experiment_id=self.experiment_id, source_id=self.model_id)
                .df[["variable_id", "table_id"]]
                .drop_duplicates()
            )

            variables = []
            for v in CMIP_VARIABLES.values():
                if (
                    (all_datasets["variable_id"] == v.id) & (all_datasets["table_id"] == v.table)
                ).any():
                    variables.append(v.name)

            if len(variables) == 0:
                raise ValueError("No variables found")

            self.variables = variables

        self.variables = CMIP6Variable.from_list(sorted(self.variables))

    def load_dataset(
        self,
        timerange: list[str] | None = None,
        member_id: str | None = None,
        version: str | None = None,
        grid_label="gn",
    ):
        if not member_id:
            all_members = list(
                self.cat.search(experiment_id=self.experiment_id, source_id=self.model_id)
                .df["member_id"]
                .unique()
            )
            if len(all_members) != 1:
                raise ValueError("If no member is specified, there must be exactly one available")
            member_id = all_members[0]

        logger.info(f"Loading dataset for {self.model_id} {self.experiment_id}, member {member_id}")
        das = {}
        for variable in self.variables:
            logger.debug(f" - {variable}")

            query = dict(
                experiment_id=self.experiment_id,
                source_id=self.model_id,
                member_id=member_id,
                table_id=variable.table,
                variable_id=variable.id,
                grid_label=grid_label,
            )
            if timerange:
                query["time_range"] = timerange
            if version:
                query["version"] = version

            query_results = self.cat.search(**query)

            # Ensure there is no ambiguity about dataset (i.e. exactly one is found)
            if len(query_results) == 0:
                raise LookupError("No datasets found for query")
            if not all(query_results.nunique().drop(["time_range", "path"]) <= 1):
                raise LookupError("Multiple datasets found for query")

            # Chunk size should be a multiple of 12 (minimum number of timesteps per file)
            ds = query_results.to_dask(
                progressbar=False, xarray_open_kwargs=dict(chunks=dict(time=600), use_cftime=True)
            )

            for var in ds.variables:
                if "units" in ds[var].attrs:
                    ds[var] = convert_to_si_units(ds[var])

            ds = standardize_coordinate_names(ds)
            ds = ds.assign_coords(lat=filter_cf_valid(ds["lat"]), lon=filter_cf_valid(ds["lon"]))

            if "nbnd" in ds.dims:
                # CESM/CAM uses different name for bounds dimension
                ds = ds.rename(nbnd="bnds")
            if "d2" in ds.dims:
                # CESM ocean uses different name for bounds dimension
                ds = ds.rename(d2="bnds")

            if self.model_id == "MIROC-ES2L" and ds.realm in ["ocean", "seaIce"]:
                # MIROC ocean + sea ice is on tripolar grid -> need to extrapolate instead of using
                # periodic condition
                periodic = False
                extrapolate = True
            elif self.model_id == "MRI-ESM2-0" and ds.realm in ["ocean", "seaIce"]:
                # MRI ocean + sea ice is on tripolar grid
                periodic = False
                extrapolate = False
            else:
                periodic = True
                extrapolate = False

            # Select variable and regrid vertically if necessary
            if "lev" in ds.dims:
                if (
                    ds["lev"].attrs["standard_name"]
                    == "atmosphere_hybrid_sigma_pressure_coordinate"
                ):
                    da = regrid_hybrid_to_pressure(ds.chunk(dict(lev=-1)), variable.id)
                elif ds["lev"].attrs["standard_name"] == "alevel":
                    # pressure level [hPa]
                    da = ds[variable.id].rename(lev="plev")
                    # Some of CESM's CMIP data have level coordinates in negative hPa, also in the
                    # wrong order
                    da = da.assign_coords(plev=np.abs(da["plev"])).sortby("plev", ascending=False)
                elif ds["lev"].attrs["standard_name"] in ["depth", "olevel"]:
                    # ocean depth [m]
                    da = ds[variable.id]
                elif ds["lev"].attrs["standard_name"] == "ocean_sigma_z":
                    # MIROC's ocean coordinate
                    eta_file = get_data_path() / "cmip6" / self.model_id / "eta_ocean_average.nc"
                    if not eta_file.exists():
                        eta_file.parent.mkdir(parents=True, exist_ok=True)
                        # Use 100-year average eta
                        # Saving this also ensures that the same eta is used across runs for the
                        # same model
                        ds["eta"].isel(time=slice(100 * 12)).mean("time").reset_coords(
                            drop=True
                        ).to_netcdf(eta_file)
                    eta = xr.open_dataset(eta_file)["eta"]

                    ds["lev"] = self.regridder.regrid(
                        get_z_from_hybrid_coordinate(ds, eta=eta).chunk(dict(lev=-1)),
                        ds.realm,
                        periodic=periodic,
                        extrapolate=extrapolate,
                    ).persist()
                    ds["lev_bnds"] = self.regridder.regrid(
                        get_z_from_hybrid_coordinate(ds, bounds=True, eta=eta).chunk(dict(lev=-1)),
                        ds.realm,
                    ).persist()
                    da = ds[variable.id]
                else:
                    raise ValueError("Unknown level coordinate: " + str(ds["lev"].attrs))
            else:
                da = ds[variable.id]

            # Regrid horizontally
            da = self.regridder.regrid(da, ds.realm, periodic=periodic, extrapolate=extrapolate)

            # Some fields can have odd chunks (I believe when they're auto-chunked and start out with lots of
            # data, e.g., 3D), which leads to problems downstream since they are different from other fields
            # Therefore, remove chunks in lat and lon dims
            da = da.chunk(dict(lat=-1, lon=-1))

            da = self._postprocess_field(variable, da, ds)

            das[variable.name] = da.rename(variable.name)

        das = dict(sorted(das.items()))

        # Merge all fields
        ds_merged = xr.merge(das.values()).drop_attrs()
        ds_merged = ds_merged.drop_vars(
            VARS_AND_DIMS_TO_DROP,
            errors="ignore",
        ).squeeze()
        return ds_merged

    def _postprocess_field(
        self, variable: CMIP6Variable, da: xr.DataArray, ds: xr.Dataset
    ) -> xr.DataArray:
        match variable.name:
            case "cldlow":
                return get_low_cloud_fraction(da)
            case "cldhigh":
                return get_high_cloud_fraction(da)
            case "ohc300":
                return convert_thetao_to_ohc(da, ds["lev"], ds["lev_bnds"], 300)
            case "ohc700":
                return convert_thetao_to_ohc(da, ds["lev"], ds["lev_bnds"], 700)
            case _:
                return da


class CESMTimeseriesLoader:
    def __init__(self, output_path: str, variables: list[str] | None = None):
        self.output_path = Path(output_path)
        self.regridder = Regridder(GLOBAL_GRID)
        self.variables = CESMVariable.from_list(variables)

    def _list_files(self, pattern, history_slice):
        files = sorted(glob.glob(pattern))
        if history_slice is None:
            return files
        else:
            return files[history_slice]

    def load_dataset(self, history_slice=None):
        das = {}
        for variable in self.variables:
            logger.debug(f" - {variable}")
            component_path = self.output_path / variable.component / "proc/tseries"

            try:
                ds = xr.open_mfdataset(
                    self._list_files(
                        f"{component_path}/month_1/*.{variable.id}.*.nc", history_slice
                    ),
                    use_cftime=True,
                )
                ds = fix_cesm_timestamp(ds)
            except OSError:
                # SST is sometimes only available daily
                logger.info(f"Using daily data for {variable.id}")
                ds = xr.open_mfdataset(
                    self._list_files(f"{component_path}/day_1/*.{variable.id}.*.nc", history_slice),
                    use_cftime=True,
                )
                ds = ds.resample(time="1MS").mean(dim="time")
                # Assign center of month
                time = (
                    ds.time.isel(time=slice(-1))
                    + (
                        ds.time.isel(time=slice(1, None)).values
                        - ds.time.isel(time=slice(0, -1)).values
                    )
                    / 2
                )
                ds = ds.isel(time=slice(-1)).assign_coords(time=time)

            if variable.component == "ocn":
                # Scalars are located on T-grid (Section 3.1 of https://www2.cesm.ucar.edu/models/cesm1.0/pop2/doc/sci/POPRefManual.pdf)
                # Will need to differentiate if I load horizontal vectors
                ds = ds.drop_vars(["ULONG", "ULAT"]).rename(
                    dict(TLONG="lon", TLAT="lat", nlat="lat", nlon="lon")
                )
            elif variable.component == "ice":
                ds = ds.drop_vars(["ULON", "ULAT"]).rename(
                    dict(TLON="lon", TLAT="lat", ni="lon", nj="lat")
                )

            for var in ds.variables:
                if "units" in ds[var].attrs:
                    ds[var] = convert_to_si_units(ds[var])

            if "nbnd" in ds.dims:
                # CESM/CAM uses different name for bounds dimension
                ds = ds.rename(nbnd="bnds")
            if "d2" in ds.dims:
                # CESM ocean uses different name for bounds dimension
                ds = ds.rename(d2="bnds")
            if "z_t" in ds.dims:
                # CESM ocean
                ds = ds.rename(z_t="lev")

            # Select variable and regrid vertically if necessary
            if "lev" in ds[variable.id].dims:
                if (
                    "standard_name" in ds["lev"].attrs
                    and ds["lev"].attrs["standard_name"]
                    == "atmosphere_hybrid_sigma_pressure_coordinate"
                ):
                    da = regrid_hybrid_to_pressure(ds.chunk(dict(lev=-1)), variable.id)
                elif ds["lev"].attrs["long_name"] == "depth from surface to midpoint of layer":
                    # ocean depth [m]
                    da = ds[variable.id]
                else:
                    raise ValueError("Unknown level coordinate: " + str(ds["lev"].attrs))
            else:
                da = ds[variable.id]

            da = self.regridder.regrid(da, variable.component)
            da = da.chunk(dict(lat=-1, lon=-1))

            da = self._postprocess_field(variable, da, ds)

            das[variable.name] = da.rename(variable.name)

        das = dict(sorted(das.items()))

        # Merge all fields
        ds_merged = xr.merge(das.values(), join="exact").drop_attrs()
        ds_merged = ds_merged.drop_vars(
            VARS_AND_DIMS_TO_DROP,
            errors="ignore",
        ).squeeze()
        return ds_merged

    def _postprocess_field(
        self, variable: CESMVariable, da: xr.DataArray, ds: xr.Dataset
    ) -> xr.DataArray:
        match variable.name:
            case "cldlow":
                return get_low_cloud_fraction(da)
            case "cldhigh":
                return get_high_cloud_fraction(da)
            case "ohc300":
                return convert_thetao_to_ohc(da, ds["lev"], self._get_level_bounds(ds), 300)
            case "ohc700":
                return convert_thetao_to_ohc(da, ds["lev"], self._get_level_bounds(ds), 700)
            case _:
                return da

    def _get_level_bounds(self, ds: xr.Dataset) -> xr.DataArray:
        return xr.concat(
            [
                ds["z_w_top"].rename(z_w_top="lev").assign_coords(lev=ds["lev"].values),
                ds["z_w_bot"].rename(z_w_bot="lev").assign_coords(lev=ds["lev"].values),
            ],
            dim="bnds",
        ).transpose("lev", "bnds")


def open_mfdataset(files: Path | str | list[Path] | list[str], **kwargs) -> xr.Dataset:
    if not isinstance(files, list):
        files = sorted(Path(files).expanduser().glob("*.nc"))

    try:
        ds = xr.open_mfdataset(
            files,
            use_cftime=True,
            **kwargs,
        )
    except Exception:
        ds = xr.open_mfdataset(
            files,
            use_cftime=True,
            **kwargs,
            engine="h5netcdf",
        )

    if "concat_dim" in kwargs:
        ds = ds.sortby(kwargs["concat_dim"])
    return ds


def save_mfdataset(
    ds: xr.Dataset, directory: Path, years_per_file: int = 100, compute=True, add_timestamp=True
):
    if add_timestamp:
        directory /= get_timestamp()
    directory.mkdir(parents=True)

    if has_cftime_timedim(ds):
        year = ds["time"].dt.year
    else:
        year = ds["time"]
    group_labels = ((year - year[0]) / years_per_file).astype(int)

    indexes, datasets = zip(*ds.groupby(group_labels))
    paths = [directory / f"{i}.nc" for i in indexes]

    logger.info(f"Saving dataset to {directory}")
    if compute:
        xr.save_mfdataset(datasets, paths)
    else:
        return xr.save_mfdataset(datasets, paths, compute=False)


def save_dataset(ds: xr.Dataset, path: Path, compute=True):
    if path.is_dir():
        raise ValueError("path must be a file")

    path = path.with_stem(f"{path.stem}-{get_timestamp()}")

    logger.info(f"Saving dataset to {path}")
    if compute:
        ds.to_netcdf(path, engine="h5netcdf")
    else:
        return ds.to_netcdf(path, compute=False, engine="h5netcdf")
