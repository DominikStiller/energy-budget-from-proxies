from __future__ import annotations

import json
import sys

import cf_xarray as cfxr
import dask
import numpy as np
import xarray as xr
from dask.distributed import Client

from lmrecon.indices import AMOIndex, IPOTripoleIndex, Nino34Index, PDOIndex
from lmrecon.io import open_mfdataset, save_mfdataset
from lmrecon.logger import get_logger, logging_disabled
from lmrecon.mapper import PhysicalSpaceForecastSpaceMapper
from lmrecon.stats import anomalize, area_weighted_mean
from lmrecon.time import add_season_coords, use_enum_season_coords
from lmrecon.units import calculate_siarea, calculate_siextent
from lmrecon.util import get_base_path, get_data_path, get_host, to_math_order, unstack_state

logger = get_logger(__name__)


def _compute_regional_means(state, rundir, type):
    logger.info(f"Computing regional means of {type}")
    regions = {
        "global": (slice(None, None), slice(None, None)),
        "NH": (slice(0, None), slice(None, None)),
        "SH": (slice(None, 0), slice(None, None)),
    }
    regional_means = xr.concat(
        [
            area_weighted_mean(state.sel(lat=lat_bnd, lon=lon_bnd))
            .expand_dims("region")
            .assign_coords(region=[name])
            for name, (lat_bnd, lon_bnd) in regions.items()
        ],
        dim="region",
    )

    save_mfdataset(
        regional_means.astype(np.float32).compute(),
        rundir / f"{type}_regional_means",
        years_per_file=100,
        add_timestamp=False,
    )


def _compute_zonal_means(state, rundir, type):
    logger.info(f"Computing zonal means of {type}")
    # Calling compute before saving is somehow much faster and eliminates shuffling
    # zonal_means is about 6 GB so fits easily in memory
    zonal_means = state.mean("lon").astype(np.float32).compute()

    save_mfdataset(
        zonal_means, rundir / f"{type}_zonal_means", years_per_file=100, add_timestamp=False
    )


def _compute_indices(state, rundir, type):
    logger.info(f"Computing indices of {type}")

    # pdo_index = PDOIndex.load(get_base_path() / "datasets/temperature/ERSST5/pdo_index.pkl")

    indices = [
        Nino34Index().compute_index(state["tos"]),
        AMOIndex().compute_index(state["tos"]),
        IPOTripoleIndex().compute_index(state["tos"]),
        # xr.concat(
        #     [pdo_index.compute_index(state["tos"].isel(ens=i)) for i in state.ens], dim="ens"
        # ),
    ]

    if "siconcn" in state:
        indices.append(calculate_siextent(state["siconcn"]).rename("siextentn"))
        indices.append(calculate_siarea(state["siconcn"]).rename("siarean"))
    if "siconcs" in state:
        indices.append(calculate_siextent(state["siconcs"]).rename("siextents"))
        indices.append(calculate_siarea(state["siconcs"]).rename("siareas"))

    save_mfdataset(
        to_math_order(xr.merge(indices)).astype(np.float32).compute(),
        rundir / f"{type}_indices",
        years_per_file=100,
        add_timestamp=False,
    )


def _compute_ensemble_statistics(state, rundir, type):
    logger.info(f"Computing ensemble mean of {type}")
    mean = state.mean("ens").astype(np.float32).compute()
    save_mfdataset(mean, rundir / f"{type}_mean", years_per_file=100, add_timestamp=False)

    logger.info(f"Computing ensemble variance of {type}")
    var = state.var("ens", ddof=1).astype(np.float32).compute()
    save_mfdataset(var, rundir / f"{type}_var", years_per_file=100, add_timestamp=False)


def _map_to_physical(type, rundir):
    """
    Map reduced state to physical state. Writes the state to disk so that it can be efficiently reused.
    """
    # This works well with casper largemem nodes (720 GB)
    time_chunk_size_in = 4  # should be a factor of time_chunk_size_out
    time_chunk_size_out = 20 * 4  # 20 years
    ens_chunk_size_out = 50  # such that chunk is ~128 MB

    # Don't chunk by ens since files are only time chunks
    state_reduced = (
        cfxr.decode_compress_to_multi_index(
            open_mfdataset(
                rundir / f"{type}",
                parallel=True,
                combine="nested",
                concat_dim="time",
                data_vars="minimal",
                coords="minimal",
                compat="override",
            )
        )["data"]
        .chunk(dict(time=time_chunk_size_in))
        .persist()
    )

    with (rundir / "metadata.json").open("r") as f:
        metadata = json.load(f)
        mapper_path = get_data_path() / "mapper" / metadata["mapper_id"]

    mapper = PhysicalSpaceForecastSpaceMapper.load(mapper_path / "mapper.pkl")

    logger.info("Mapping reduced reconstruction to physical space")
    with logging_disabled():
        state_physical = unstack_state(
            mapper.backward(state_reduced.stack(dict(sample=["time", "ens"]))).unstack("sample")
        )
        state_physical.chunk(chunks=dict(time=time_chunk_size_out, ens=ens_chunk_size_out)).astype(
            np.float32
        ).to_zarr(rundir / f"{type}_physical.zarr")

    logger.info(f"Re-loading physical state of {type}")
    state = xr.open_zarr(rundir / f"{type}_physical.zarr")
    return state


def _add_sic_climatology(state):
    if "siconcn" not in state:
        return state

    logger.info("Postprocessing SIC")
    climatology_target = (
        use_enum_season_coords(
            xr.open_dataset(
                get_base_path() / "datasets/climo_vince/climatology_hybrid_siconc_1961-1990.nc"
            )
        )
        .sel(season=add_season_coords(state).season)
        .drop_vars("season")
    )
    climatology_target["siconcn"] = climatology_target["siconc"].where(climatology_target.lat > 0)
    climatology_target["siconcs"] = climatology_target["siconc"].where(climatology_target.lat < 0)

    for field in ["siconcn", "siconcs"]:
        state[field] = (state[field] + climatology_target[field]).clip(0, 1)

    return state


def _postprocess(rundir, type):
    logger.info(f"Loading {type} (run id {rundir.name})")

    if (rundir / f"{type}_physical.zarr").exists():
        state = xr.open_zarr(rundir / f"{type}_physical.zarr")
    else:
        state = _map_to_physical(type, rundir)

    logger.info("Calculating anomalies")
    # Shift everything to 1961-1990 anomalies
    state = anomalize(state, (1961, 1991), use_ensemble_mean=False)
    state = _add_sic_climatology(state)

    _compute_indices(state, rundir, type)
    _compute_regional_means(state, rundir, type)
    _compute_zonal_means(state, rundir, type)
    _compute_ensemble_statistics(state, rundir, type)


def postprocess_reconstruction(rundir):
    host = get_host()
    if host == "enkf":
        client = Client(n_workers=dask.system.CPU_COUNT // 2, threads_per_worker=1)  # noqa: F841
        # client = Client("tcp://127.0.0.1:37809")
    elif host == "mbp-dominik":
        # Very memory-limited
        client = Client(n_workers=2, threads_per_worker=2)  # noqa: F841
    else:
        client = Client(n_workers=dask.system.CPU_COUNT, threads_per_worker=1)  # noqa: F841
    # _postprocess(rundir, "prior")
    _postprocess(rundir, "posterior")

    logger.info("Postprocessing completed")


if __name__ == "__main__":
    postprocess_reconstruction(get_data_path() / "reconstructions" / sys.argv[1])
