from __future__ import annotations

import json
import shutil
import sys

import dask
import numpy as np
import xarray as xr
from distributed import Client

from lmrecon.io import open_mfdataset, save_mfdataset
from lmrecon.logger import get_logger
from lmrecon.scripts.postprocess_reconstruction import postprocess_reconstruction
from lmrecon.util import get_data_path, subsample_ensemble

logger = get_logger(__name__)


def _collate_verification_proxies(rundir, iterations):
    logger.info("Collating verification proxies")
    for type in [
        "seasonal_assimilated",
        "annual_assimilated",
        "seasonal_withheld",
        "annual_withheld",
    ]:
        verifications = xr.concat(
            [xr.open_dataset(rundir / i / f"verification_{type}.nc") for i in iterations],
            dim="iteration",
        ).assign_coords(iteration=range(len(iterations)))
        # Need to force length because some strings are clipped otherwise
        verifications.to_netcdf(
            rundir / f"verification_{type}.nc", encoding=dict(pid=dict(dtype="<U50"))
        )


def _collate_ensemble_members(rundir, iterations, n_ens_per_iteration, type):
    logger.info("Collating ensemble members")

    posteriors = []
    ens_coords = np.arange(n_ens_per_iteration * len(iterations))

    for i in iterations:
        full_ensemble = open_mfdataset(
            rundir / i / type,
            parallel=True,
            combine="nested",
            concat_dim="time",
            data_vars="minimal",
            coords="minimal",
            compat="override",
        )
        posteriors.append(
            subsample_ensemble(full_ensemble, n_ens_per_iteration, offset=int(i)).assign_coords(
                ens=ens_coords[(int(i) - 1) * n_ens_per_iteration : int(i) * n_ens_per_iteration]
            )
        )

    posteriors = xr.concat(posteriors, dim="ens")

    # Align chunks with time_chunk_size_in in postprocessing (1*4)
    # Compute to speed up saving, full dataset is ~2 GB
    save_mfdataset(
        posteriors.chunk(dict(time=4)).compute(),
        rundir / type,
        years_per_file=1,
        add_timestamp=False,
    )


def postprocess_mc_reconstruction(rundir, n_ens_per_iteration):
    client = Client(n_workers=dask.system.CPU_COUNT, threads_per_worker=1)  # noqa: F841

    iterations = [d.name for d in rundir.glob("*") if d.name.isdigit()]

    with (rundir / iterations[0] / "metadata.json").open("r") as f:
        metadata = json.load(f)
    del metadata["seed_offset"]
    metadata["n_ens_per_iteration"] = n_ens_per_iteration
    json.dump(metadata, (rundir / "metadata.json").open("w"), indent=4)

    shutil.copy(get_data_path() / metadata["obs_dataset"] / "pdb.pkl", rundir / "pdb_full.pkl")

    _collate_verification_proxies(rundir, iterations)
    _collate_ensemble_members(rundir, iterations, n_ens_per_iteration, "posterior")
    client.shutdown()

    postprocess_reconstruction(rundir)


if __name__ == "__main__":
    postprocess_mc_reconstruction(
        get_data_path() / "reconstructions" / sys.argv[1], int(sys.argv[2])
    )
