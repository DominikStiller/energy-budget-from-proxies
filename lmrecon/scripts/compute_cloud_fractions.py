from __future__ import annotations

import dask
import xarray as xr
from dask.distributed import Client

from lmrecon.clouds import (
    get_high_cloud_fraction,
    get_low_cloud_fraction,
    get_medium_cloud_fraction,
)
from lmrecon.grid import CMIP6_PLEV19, regrid_hybrid_to_pressure


def load_cmip_data(root, time_chunks=-1):
    print(root)
    ds = xr.merge(
        [
            xr.open_mfdataset(
                f"{root}/Amon/{field}/*/*/*.nc",
                combine="nested",
                concat_dim="time",
                data_vars="minimal",
                coords="minimal",
                compat="override",
                chunks=dict(time=time_chunks, lev=-1),
                decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
            )
            for field in ["cl"]
        ],
        compat="override",
    )
    ds["cl"] = ds["cl"] / 100
    return ds


if __name__ == "__main__":
    client = Client(n_workers=dask.system.CPU_COUNT // 2, threads_per_worker=1)  # noqa: F841

    ## CMIP version
    # path = "PMIP/MRI/MRI-ESM2-0/past1000/r1i1p1f1"
    # time_chunks = -1

    # path = "PMIP/MPI-M/MPI-ESM1-2-LR/past2k/r1i1p1f1"
    # time_chunks = 120

    path = "PMIP/MIROC/MIROC-ES2L/past1000/r1i1p1f2"
    time_chunks = -1

    ds = load_cmip_data(f"/glade/campaign/univ/uwas0141/CMIP6/{path}", time_chunks=time_chunks)
    ds = regrid_hybrid_to_pressure(ds, "cl", lev_target=CMIP6_PLEV19)

    ## CESM version
    # path = "PMIP/NCAR/CESM-WACCM-FV2/past1000/r1i1p1f1"
    # inpath = "/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/b.e21.BWmaHIST.f19_g17.PMIP4-past1000.001"

    # path = "PMIP/NCAR/CESM-WACCM-FV2/past1000/r1i1p1f2"
    # inpath = "/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002"

    # from xgcm import Grid
    # from lmrecon.cesm_tools import load_cesm_timeseries
    # ds = load_cesm_timeseries(inpath, variables=["CLOUD"])
    # grid = Grid(ds, coords={"lev": {"center": "lev"}}, periodic=False, autoparse_metadata=False)
    # ds = grid.transform(ds["CLOUD"], "lev", CMIP6_PLEV19, target_data=ds["lev"], method="log")

    ## CMIP + CESM
    ds_clouds = xr.merge(
        [
            get_low_cloud_fraction(ds).rename("cll"),
            get_medium_cloud_fraction(ds).rename("clm"),
            get_high_cloud_fraction(ds).rename("clh"),
        ]
    )
    ds_clouds.chunk(time=2400).to_zarr(
        f"/glade/campaign/univ/uwas0141/CMIP6-proc/{path}/Amon/clouds.zarr"
    )

    print("Computing of cloud fractions completed")
