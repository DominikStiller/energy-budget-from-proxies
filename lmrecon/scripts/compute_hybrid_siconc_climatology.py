from __future__ import annotations

import xarray as xr

from lmrecon.datasets import load_cdr_sic
from lmrecon.grid import Regridder
from lmrecon.logger import get_logger
from lmrecon.stats import anomalize, average_seasonally
from lmrecon.time import use_string_season_coords
from lmrecon.util import get_base_path

logger = get_logger(__name__)


if __name__ == "__main__":
    # mmm_path = get_data_path() / "cmip6" / "mmm" / "past1000_historical"
    # mmm = xr.open_zarr(mmm_path / "seasonal_averages.zarr")

    mmm_path = get_base_path() / "datasets/climo_vince"
    mmm = Regridder().regrid(
        average_seasonally(
            xr.open_mfdataset(
                str(mmm_path / "*_siconc_1961_1990.nc"), combine="nested", concat_dim="ens"
            ).mean("ens")
        )
        / 100
    )

    ds_sic = load_cdr_sic()

    hybrid = xr.concat(
        [
            mmm[["siconc"]].sel(time=slice(1961, ds_sic.time[0] - 0.5 / 12)),
            ds_sic[["siconc"]].sel(time=slice(None, 1991)),
        ],
        dim="time",
    )
    # hybrid.to_netcdf(mmm_path / "hybrid_siconc_1961-1990.nc")
    _, climatology_1961_1990 = anomalize(hybrid, (1961, 1991), return_climatology=True)
    use_string_season_coords(climatology_1961_1990).compute().to_netcdf(
        mmm_path / "climatology_hybrid_siconc_1961-1990.nc"
    )

    logger.info("Computing of hybrid climatology completed")
