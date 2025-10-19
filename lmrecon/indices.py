from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np
import xarray as xr

from lmrecon.eof import EOF
from lmrecon.logger import get_logger, logging_disabled
from lmrecon.stats import area_weighted_mean
from lmrecon.util import NanMask, get_base_path, stack_state, to_math_order

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

logger = get_logger(__name__)


class VariabilityIndex:
    pass


class PDOIndex(VariabilityIndex):
    """
    Index for the Pacific Decadal Oscillation (PDO). The index is based on the leading EOF of SST
    anomalies over 110W-100E, 20N-70N. The global mean is removed from the SSTs anomalies as a form
    of detrending. See Trenberth & Fasullo (2013; 10.1002/2013EF000165).

    The original authors smooth the PDO index using a 61-month (~5 years) running mean filter.
    """

    def __init__(self) -> None:
        self.eof = EOF(1)
        self.pdo_index_std = None
        self.nan_mask = None

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        outfile = directory / "pdo_index.pkl"
        logger.info(f"Saving PDO index to {outfile}")
        pickle.dump(self, outfile.open("wb"))

    @classmethod
    def load(cls, file: Path | str | None = None) -> Self:
        if file is None:
            file = get_base_path() / "datasets/temperature/ERSST5/pdo_index.pkl"
        return pickle.load(Path(file).open("rb"))

    def _prepare_for_eof(self, da: xr.DataArray) -> xr.DataArray:
        da = da - area_weighted_mean(da)
        da = da.sel(lat=slice(20, 70), lon=slice(110, 260))
        da = to_math_order(stack_state(da))

        if self.nan_mask is None:
            self.nan_mask = NanMask()
            self.nan_mask.fit(da.data)
        da_nonan = self.nan_mask.forward(da.data)

        lats = self.nan_mask.forward(da.lat.data)[:, np.newaxis]
        weights = np.cos(np.radians(lats))

        # Fill nans since nan mask can be different between datasets used for fitting and index computation
        return np.nan_to_num(da_nonan * weights)

    def fit(self, da_tos: xr.DataArray):
        """
        Fit the EOF for the PDO index.

        Args:
            da_tos: dataset with SST anomalies (seasonal cycle removed)
        """
        assert da_tos.ndim == 3
        da = self._prepare_for_eof(da_tos)
        with logging_disabled():
            self.eof.fit(da)
        # Force computation
        self.eof = pickle.loads(pickle.dumps(self.eof))
        if self.eof.U.mean() > 0:
            # Flip EOF so that positive index = negative SST anomalies over northern pacific
            # Very heuristic
            self.eof.U *= -1
        self.pdo_index_std = np.array(self._compute_unstandardized_index(da).std())

    def _compute_unstandardized_index(self, da: xr.DataArray) -> ArrayLike:
        return self.eof.project_forwards(da)[0, :]

    def compute_index(self, da_tos: xr.DataArray) -> xr.DataArray:
        """
        Compute the PDO index.

        Args:
            da_tos: dataset with SST anomalies (seasonal cycle removed)
        """
        # Can't easily accept more than one sampling dimension to keep matrix math simple
        assert da_tos.ndim == 3
        da_tos = to_math_order(da_tos)
        pdo_index = (
            self._compute_unstandardized_index(self._prepare_for_eof(da_tos)) / self.pdo_index_std
        )
        coords = da_tos.drop_vars(["lat", "lon"]).coords.copy()
        return xr.DataArray(pdo_index, dims=[da_tos.dims[-1]], coords=coords, name="pdo_index")

    def fit_and_compute_index(self, da_tos: xr.DataArray) -> xr.DataArray:
        self.fit(da_tos)
        return self.compute_index(da_tos)


class IPOTripoleIndex(VariabilityIndex):
    """
    Index for the Interdecadal Pacific Oscillation (IPO). The index is based on the tripole SST
    anomaly pattern between the equatorial and northern/southern Pacific.
    See Henley at al. (2015; 10.1007/s00382-015-2525-1).

    The original authors smooth the IPO tripole index using a 13-year Chebyshev lowpass filter.

    Note: The original definition of the tripole index uses 1971-2000 as base period for the SST
    anomalies. Here, we use whatever the input is.
    """

    def compute_index(self, da_tos: xr.DataArray) -> xr.DataArray:
        """
        Compute the IPO tripole index.

        Args:
            da_tos: dataset with SST anomalies (seasonal cycle removed)
        """
        T1 = area_weighted_mean(da_tos.sel(lat=slice(25, 45), lon=slice(140, 360 - 145)))
        T2 = area_weighted_mean(da_tos.sel(lat=slice(-10, 10), lon=slice(170, 360 - 90)))
        T3 = area_weighted_mean(da_tos.sel(lat=slice(-50, -15), lon=slice(150, 360 - 160)))
        ipo_index = T2 - (T1 + T3) / 2
        return ipo_index.rename("ipo_index")


class AMOIndex(VariabilityIndex):
    """
    Index for the Atlantic Multi-Decadal Oscillation (AMO). The index is based on the SST anomaly
    over the North Atlantic (0-60N). The mean over 60S-60N is removed from the SSTs anomalies as a
    form of detrending.
    See Trenberth and Shea (2006; 10.1029/2006GL026894).

    The alternative definition by Enfield et al. (2001; 10.1029/2000GL012745) uses linearly detrended
    SSTs over 0-70N.

    The AMO index is commonly smoothed using a 10-year lowpass or running mean filter.
    """

    def compute_index(self, da_tos: xr.DataArray) -> xr.DataArray:
        """
        Compute the AMO index.

        Args:
            da_tos: dataset with SST anomalies (seasonal cycle removed)
        """
        da_tos = da_tos - area_weighted_mean(da_tos.sel(lat=slice(-60, 60)))
        amo_index = area_weighted_mean(da_tos.sel(lat=slice(0, 60), lon=slice(360 - 80, 360)))
        return amo_index.rename("amo_index")


class Nino34Index(VariabilityIndex):
    """
    Index for the El Nino-Southern Oscillation (ENSO). The index is based on the SST pattern in the
    Nino 3.4 region.

    The Nino 3.4 index is commonly smoothed using a 5-month running mean filter.
    """

    def compute_index(self, da_tos: xr.DataArray) -> xr.DataArray:
        """
        Compute the Nino 3.4 index.

        Args:
            da_tos: dataset with SST anomalies (seasonal cycle removed)
        """
        nino34_index = area_weighted_mean(
            da_tos.sel(lat=slice(-5, 5), lon=slice(360 - 170, 360 - 120))
        )
        return nino34_index.rename("nino34_index")
