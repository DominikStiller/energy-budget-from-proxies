from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import sklearn.linear_model

from lmrecon.signal import autocorrelation

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from lmrecon.mapper import PhysicalSpaceForecastSpaceMapper
    from lmrecon.time import Season


class PSM(ABC):
    """
    Base class for proxy system models (PSMs), or forward/observation model for proxies.
    """

    def __init__(
        self,
        pid: str,
        field: str,
        lat: float,
        lon: float,
        seasonality: None | Season | list[Season],
    ):
        """
        Initialize the PSM. The latitude and longitude used for calibration do not need to correspond
        necessarily to the proxy coordinates: the proxy is likely not on the grid, and even the nearest
        gridpoint may not have data for marine proxies (should use nearest ocean gridpoint, which may
        not be the absolute nearest gridpoint due to the land mask). Therefore, during calibration we
        select the nearest gridpoint that has data available. We do not need to check both the
        calibration data and the DA state for availability (non-nan-ness) since the calibration data
        are truncated to the DA state and thus inherits its nan mask.

        Args:
            pid: Proxy ID
            field: The field that the proxy is sensitive to
            lat: Latitude of gridpoint used for calibration
            lon: Longitude of gridpoint used for calibration
            seasonality: The seasons that the proxy is sensitive to. None means any season (for seasonal proxies).
        """
        self.pid = pid
        self.field = field
        self.lat = lat
        self.lon = lon
        self.seasonality = (
            seasonality if isinstance(seasonality, list) or seasonality is None else [seasonality]
        )
        self.err_std = None

    @abstractmethod
    def calibrate(self, x: ArrayLike, y: ArrayLike):
        """
        Calibrates the PSM.

        Args:
            x: physical state ([1 x n_sample])
            y: observation from calibration dataset ([1 x n_sample])

        Returns:
            None
        """
        if x.ndim != 2:
            raise ValueError("x and y must have one sample dimension sample dimension")

        if y.shape[0] != 1:
            raise ValueError("y must have a single state dimension")
        if x.shape[0] != 1:
            raise ValueError("x must have a single state dimension")

    @abstractmethod
    def forward(self, x: ArrayLike) -> ArrayLike:
        """
        Estimates the proxy value from physical state. Often, the physical state is simply the surface temperature
        at the proxy location.

        Args:
            x: physical state ([1] or [1 x n_sample])

        Returns:
            Proxy estimate ([1] or [1 x n_sample])
        """
        if x.shape[0] != 1:
            raise ValueError("x must have a single state dimension")


class IdentityPSM(PSM):
    def __init__(
        self,
        pid: str,
        field: str,
        lat: float,
        lon: float,
        err_std: float,
        seasonality: None | Season | list[Season],
    ):
        super().__init__(pid, field, lat, lon, seasonality)
        self.err_std = err_std

    def calibrate(self, x: ArrayLike, y: ArrayLike):
        super().calibrate(x, y)
        if x.shape[0] > 1:
            raise ValueError("IdentityPSM only supports one-dimensional x")

    def forward(self, x: ArrayLike) -> ArrayLike:
        super().forward(x)
        return x


class LinearPSM(PSM):
    def __init__(
        self,
        pid: str,
        field: str,
        lat: float,
        lon: float,
        seasonality: None | Season | list[Season],
    ):
        super().__init__(pid, field, lat, lon, seasonality)
        self._model = sklearn.linear_model.LinearRegression()

        # Diagnostics
        self.SNR = None
        self.R2 = None
        self.R2adj = None
        self.BIC = None
        self.AIC = None
        self.corr = None
        self.annual_error_acor = None

    def copy(self):
        return copy.deepcopy(self)

    def calibrate(self, x: ArrayLike, y: ArrayLike, year_steps=1):
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)

        super().calibrate(x, y)

        self._model.fit(x.T, y.T)

        yhat = self._model.predict(x.T).T
        residual = y - yhat
        ss_tot = ((y - y.mean()) ** 2).sum()
        ss_res = (residual**2).sum()

        nobs = y.shape[1]
        self.err_std = np.sqrt(ss_res / nobs)

        self.SNR = np.std(yhat) / np.std(residual)

        # Diagnostics copied from https://github.com/frodre/LMROnline/blob/master/LMR_psms.py#L480
        # Model fit
        self.R2 = 1 - (ss_res / ss_tot)
        # denom in last term is (n - p - 1) where n is nsamples and p is num
        # explanatory variables (which is 1 for univariate reg)
        self.R2adj = 1 - (1 - self.R2) * ((nobs - 1) / (nobs - 1 - 1))

        # BIC assumes errors are IID
        # k is the number of estimated parameters (intercept, slope, ss_res)
        k = 3
        self.BIC = nobs * np.log(ss_res / nobs) + k * np.log(nobs)
        self.AIC = 2 * k - 2 * np.log(ss_res)

        self.corr = np.sqrt(self.R2)
        if np.squeeze(self._model.coef_) < 0:
            self.corr = -self.corr

        self.annual_error_acor = autocorrelation(np.squeeze(residual), year_steps)

    def forward(self, x: ArrayLike) -> ArrayLike:
        x = np.atleast_2d(x)
        super().forward(x)
        return self._model.predict(x.T).T


class PhysicalSpacePSM(PSM):
    """
    Wraps another physical-space PSM and selects relevant state from the full state.
    """

    def __init__(self, psm: PSM, physical_state_idx: int):
        super().__init__(psm.pid, psm.field, psm.lat, psm.lon, psm.seasonality)
        if isinstance(psm, ReducedSpacePSM | PhysicalSpacePSM):
            raise ValueError("Cannot wrap ReducedSpacePSM or PhysicalSpacePSM in PhysicalSpacePSM")

        self.psm = psm
        self.physical_state_idx = physical_state_idx
        self.err_std = psm.err_std

    def calibrate(self, x: ArrayLike, y: ArrayLike):
        raise NotImplementedError()

    def forward(self, x: ArrayLike) -> ArrayLike:
        return self.psm.forward(x[[self.physical_state_idx], :])


class ReducedSpacePSM(PSM):
    """
    Wraps another physical-space PSM but takes reduced-space inputs, which it maps to the physical space.
    """

    def __init__(self, psm: PSM, physical_state_idx: int, mapper: PhysicalSpaceForecastSpaceMapper):
        super().__init__(psm.pid, "state", None, None, psm.seasonality)
        if isinstance(psm, ReducedSpacePSM | PhysicalSpacePSM):
            raise ValueError("Cannot wrap ReducedSpacePSM or PhysicalSpacePSM in ReducedSpacePSM")

        self.psm = psm
        self.physical_state_idx = physical_state_idx
        self.err_std = psm.err_std
        self.mapper = mapper

    def calibrate(self, x: ArrayLike, y: ArrayLike):
        raise NotImplementedError()

    def forward(self, x: ArrayLike) -> ArrayLike:
        return self.psm.forward(self.mapper.backward_matrix[[self.physical_state_idx], :] @ x)
