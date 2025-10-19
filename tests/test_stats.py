from __future__ import annotations

from unittest import TestCase

import numpy as np
import xarray as xr

from lmrecon.stats import (
    annualize_seasonal_data,
    average_annually,
    average_seasonally,
    detrend_polynomial,
    mse,
)
from lmrecon.time import Season, convert_decimal_year_to_datetime


class TestStats(TestCase):
    def test_annualize_seasonal_data_sameyear(self):
        # All seasonalities span the same year
        # Test array: 3 years of data with 1s for DJF, 2s for MAM...
        da = xr.DataArray(
            np.array([1, 2, 3, 4] * 3),
            dims=["time"],
            coords=dict(time=np.arange(0.5 / 12, 3, 1 / 4)),
        )

        # Single seasons
        np.testing.assert_almost_equal(annualize_seasonal_data(da, [Season.DJF]), np.array([1] * 3))
        np.testing.assert_almost_equal(annualize_seasonal_data(da, [Season.MAM]), np.array([2] * 3))
        np.testing.assert_almost_equal(annualize_seasonal_data(da, Season.MAM), np.array([2] * 3))
        np.testing.assert_almost_equal(annualize_seasonal_data(da, [Season.JJA]), np.array([3] * 3))
        np.testing.assert_almost_equal(annualize_seasonal_data(da, [Season.SON]), np.array([4] * 3))

        # Double seasons
        np.testing.assert_almost_equal(
            annualize_seasonal_data(da, [Season.DJF, Season.MAM]), np.array([1.5] * 3)
        )
        np.testing.assert_almost_equal(
            annualize_seasonal_data(da, [Season.MAM, Season.JJA]), np.array([2.5] * 3)
        )
        np.testing.assert_almost_equal(
            annualize_seasonal_data(da, [Season.JJA, Season.SON]), np.array([3.5] * 3)
        )
        np.testing.assert_almost_equal(
            annualize_seasonal_data(da, [Season.SON, Season.DJF]), np.array([2.5] * 2)
        )

        # Triple seasons
        np.testing.assert_almost_equal(
            annualize_seasonal_data(da, [Season.DJF, Season.MAM, Season.JJA]), np.array([2] * 3)
        )

        # Annual
        np.testing.assert_almost_equal(
            annualize_seasonal_data(da, [Season.DJF, Season.MAM, Season.JJA, Season.SON]),
            np.array([2.5] * 3),
        )

    def test_annualize_seasonal_data_twoyear(self):
        # All seasonalities span the previous and current year
        # Test array: 3 years of data with 1, 2, 3, 4, ...
        da = xr.DataArray(
            np.arange(1, 13), dims=["time"], coords=dict(time=np.arange(0.5 / 12, 3, 1 / 4))
        )

        # Compare to manually calculated averages
        np.testing.assert_almost_equal(
            annualize_seasonal_data(da, [Season.MAM, Season.JJA, Season.SON, Season.DJF]),
            np.array([3.5, 7.5]),
        )
        np.testing.assert_almost_equal(
            annualize_seasonal_data(da, [Season.JJA, Season.SON, Season.DJF, Season.MAM]),
            np.array([4.5, 8.5]),
        )
        np.testing.assert_almost_equal(
            annualize_seasonal_data(da, [Season.SON, Season.DJF]), np.array([4.5, 8.5])
        )

    def test_annualize_seasonal_data_invalid(self):
        # Test invalid seasonalities
        da = xr.DataArray(
            np.arange(1, 13), dims=["time"], coords=dict(time=np.arange(0.5 / 12, 3, 1 / 4))
        )

        with self.assertRaises(ValueError):
            # Out of order in same year
            annualize_seasonal_data(da, [Season.JJA, Season.SON, Season.MAM])

        with self.assertRaises(ValueError):
            # Out of order in previous year
            annualize_seasonal_data(da, [Season.SON, Season.JJA, Season.DJF, Season.MAM])

        with self.assertRaises(ValueError):
            # Non-consecutive across year
            annualize_seasonal_data(da, [Season.JJA, Season.DJF, Season.MAM])

        with self.assertRaises(ValueError):
            # Non-consecutive within year
            annualize_seasonal_data(da, [Season.DJF, Season.JJA, Season.SON])

        with self.assertRaises(ValueError):
            # Duplicate season
            annualize_seasonal_data(
                da, [Season.SON, Season.DJF, Season.MAM, Season.JJA, Season.SON]
            )

    def test_average_annually(self):
        # Averaging annually = averaging seasonally, then annually
        tt = np.arange(10, 15, 1 / 12)
        da = xr.DataArray(
            np.cos(tt) + 3 * np.cos(11 * tt) + np.random.randn(*tt.shape),
            dims=["time"],
            coords=dict(time=[convert_decimal_year_to_datetime(t) for t in tt]),
        )
        self.assertAlmostEqual(
            mse(
                average_annually(da),
                average_annually(average_seasonally(da), weight_months=False),
                mean_dims="time",
            )
            .compute()
            .item(),
            0,
            places=3,
        )


class TestDetrend(TestCase):
    def test_linear(self):
        x = np.arange(0, 10)
        coeffs = np.arange(1, 5)[:, np.newaxis]
        y = coeffs @ x[np.newaxis, :]
        da = xr.DataArray(y, coords=dict(state=np.arange(coeffs.shape[0]), time=x))

        da_detrended = detrend_polynomial(da, by_season=False)
        self.assertTrue(np.allclose(da_detrended, 0))

    def test_seasonal(self):
        da = np.zeros((10, 4))
        da[:, 3] = np.arange(10) + 1
        da = xr.DataArray(
            da.flatten(), dims="time", coords=dict(time=np.arange(0.5 / 12, 10, 1 / 4))
        )

        da_detrended = detrend_polynomial(da, by_season=True)
        self.assertTrue(np.allclose(da_detrended, 0))
