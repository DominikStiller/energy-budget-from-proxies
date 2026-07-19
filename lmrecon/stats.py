from __future__ import annotations

import itertools
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pyleoclim
import scipy
import sklearn
import xarray as xr
from xarray import DataArray

from lmrecon.logger import get_logger
from lmrecon.time import (
    Season,
    add_season_coords,
    is_annual_resolution,
    map_season_to_decimal,
    round_to_nearest_season,
    split_seasonality,
    use_tuple_time_coords,
)
from lmrecon.util import (
    NanMask,
    area_weighted,
    get_position_dims,
    has_cftime_timedim,
    has_npdatetime_timedim,
    local_np_seed,
    stack_state,
    to_math_order,
    unstack_state,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from xarray.core.types import T_Xarray

logger = get_logger(__name__)


def _normalize_alternative(alternative: str) -> str:
    """Normalize alternative-hypothesis aliases to canonical labels."""
    alternative_map = {
        "!=": "two-sided",
        "two-sided": "two-sided",
        ">": "greater",
        "greater": "greater",
    }
    try:
        return alternative_map[alternative]
    except KeyError as exc:
        raise ValueError("alternative must be one of '!=', 'two-sided', '>', or 'greater'") from exc


def calculate_correlation(x, y, axis=1):
    """
    Calculates Pearson correlation coefficient between x and y along a specific axis.
    Optimized for high-dimensional arrays (e.g., time, features).

    Args:
        x (np.ndarray): First array
        y (np.ndarray): Second array
        axis (int): The axis representing samples/time (default 1).

    Returns:
        np.ndarray: Correlation coefficients reduced by the specified axis.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # 1. Center the data (Anomalies)
    x_mean = np.mean(x, axis=axis, keepdims=True)
    y_mean = np.mean(y, axis=axis, keepdims=True)

    x_centered = x - x_mean
    y_centered = y - y_mean

    # 2. Calculate Numerator (Covariance)
    numerator = np.sum(x_centered * y_centered, axis=axis)

    # 3. Calculate Denominator (Std Devs)
    # Note: Division by (N-1) cancels out
    denom = np.sqrt(np.sum(x_centered**2, axis=axis)) * np.sqrt(np.sum(y_centered**2, axis=axis))

    # 4. Calculate Correlation (handle division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = numerator / denom

    return corr


def estimate_effective_dof(
    da: ArrayLike, da2: ArrayLike | None = None, mean_over_rows=True
) -> float:
    """
    Estimate effective degrees of freedom of an autocorrelated time series, or two correlated time series.
    """
    if da.ndim == 1:
        da = da[np.newaxis, :]
    assert da.ndim == 2

    da = da - da.mean(axis=1, keepdims=True)
    # Estimate 1-lag autocorrelation
    autocorrelation = np.mean(da[:, 1:] * da[:, :-1], axis=1) / np.var(da, axis=1)

    if da2 is not None:
        if da2.ndim == 1:
            da2 = da2[np.newaxis, :]
        assert da.shape == da2.shape
        da2 = da2 - da2.mean(axis=1, keepdims=True)
        autocorrelation2 = np.mean(da2[:, 1:] * da2[:, :-1], axis=1) / np.var(da2, axis=1)
    else:
        autocorrelation2 = autocorrelation

    if mean_over_rows:
        autocorrelation = np.nanmean(autocorrelation)
        autocorrelation2 = np.nanmean(autocorrelation2)

    # Clamp small negative autocorrelations to zero. Slightly negative estimates arise from
    # sampling noise in short or weakly-autocorrelated series and are not physically meaningful.
    autocorrelation = np.maximum(autocorrelation, 0)
    autocorrelation2 = np.maximum(autocorrelation2, 0)

    n_samples = da.shape[1]

    # Estimate degrees of freedom
    # Bretherton et al., 1999. "The Effective Number of Spatial Degrees of Freedom of a Time-Varying Field", Eq. (31).
    dof = (
        n_samples
        * (1 - autocorrelation * autocorrelation2)
        / (1 + autocorrelation * autocorrelation2)
    )

    return dof


def area_weighted_mean(
    ds: T_Xarray, square_weights=False, zonal_only=False, longitudinal_only=False
) -> T_Xarray:
    """
    Calculate mean of dataset across spatial dimensions, weighted by cosine of latitude.

    Args:
        ds: Dataset
        square_weights: Whether weights should be squared, e.g., when calculating mean of variance
        zonal_only: Whether average should only be over longitudes
        longitudinal_only: Whether average should only be over latitudes

    Returns:
        Mean
    """
    if isinstance(ds, xr.Dataset) and len(ds.keys()) == 0:
        return ds

    ds_weighted = area_weighted(ds, square_weights=square_weights)

    if "location" in ds.dims:
        return ds_weighted.mean("location")
    elif "state" in ds.dims:
        return ds_weighted.mean("state")
    else:
        lat_dim, lon_dim = get_position_dims(ds)
        if zonal_only:
            return ds_weighted.mean(lon_dim)
        if longitudinal_only or not lon_dim:
            # Longitude only requested, or input data is already zonally averaged
            return ds_weighted.mean(lat_dim)
        else:
            # Data is in lat/lon coordinates (not yet zonally averaged)
            return ds_weighted.mean([lat_dim, lon_dim])


def mse(x: T_Xarray, y: T_Xarray, mean_dims="time") -> T_Xarray:
    return ((x - y) ** 2).mean(mean_dims)


def rmse(x: T_Xarray, y: T_Xarray, mean_dims="time") -> T_Xarray:
    return mse(x, y, mean_dims) ** (1 / 2)


def area_weighted_mse(x: T_Xarray, y: T_Xarray, mean_dims="time") -> T_Xarray:
    if mean_dims is None:
        return area_weighted_mean((x - y) ** 2)
    else:
        return area_weighted_mean(((x - y) ** 2).mean(mean_dims))


def area_weighted_rmse(x: T_Xarray, y: T_Xarray, mean_dims="time") -> T_Xarray:
    return area_weighted_mse(x, y, mean_dims) ** (1 / 2)


def ce(test: T_Xarray, verification: T_Xarray, dim="time"):
    test, verification = xr.align(test, verification, join="inner")
    return 1 - np.divide(
        ((verification - test) ** 2).sum(dim),
        ((verification - verification.mean(dim)) ** 2).sum(dim),
    )


def compute_spread_skill_ratio(test: T_Xarray, verification: T_Xarray) -> T_Xarray:
    """Compute the spread-skill ratio.

    The spread-skill ratio compares the ensemble spread to the forecast skill.
    A ratio close to 1 indicates good calibration, where the ensemble spread
    appropriately matches the forecast error. Ratios greater than 1 indicate
    overdispersion (ensemble too spread out/underconfidence), while ratios less than 1 indicate
    underdispersion (ensemble too narrow/overconfidence).

    Args:
        test: Ensemble forecast data with an 'ens' dimension and 'case' dimension.
        verification: Verification/observational data with a 'case' dimension.

    Returns:
        Spread-skill ratio as an xarray object. Values close to 1 indicate
        well-calibrated ensembles.
    """
    skill = rmse(test.mean("ens"), verification, mean_dims="case")
    spread = np.sqrt(test.var("ens", ddof=1).mean("case"))
    return spread / skill


def compute_field_stds(ds: T_Xarray, dim="time") -> T_Xarray:
    """Compute total field standard deviation"""
    weights = np.cos(np.radians(ds.lat))
    return np.sqrt(ds.var(dim, ddof=1).weighted(weights).sum(["lat", "lon"])).compute()


def compute_field_stds_by_season(ds: T_Xarray) -> T_Xarray:
    """Compute total field standard deviation, grouped by season"""
    weights = np.cos(np.radians(ds.lat))
    return np.sqrt(
        use_tuple_time_coords(ds)
        .groupby("season")
        .var("time", ddof=1)
        .weighted(weights)
        .sum(["lat", "lon"])
    ).compute()


def average_annually(
    ds: T_Xarray, first_month: int | None = None, weight_months=True, remove_incomplete=True
) -> T_Xarray:
    """
    Average annually and remove partial years.

    For seasonal data with float timestamps, generated via average_seasonally(),
    the year is defined as
    DJF] [MAM JJA SON
    Y-1        Y
    i.e. DJF is assigned to the previous year.

    For monthly data with cftime timestamps and first_month=-1 (default), the result
    of averaging annually is identical to averaging seasonally, then annually.

    For monthly data with cftime timestamps and first_month=4, the year is defined as
    J F M] [A M J J A S O N D
      Y-1           Y
    i.e. January-March are assigned to previous year. This split
    is similar to Perkins & Hakim (2021).

    Args:
        ds: dataset at monthly or seasonal resolution
        first_month: for monthly data, the month to begin the year at. Default: -1 (Dec of previous year, to be
            consistent with averaging seasonally, then annually).
        weight_months: weight months by their length
        remove_incomplete: remove years for which not all months/seasons are present

    Returns:
        Annually averaged dataset
    """
    if is_annual_resolution(ds):
        return ds

    # Assign January-March to previous year
    if has_cftime_timedim(ds) or has_npdatetime_timedim(ds):
        if first_month is None:
            first_month = -1

        if isinstance(ds, xr.Dataset) and "time" in ds.cf.bounds:
            # Drop time bounds since they might be cftime objects which interferes with weighting
            ds = ds.drop_vars(ds.cf.bounds["time"])

        # Assume monthly timestamps
        if first_month < 0:
            year = ds.time.dt.year + (ds.time.dt.month > 12 + first_month)
        else:
            year = ds.time.dt.year - (ds.time.dt.month < first_month)

        if weight_months:
            month_length = ds.time.dt.days_in_month
            month_length = month_length.assign_coords(time=year).groupby("time")
            weights = month_length / month_length.mean()
            ds = ds.assign_coords(time=year) * weights
        else:
            ds = ds.assign_coords(time=year)

        groupby = ds.groupby("time")
        groups = list(groupby.groups.values())
        annual_averages = groupby.mean()
        if remove_incomplete:
            if len(groups[0]) < 12:
                annual_averages = annual_averages.isel(time=slice(1, None))
            if len(groups[-1]) < 12:
                annual_averages = annual_averages.isel(time=slice(None, -1))
        return annual_averages
    else:
        if first_month is not None:
            raise NotImplementedError(
                "Changing alignment of annual averaging of seasonal data not yet supported"
            )

        ds_tuple = use_tuple_time_coords(ds)
        season = ds_tuple.season.values
        year = ds_tuple.year.values

        if weight_months:
            season_lengths = xr.DataArray(
                [90.25, 92, 92, 91],  # 90.25 to account for leap years
                dims="time",
                coords=dict(time=[Season.DJF, Season.MAM, Season.JJA, Season.SON]),
            )
            season_lengths = (
                season_lengths.sel(time=season).assign_coords(time=year).groupby("time")
            )
            weights = season_lengths / season_lengths.mean()
            ds = ds.assign_coords(time=year) * weights
        else:
            ds = ds.assign_coords(time=year)

        # Assume seasonal float timestamps
        groupby = ds.groupby("time")
        groups = list(groupby.groups.values())
        annual_averages = groupby.mean()
        if remove_incomplete:
            if len(groups[0]) < 4:
                annual_averages = annual_averages.isel(time=slice(1, None))
            if len(groups[-1]) < 4:
                annual_averages = annual_averages.isel(time=slice(None, -1))
        return annual_averages


def average_seasonally(ds: T_Xarray, weight_months=True) -> T_Xarray:
    # Average monthly data to DJF, MAM, JJA, SON and remove partial seasons

    if has_cftime_timedim(ds) or has_npdatetime_timedim(ds):
        # Generate labels, assigning December to next year
        year = ds.time.dt.year + (ds.time.dt.month == 12)
        decimal_season = year.data + [
            map_season_to_decimal(Season[s]) for s in ds.time.dt.season.data
        ]

        if weight_months:
            month_length = ds.time.dt.days_in_month
            month_length = month_length.assign_coords(time=decimal_season).groupby("time")
            weights = month_length / month_length.mean()
            ds = ds.assign_coords(time=decimal_season) * weights
        else:
            ds = ds.assign_coords(time=decimal_season)
    else:
        if weight_months:
            raise NotImplementedError("Weighting months not supported for float timestamps")

        ds = ds.assign_coords(time=round_to_nearest_season(ds.time))

    # Group by season and remove first + last year
    groupby = ds.groupby("time")
    groups = list(groupby.groups.values())
    ds_seasonal = groupby.mean()
    if len(groups[0]) < 3:
        ds_seasonal = ds_seasonal.isel(time=slice(1, None))
    if len(groups[-1]) < 3:
        ds_seasonal = ds_seasonal.isel(time=slice(None, -1))
    return ds_seasonal


def anomalize(
    ds: T_Xarray,
    period=None,
    climatology=None,
    return_climatology=False,
    persist_climatology=True,
    use_ensemble_mean=False,
) -> (T_Xarray) | tuple[(T_Xarray), T_Xarray]:
    """
    Compute anomalies by removing climatology.

    Args:
        ds: Dataset to anomalize.
        period: Period to compute climatology over. Default: all times in dataset.
        use_ensemble_mean: Whether the climatology should be computed from the ensemble mean, or
            for each member individually (individually is generally preferred)

    Returns:
        Anomalized dataset
    """
    if period is None:
        period = slice(None)
    else:
        period = slice(*period)

    if has_npdatetime_timedim(ds) or has_cftime_timedim(ds):
        # Monthly resolution
        if climatology is None:
            climatology = ds.sel(time=period).groupby("time.month").mean("time")
            if persist_climatology:
                climatology = climatology.compute()
        anomalies = ds.groupby("time.month") - climatology
    elif is_annual_resolution(ds):
        # Annual resolution
        if climatology is None:
            climatology = ds.sel(time=period).mean("time")
            if persist_climatology:
                climatology = climatology.compute()
        anomalies = ds - climatology
    else:
        # Seasonal resolution
        ds = add_season_coords(ds)
        if climatology is None:
            if use_ensemble_mean and "ens" in ds.dims:
                # Use the ensemble mean climatology for all ensemble members
                climatology = ds.mean("ens")
            else:
                climatology = ds
            climatology = climatology.sel(time=period).groupby("season").mean("time")
            if persist_climatology:
                climatology = climatology.compute()
        anomalies = (ds.groupby("season") - climatology).drop_vars("season")

    if ds.chunks:
        anomalies = anomalies.chunk(ds.chunks)

    if return_climatology:
        return anomalies, climatology
    else:
        return anomalies


def annualize_seasonal_data(ds: T_Xarray, seasonality: Season | list[Season] = None) -> T_Xarray:
    """
    Annualize seasonal data with arbitrary seasonality. This converts from seasonal to annual data
    by taking the average over the specified seasons. This is for example useful for seasonally
    sensitive proxies.

    Args:
        ds: Dataset to annualize. Must have seasonal resolution.
        seasonality: List of seasons for annualization. If DJF is included, all seasons in the list before are
            considered to belong to the previous year. Example: [SON, DJF, MAM] means SON of previous year and
            DJF and MAM of current year. Seasons must be consecutive.
            Default: [DJF, MAM, JJA, SON]

    Returns:
        Annualized dataset
    """
    if seasonality is None:
        seasonality = Season.ANNUAL

    if not isinstance(seasonality, list):
        seasonality = [seasonality]

    seasons_previous_year, seasons_current_year = split_seasonality(seasonality)

    # Check if seasons are consecutive and in order within year
    for season, nextseason in itertools.pairwise(seasons_previous_year):
        if season.value + 1 != nextseason.value:
            raise ValueError("Seasons must be consecutive and in order")
    for season, nextseason in itertools.pairwise(seasons_current_year):
        if season.value + 1 != nextseason.value:
            raise ValueError("Seasons must be consecutive and in order")

    # Check if seasons are consecutive across years
    if seasons_previous_year and not seasons_previous_year[-1] == Season.SON:
        raise ValueError("Seasons must be contiguous across years")

    ds = use_tuple_time_coords(ds)
    # Select only seasons that will be used for annualization
    ds = ds.sel(time=ds["season"].isin(seasonality))

    # Group labels are year that season belongs to for annualization
    group_labels = xr.where(ds["season"].isin(seasons_current_year), ds["year"], ds["year"] + 1)
    groupby = ds.groupby(group_labels)
    groups = list(groupby.groups.values())

    ds_annualized = groupby.mean().rename(season="time")

    # Check that there are no partial years
    if len(groups[0]) < len(seasonality):
        ds_annualized = ds_annualized.isel(time=slice(1, None))
    if len(groups[-1]) < len(seasonality):
        ds_annualized = ds_annualized.isel(time=slice(None, -1))

    return ds_annualized


def localize_gc5thorder(dist: float | ArrayLike, loc_rad: float) -> float | ArrayLike:
    # Gaspari and Cohn (1999), Eq. 4.10
    c = loc_rad / 2
    return np.piecewise(
        dist,
        [
            np.logical_and(0 <= dist, dist <= c),
            np.logical_and(c < dist, dist <= 2 * c),
            2 * c < dist,
        ],
        [
            lambda z: (
                -((z / c) ** 5) / 4
                + (z / c) ** 4 / 2
                + (z / c) ** 3 * 5 / 8
                - (z / c) ** 2 * 5 / 3
                + 1
            ),
            lambda z: (
                (z / c) ** 5 / 12
                - (z / c) ** 4 / 2
                + (z / c) ** 3 * 5 / 8
                + (z / c) ** 2 * 5 / 3
                - (z / c) * 5
                + 4
                - (c / z) * 2 / 3
            ),
            lambda z: 0,
        ],
    )


def detrend_linear(ds, by_season=False, period=None):
    return detrend_polynomial(ds, by_season=by_season, degree=1, period=period)


def detrend_polynomial(ds, by_season=False, degree=1, period=None):
    is_dataset = False
    if isinstance(ds, xr.DataArray):
        is_dataset = True
        ds = ds.to_dataset(name="data")

    ds_for_trend = ds
    if period:
        ds_for_trend = ds_for_trend.sel(time=slice(*period))

    if by_season:
        ds_by_season = add_season_coords(ds_for_trend).groupby("season")
        coeffs = ds_by_season.map(lambda x: x.polyfit("time", deg=degree)).compute()
        trend = ds_by_season.map(
            lambda x: xr.polyval(ds.time, coeffs.sel(season=x.season))
        ).drop_vars("season")
    else:
        coeffs = ds_for_trend.polyfit("time", deg=degree).compute()
        trend = xr.polyval(ds.time, coeffs)

    trend = trend.rename({var: var.replace("_polyfit_coefficients", "") for var in trend.variables})
    ds_detrended = ds - trend

    if is_dataset:
        return ds_detrended["data"]
    return ds_detrended


def calculate_trend(ds: T_Xarray, dim="time") -> T_Xarray:
    if isinstance(ds, xr.DataArray):
        return ds.polyfit(dim=dim, deg=1).sel(degree=1)["polyfit_coefficients"]
    else:
        trend = ds.polyfit(dim=dim, deg=1).sel(degree=1)
        trend = trend.rename(
            {var: var.replace("_polyfit_coefficients", "") for var in trend.variables}
        )
        return trend


def regress_field(
    x: xr.DataArray,
    Y: xr.DataArray,
    dim: str = "time",
    fit_intercept=True,
    compute_pvalues=False,
    effective_n=True,
    alternative: str | None = None,
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    """
    Regress field onto scalar (Y onto x). Identical to multiple univariate linear regressions.

    Args:
        index: x, scalar
        da: Y, field
        dim: sampling dimension
        fit_intercept: whether to fit the intercept
        compute_pvalues: compute p-values of regression coefficient
        effective_n: use effective sample size for p-values
        alternative: Alternative hypothesis for p-values. Supported values are ``"!="`` or
            ``"two-sided"`` for non-zero slope, and ``">"`` or ``"greater"`` for
            positive slope.

    Returns:
        Regression of Y onto x, (p-values)
    """
    index, da = x, Y
    assert da.ndim == 3
    assert index.ndim == 1
    da, index = xr.align(da, index.dropna(dim), join="inner")
    da_stacked = to_math_order(stack_state(da))

    # Cannot regress field with nans, e.g. ocean fields -> mask out and fill with nan again after
    nanmask = NanMask()
    nanmask.fit(da_stacked)
    da_nonan = nanmask.forward(da_stacked)

    reg = sklearn.linear_model.LinearRegression(fit_intercept=fit_intercept)
    X = index.values[:, np.newaxis]
    y = da_nonan.values.T
    reg.fit(X, y)
    pattern = nanmask.backward(reg.coef_)
    pattern_da = unstack_state(xr.DataArray(pattern[:, 0], coords=dict(state=da_stacked.state)))

    if compute_pvalues:
        if alternative is None:
            raise ValueError("When computing p-values, must provide an alternative hypothesis")
        normalized_alternative = _normalize_alternative(alternative)

        y_pred = reg.predict(X)
        # Correlation between y and y_pred is always non-negative; recover the regression-sign.
        r = calculate_correlation(y, y_pred, axis=0) * np.sign(reg.coef_[:, 0])
        r_squared = r**2

        p = []
        for i in range(y.shape[1]):
            if effective_n:
                n_eff = estimate_effective_dof(X.T, y[:, [i]].T)
            else:
                n_eff = y.shape[0]
            if r_squared[i] == 1.0:
                t_stat = np.inf  # Avoid division by zero for perfect correlation
            else:
                t_stat = r[i] * np.sqrt(n_eff - 2) / np.sqrt(1 - r_squared[i])
            if normalized_alternative == "two-sided":
                p_value = 2 * (1 - scipy.stats.t.cdf(np.abs(t_stat), df=n_eff - 2))
            elif normalized_alternative == "greater":
                p_value = 1 - scipy.stats.t.cdf(t_stat, df=n_eff - 2)
            p.append(p_value)

        p = nanmask.backward(np.array(p))
        p_da = unstack_state(xr.DataArray(p, coords=dict(state=da_stacked.state)))
        return pattern_da, p_da
    else:
        return pattern_da


def regress_scalar(x: xr.DataArray, y: xr.DataArray, dim: str = "time") -> np.array:
    """
    Regress scalar onto scalar (y onto x). Identical to multiple univariate linear regressions.

    Args:
        index: x
        scalar: y
        dim: sampling dimension

    Returns:
        Regression of y onto x
    """
    index, scalar = x, y
    assert scalar.ndim == 1
    assert index.ndim == 1
    scalar, index = xr.align(scalar.dropna(dim), index.dropna(dim), join="inner")

    if np.all(np.isnan(scalar)) or np.all(np.isnan(index)):
        return np.nan

    reg = sklearn.linear_model.LinearRegression()
    X = index.values[:, np.newaxis]
    y = scalar.values[:, np.newaxis]
    reg.fit(X, y)

    return np.squeeze(reg.coef_)


def compute_surrogate_correlation_significance(
    a: pyleoclim.Series,
    b: pyleoclim.Series,
    alternative: str,
    n_surr: int = 1000,
    method="phaseran",
) -> float:
    """Compute surrogate-based correlation significance for two Pyleoclim series.

    This mirrors the surrogate-based branch of ``pyleoclim.Series.correlation()``
    for Pearson correlation, with support for both two-sided and one-sided
    hypotheses.

    The supported methods are:
     - phaseran: phase randomization
     - ar1sim: AR(1) model
     - uar1: AR(1) model, maximum likelihood estimate
     - CN: colored noise

    Args:
        a: First Pyleoclim series.
        b: Second Pyleoclim series.
        alternative: Alternative hypothesis. Supported values are ``"!="`` or
            ``"two-sided"`` for $r \neq 0$, and ``">"`` or ``"greater"`` for
            $r > 0$.
        n_surr: Number of phase-randomized surrogate pairs.
        method: Significance test method. Defaults to "phaseran".

    Returns:
        Monte Carlo p-value for the correlation significance test.
    """
    from pyleoclim.core.surrogateseries import SurrogateSeries

    normalized_alternative = _normalize_alternative(alternative)

    overlap = a.overlap(b)
    if overlap <= 5:
        warnings.warn(
            f"The series have insufficient overlap ({overlap} {a.time_unit}); returning nan.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")

    ms = a & b
    if list(a.time) != list(b.time):
        ms = ms.common_time(method="interp")

    series_a = ms.series_list[0]
    series_b = ms.series_list[1]

    corr = scipy.stats.pearsonr(series_a.value, series_b.value).statistic

    surrogates_a = SurrogateSeries(method=method, number=n_surr)
    surrogates_a.from_series(series_a)
    surrogates_b = SurrogateSeries(method=method, number=n_surr)
    surrogates_b.from_series(series_b)

    surrogate_corrs = np.empty(n_surr)
    for idx in range(n_surr):
        surrogate_corrs[idx] = scipy.stats.pearsonr(
            surrogates_a.series_list[idx].value,
            surrogates_b.series_list[idx].value,
        ).statistic

    if normalized_alternative == "two-sided":
        return float(np.mean(np.abs(surrogate_corrs) >= np.abs(corr)))
    elif normalized_alternative == "greater":
        return float(np.mean(surrogate_corrs >= corr))


def compute_correlation_significance(
    a: DataArray,
    b: DataArray,
    alternative: str,
    method="phaseran",
    n_surr=1000,
    corr_threshold=None,
    dim="time",
    seed=41561564,
) -> DataArray:
    """
    Compute the statistical significance of a correlation. An optional correlation threshold can
    reduce the computational burden by only computing the significance for low correlations.

    See https://pyleoclim-util.readthedocs.io/en/latest/core/api.html#pyleoclim.core.series.Series.correlation
    for a description of the methods.

    Args:
        a: First timeseries
        b: Second timeseries
        alternative: Alternative hypothesis. Supported values are ``"!="`` or
            ``"two-sided"`` for $r \neq 0$, and ``">"`` or ``"greater"`` for
            $r > 0$.
        method: Significance test method. Defaults to "phaseran".
        n_surr: Number of surrogate timeseries. Defaults to 1000.
        corr_threshold: Only compute significance for correlations below threshold if given. Defaults to None.
        dim: Dimension along which to compute correlation. Defaults to "time".

    Returns:
        p-value of correlation significance
    """

    def _corr_sig(a, a_time, b, b_time):
        if np.isnan(a).any() or np.isnan(b).any():
            return np.nan
        series1 = pyleoclim.Series(a_time, a, verbose=False)
        series2 = pyleoclim.Series(b_time, b, verbose=False)
        if method == "phaseran":
            return compute_surrogate_correlation_significance(
                series1,
                series2,
                n_surr=n_surr,
                alternative=alternative,
            )
        return series1.correlation(
            series2,
            statistic="pearsonr",
            method=method,
            number=n_surr,
            mute_pbar=True,
        ).p

    from pyleoclim.core.surrogateseries import supported_surrogates

    normalized_alternative = _normalize_alternative(alternative)

    if normalized_alternative == "two-sided" and method not in supported_surrogates:
        raise NotImplementedError(
            "Only surrogate-based methods support the two-sided alternative hypothesis"
        )

    a, b = xr.align(a.dropna(dim), b.dropna(dim), join="inner")

    if corr_threshold:
        corr = xr.corr(a, b, dim=dim)
        a = a.where(corr < corr_threshold)
        b = b.where(corr < corr_threshold)

    # Rechunk so the full time axis is in a single chunk; spatial dims become parallel dask tasks.
    a = a.chunk({dim: -1})
    b = b.chunk({dim: -1})

    with local_np_seed(seed):
        p_values = xr.apply_ufunc(
            _corr_sig,
            a,
            a[dim],
            b,
            b[dim],
            input_core_dims=[[dim], [dim], [dim], [dim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        ).compute()
    return p_values


def compute_correlation_and_ce(
    test: DataArray,
    ver: DataArray,
    alternative: str,
    n_surr: int = 1000,
    period: tuple[float, float] | None = None,
) -> tuple[float, float, float]:
    """
    Compute comparison metrics between two time series.

    Calculates the correlation, correlation p-value, and coefficient of efficiency
    between two one-dimensional DataArrays after aligning them.

    Args:
        test: The first time series to compare (must be 1-dimensional).
        ver: The second time series to compare (must be 1-dimensional).
        alternative: The alternative hypothesis for correlation p-values. See compute_correlation_significance().
        n_surr: Number of surrogate timeseries. Defaults to 1000.
        period: Period over which to compare test and ver.

    Returns:
        A tuple containing:
        - corr (float): Pearson correlation coefficient between test and ver.
        - corr_p (float): P-value indicating the statistical significance of the correlation.
        - ce (float): Coefficient of efficiency between test and ver.
    """
    assert test.ndim == 1
    assert ver.ndim == 1
    if period is not None:
        test = test.sel(time=slice(*period))
        ver = ver.sel(time=slice(*period))
    test, ver = xr.align(test, ver)
    corr = xr.corr(test, ver).compute().item()
    ce_ = ce(test, ver).compute().item()
    corr_p = (
        compute_correlation_significance(test, ver, alternative=alternative, n_surr=n_surr)
        .compute()
        .item()
    )
    return corr, corr_p, ce_


def compute_sliding_window_trend(
    da: xr.DataArray, window: int, dim="time", center=True, rechunk=True
) -> xr.DataArray:
    """
    Computes the slope of the linear trend over a sliding window.

    Args:
        da: DataArray input
        window: window size (number of samples, not time units)
        dim: Dimension to slide over. Defaults to "time".
        center: Set the labels at the center or the right edge of the window. Defaults to True.
        rechunk: Whether to automatically rechunk the output to avoid exploding chunk sizes

    Returns:
        DataArray with trends
    """
    windows = da.rolling(
        {dim: window},
        center=center,
    ).construct("window", sliding_window_view_kwargs={"automatic_rechunk": rechunk})
    complete = (windows.count("window") >= window).compute()
    trends = windows.polyfit("window", deg=1)["polyfit_coefficients"].sel(degree=1)
    return trends.where(complete, drop=True)
