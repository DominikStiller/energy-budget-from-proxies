from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np
import pyleoclim
import pyshtools as pysh
import scipy
import scipy.fft
import scipy.optimize
import xarray as xr
from matplotlib import pyplot as plt
from scipy.signal import butter
from xarray import DataArray
from xrscipy.signal.filters import sosfiltfilt

from lmrecon.time import use_decimal_year_time_coords
from lmrecon.util import create_pyleoclim_series, list_complement

if TYPE_CHECKING:
    from xarray.core.types import T_Xarray


def autocorrelation(da, lag=1):
    if lag < 1:
        raise ValueError(f"Lag must be positive but is {lag}")
    assert da.ndim == 1
    N = len(da)
    da_demean = da - da.mean()
    acor = np.dot(da_demean[:-lag], da_demean[lag:]) / ((N - lag) * da.var())
    return float(acor)


def compute_lag_correlations(
    a: DataArray, b: DataArray, max_lag: int, dim: str = "time"
) -> DataArray:
    """
    Computes the lag correlation for a range of lags. A correlation at negative lags means that
    a leads b (signal occurs in a first, followed by b).

    Args:
        a: First timeseries
        b: Second timeseries
        max_lag: Maximum lag in units of dim, will be used symmetrically

    Returns:
        Lag correlations for a and b
    """
    dt = np.mean(np.diff(a[dim]))
    max_lag_n = int(max_lag / dt)
    lags = np.arange(-max_lag_n, max_lag_n + 1)
    correlations = []

    for lag in lags:
        if lag < 0:
            shifted_data1 = a.shift({dim: -lag})
            corr = xr.corr(shifted_data1, b, dim=dim)
        elif lag > 0:
            shifted_data2 = b.shift({dim: lag})
            corr = xr.corr(a, shifted_data2, dim=dim)
        else:
            corr = xr.corr(a, b, dim=dim)
        correlations.append(corr)

    return xr.concat(correlations, dim="lag", coords="minimal").assign_coords(lag=lags * dt)


def prewhiten(da):
    coords = da.coords
    name = da.name
    assert da.ndim == 1
    da = da.data.copy()

    da[1:] = da[1:] - autocorrelation(da) * da[:-1]

    return xr.DataArray(da, coords=coords, name=name)


def red_noise_spectrum(f, rho, dt):
    S = (1 - rho**2) / (1 - 2 * rho * np.cos(2 * np.pi * f * dt) + rho**2)
    return S / np.sum(S)


def get_spectral_nullhypothesis(
    da, psd, significance_level, median_smoothing_fraction=0.1, nw=4, prewhite=False
):
    da = use_decimal_year_time_coords(da)
    dt = np.median(np.diff(da.time))

    if prewhite:
        # White noise
        h0 = xr.ones_like(psd.f)
        rho = None
    else:
        # Red noise
        # Uses robust method of Mann and Lees (1996), or at least parts of it
        # Loosely based on R version (https://github.com/cran/astrochron/blob/master/R/FUNCTION-mtmML96_v16.R)
        # Could also use surrogate series like in pyleoclim's signif_test
        median_length = int(median_smoothing_fraction * len(psd.f))
        if median_length % 2 == 0:
            median_length += 1
        psd_median = scipy.ndimage.median_filter(psd.values, median_length)
        S0 = np.sum(psd_median)
        rho = float(
            scipy.optimize.curve_fit(
                lambda f, rho: S0 * red_noise_spectrum(f, rho, dt),
                psd.f.values,
                psd_median,
                p0=autocorrelation(da, lag=1),
            )[0]
        )
        h0 = S0 * red_noise_spectrum(psd.f, rho, dt)

    # F test is used to determine significance threshold for multitaper method (10.1007/BF00142586)
    # The maximum number of tapers K is calculated as 2*nw (see https://github.com/nipy/nitime/blob/master/nitime/algorithms/spectral.py#L536, which is the actual method used by pyleoclim.utils.spectral.mtm),
    # although the actual K may be slightly lower
    K = int(2 * nw)
    dof_null = 2
    dof_spectrum = 2 * K - 2
    f_crit = scipy.stats.f.ppf(significance_level, dof_null, dof_spectrum)

    return h0, h0 * f_crit, rho


def compute_spectrum_multitaper(
    da: DataArray, dim="time", nw=4, standardize=False, detrend=False, adaptive=True
):
    def _compute_spectrum(values, time_values):
        detrend_method = "linear" if detrend else False
        psd = pyleoclim.utils.spectral.mtm(
            values,
            time_values,
            standardize=standardize,
            NW=nw,
            adaptive=adaptive,
            detrend=detrend_method,
        )
        return psd["psd"], psd["freq"]

    if dim == "time":
        da = use_decimal_year_time_coords(da)

        # Monthly decimal years are not evenly spaced because month lenghts differ, which causes
        # problems with mtm -> assign evenly spaced timestampts
        dt = np.diff(da.time)
        if 28 <= np.median(dt) * 365 <= 31:
            da = da.assign_coords(
                time=np.linspace(da.time[0].item(), da.time[-1].item(), len(da.time))
            )

    n_freq = len(scipy.fft.fftfreq(len(da[dim]))) // 2 + 1
    psd, freq = xr.apply_ufunc(
        _compute_spectrum,
        da,
        da[dim],
        input_core_dims=[[dim], [dim]],
        output_core_dims=[["f"], ["f"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float],
        dask_gufunc_kwargs={"allow_rechunk": True, "output_sizes": dict(f=n_freq)},
    )
    # All frequencies are the same so we can squeeze non-f dimensions
    freq = freq.isel(dict.fromkeys(list_complement(freq.dims, ["f"]), 0), drop=True)
    return psd.assign_coords(f=freq).dropna("f")


def compute_spharm_spectrum_multitaper(da: T_Xarray) -> T_Xarray:
    """
    Compute spherical harmonics spectrum using multitaper method.
    """

    def _do(x):
        # Multitaper only available for masked/localized SH -> use mask with nothing masked
        window = pysh.SHWindow.from_mask(np.ones_like(x), 4)
        k = window.number_concentrated(0.99)
        grid = pysh.SHGrid.from_array(x).expand()
        return window.multitaper_spectrum(grid, k)[0]

    spectrum = xr.apply_ufunc(
        # lambda x: pysh.SHGrid.from_array(x).expand().spectrum(),
        _do,
        da,
        input_core_dims=[["lat", "lon"]],
        output_core_dims=[["degree"]],
        vectorize=True,
        dask="parallelized",
    )
    return spectrum.assign_coords(degree=np.arange(spectrum.sizes["degree"]))


def plot_spectrum(da, nw=4, prewhite=False, density=True):
    if prewhite:
        da = prewhiten(da)

    psd = compute_spectrum_multitaper(da, nw=nw)
    h0, h0_crit, _ = get_spectral_nullhypothesis(da, psd, 0.95, nw=nw, prewhite=prewhite)

    fig, ax = plt.subplots(figsize=(14, 6))

    if density:
        ax.plot(psd.f, psd, label="Spectrum")
        ax.plot(h0.f, h0, label="Red noise fit ", ls="--", c="C1")
        ax.plot(h0_crit.f, h0_crit, label="Red noise threshold (95 %)", ls=":", c="C1")
        ax.set_ylabel("Power spectral density [K$^2$ year]")
    else:
        ax.plot(psd.f, psd * psd.f, label="Spectrum")
        ax.plot(h0.f, h0 * h0.f, label="Red noise fit ", ls="--", c="C1")
        ax.plot(h0_crit.f, h0_crit * h0_crit.f, label="Red noise threshold (95 %)", ls=":", c="C1")
        ax.set_ylabel("Power spectrum [K$^2$]")

    ax.set_xlabel("Period [year]")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.tick_params(axis="y", which="minor", left=True)
    ax.legend()

    from lmrecon.plotting import format_plot

    format_plot(major_grid=True)
    ticks = [1000, 100, 50, 20, 10, 7, 5, 3, 2, 1, 1 / 2]
    ax.set_xticks([1 / t for t in ticks], ticks)

    return ax


def plot_wavelet(da, significance_test=False, log_colors=False, **kwargs):
    ps = create_pyleoclim_series(da)
    wavelet = ps.wavelet()
    if significance_test:
        wavelet = wavelet.signif_test(method="ar1asym", qs=[0.95])

    contourf_style = kwargs
    if log_colors:
        contourf_style["norm"] = mpl.colors.LogNorm()

    fig, ax = plt.subplots(figsize=(10, 6))
    wavelet.plot(variable="power", ax=ax, contourf_style=contourf_style)


def generate_autocorrelated_sinusoid_timeseries(
    autocorrelation, frequency, timesteps_per_year=4, total_years=10000, mean=0, std=1
) -> DataArray:
    """
    Generate a sinusoidal timeseries with red noise. This is useful to test the spectral analysis.
    """
    time = np.arange(0, total_years, 1 / timesteps_per_year)

    sinusoid = np.sin(2 * np.pi * frequency * time)

    # Initialize the timeseries array
    red_noise = np.zeros(len(time))

    # Generate white noise
    white_noise = np.random.normal(mean, std, size=len(time))

    # Generate the timeseries with the specified autocorrelation
    for t in range(1, len(time)):
        red_noise[t] = autocorrelation * red_noise[t - 1] + white_noise[t]

    # Convert to xarray DataArray
    timeseries_da = xr.DataArray(red_noise + sinusoid, dims=["time"], coords=dict(time=time))

    return timeseries_da


def lowpass_filter(
    ds: T_Xarray,
    smoothing_years: float | None,
    method="butterworth",
    force_pyleoclim=False,
    **kwargs,
) -> T_Xarray:
    """
    Low-pass filter. Uses xrscipy for Butterworth and pyleoclim for other filters.

    See https://pyleoclim-util.readthedocs.io/en/latest/core/api.html#pyleoclim.core.series.Series.filter
    for other filter methods like Lanczos.

    Args:
        ds: dataset
        smoothing_years: cutoff in years
        method: Filter method. Defaults to "butterworth".

    Returns:
        Filtered dataset
    """
    if method == "butterworth" and not force_pyleoclim:
        # Prefer xrscipy's butterworth filter since it works better with xarray inputs
        # It also adds more reasonable padding (see https://github.com/LinkedEarth/Pyleoclim_util/pull/653#issuecomment-2974282732, black)
        return _lowpass_filter_xrscipy(ds, smoothing_years, **kwargs)
    else:
        return _lowpass_filter_pyleoclim(ds, smoothing_years, method=method, **kwargs)


def _lowpass_filter_xrscipy(
    ds: T_Xarray,
    smoothing_years: float | None,
    dim: str = "time",
    order: int = 3,
    **kwargs,
) -> T_Xarray:
    """
    Low-pass filter using scipy's Butterworth implementation with iir and sosfiltfilt.

    Args:
        ds: dataset
        smoothing_years: cutoff in years
        dim: dimension
        order: filter order

    Returns:
        Filtered dataset
    """
    if not smoothing_years:
        return ds
    ds = ds.dropna(dim)
    if isinstance(ds, DataArray):
        return _butter_xrscipy(
            ds, btype="low", cutoff=smoothing_years, dim=dim, order=order, **kwargs
        )
    else:
        return ds.map(lambda da: _lowpass_filter_xrscipy(da, smoothing_years, dim, order, **kwargs))


def _lowpass_filter_pyleoclim(ds: T_Xarray, smoothing_years, **kwargs) -> T_Xarray:
    """
    Low-pass filter using pyleoclim. Uses a third-order Butterworth filter by default.

    Args:
        da: data array
        smoothing_years: cutoff in years

    Returns:
        Filtered data array
    """
    if not smoothing_years:
        return ds
    if isinstance(ds, DataArray):
        return ds.copy(
            data=create_pyleoclim_series(ds).filter(cutoff_scale=smoothing_years, **kwargs).value
        )
    else:
        return ds.map(lambda da: _lowpass_filter_pyleoclim(da, smoothing_years, **kwargs))


def highpass_filter(
    ds: T_Xarray,
    cutoff: float,
    dim: str = "time",
    order: int = 3,
    **kwargs,
) -> T_Xarray:
    """
    High-pass filter using scipy's Butterworth implementation with iir and sosfiltfilt.

    Args:
        ds: dataset
        cutoff: cutoff in years
        dim: dimension
        order: filter order

    Returns:
        Filtered dataset
    """
    if isinstance(ds, DataArray):
        return _butter_xrscipy(ds, btype="high", cutoff=cutoff, dim=dim, order=order, **kwargs)
    else:
        return ds.map(lambda da: highpass_filter(da, cutoff, dim, order, **kwargs))


def bandpass_filter(
    ds: T_Xarray,
    cutoff_lower: float,
    cutoff_upper: float,
    dim: str = "time",
    order: int = 3,
    **kwargs,
) -> T_Xarray:
    """
    Band-pass filter using scipy's Butterworth implementation with iir and sosfiltfilt.

    Args:
        ds: dataset
        cutoff_lower: lower cutoff in years
        cutoff_upper: upper cutoff in years
        dim: dimension
        order: filter order

    Returns:
        Filtered dataset
    """
    if isinstance(ds, DataArray):
        return _butter_xrscipy(
            ds,
            btype="bandpass",
            cutoff=(cutoff_upper, cutoff_lower),
            dim=dim,
            order=order,
            **kwargs,
        )
    else:
        return ds.map(
            lambda da: bandpass_filter(da, cutoff_lower, cutoff_upper, dim, order, **kwargs)
        )


def _butter_xrscipy(
    da: DataArray,
    btype: str,
    cutoff: float | tuple[float, float],
    dim: str,
    order: int = 3,
    **kwargs,
) -> DataArray:
    if isinstance(cutoff, tuple):
        assert len(cutoff) == 2
        Wn = (1 / cutoff[0], 1 / cutoff[1])
    else:
        Wn = 1 / cutoff
    sos = butter(
        order,
        Wn,
        btype=btype,
        fs=1 / np.median(np.diff(da[dim])),
        output="sos",
    )
    return sosfiltfilt(sos, da.compute(), dim=dim, **kwargs).rename(da.name)
