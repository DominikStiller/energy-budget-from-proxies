from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from lmrecon.logger import get_logger

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from xarray import DataArray

    from lmrecon.psm import PhysicalSpacePSM, ReducedSpacePSM
    from lmrecon.time import YearAndSeason

logger = get_logger(__name__)


class EnKF(ABC):
    pass


class InstantaneousSerialEnKF(EnKF, ABC):
    def __init__(self, update_method="EnSRF"):
        self._update_fn = {"EnSRF": ensrf_update}.get(update_method, ensrf_update)

    def assimilate(
        self,
        prior_da: DataArray,
        obs: list[float],
        psms: list[PhysicalSpacePSM | ReducedSpacePSM],
        loc_matrices: list[ArrayLike] | None = None,
    ) -> DataArray:
        """
        Assimilate multiple instantaneous observations.

        Args:
            prior_da: prior state [n_state x n_ens]
            obs: scalar observations [n_obs]
            psms: list of PSMs corresponding to observations [n_obs]
            loc_matrices: list of localization matrices, if applicable [[n_state x 1] x n_obs]

        Returns:
            Posterior state [n_state x n_ens]
        """
        # Assign prior to posterior in case there are no obs
        posterior = np.array(prior_da.data)

        assert len(obs) == len(psms)

        for i in tqdm(range(len(obs)), unit="obs"):
            prior = posterior

            posterior = self._update_fn(
                prior,
                obs[i],
                psms[i].forward(prior),
                psms[i].err_std,
                loc_matrices[i] if loc_matrices else None,
            )

            assert isinstance(posterior, np.ndarray)
            if np.all(~np.isfinite(posterior)):
                # Fail quickly if something went wrong during update
                raise ValueError("All posterior values are non-finite")

        posterior_da = prior_da.copy(data=posterior)
        return posterior_da


class TimeAveragingSerialEnKF(EnKF, ABC):
    def __init__(self, update_method="EnSRF"):
        self._update_fn = {"EnSRF": ensrf_update}.get(update_method, ensrf_update)

    def assimilate(
        self,
        priors_da: DataArray,
        obs: list[float],
        psms: list[PhysicalSpacePSM | ReducedSpacePSM],
        seasonalities: list[list[YearAndSeason]],
        loc_matrices: list[ArrayLike] | None = None,
    ) -> DataArray:
        """
        Assimilate multiple time-averaged observations. This method is based on Huntley & Hakim (2010) and described
        in detail in Appendix 1.

        The method is based on the assumption that perturbations from the time mean are uncorrelated with the mean.
        Then, the time mean can be updated using the normal EnSRF (which itself decomposes the time mean into an
        ensemble mean and perturbations) and prior time perturbations are added to the posterior time mean to obtain
        the posterior at every time step.

        Args:
            priors_da: prior states [n_state x n_ens x n_time]
            obs: scalar observations [n_obs]
            psms: list of PSMs corresponding to observations [n_obs]
            seasonalities: seasonalities for each observations [[n_season] x n_obs]
            loc_matrices: list of localization matrices, if applicable [[n_state x 1] x n_obs]

        Returns:
            Posterior state [n_state x n_ens x n_time]
        """
        # Assign prior to posterior in case there are no obs
        posteriors = np.array(priors_da.data)  # [n_state x n_ens x n_time]

        assert priors_da.dims == ("state", "ens", "time")
        assert len(obs) == len(psms)

        for i in tqdm(range(len(obs)), unit="obs"):
            priors = posteriors
            # Indexes along time axis of priors that are needed for this obs
            prior_time_idxs_i = [priors_da.indexes["time"].get_loc(s) for s in seasonalities[i]]
            prior_i = priors[
                :, :, prior_time_idxs_i
            ]  # priors corresponding to the averaging time of the obs

            # Eq. 14; decompose prior into time mean and perturbations
            prior_i_mean = prior_i.mean(axis=2)
            prior_i_perturbations = prior_i - prior_i_mean[..., np.newaxis]

            # Eq. 13; take time mean of obs, not obs of time mean
            obs_est_i = np.mean(
                [psms[i].forward(priors[:, :, time_idx]) for time_idx in prior_time_idxs_i], axis=0
            )

            posterior_i_mean = self._update_fn(
                prior_i_mean,
                obs[i],
                obs_est_i,
                psms[i].err_std,
                loc_matrices[i] if loc_matrices else None,
            )

            # Eq. 22; add perturbations back to prior time mean
            posteriors[:, :, prior_time_idxs_i] = (
                posterior_i_mean[..., np.newaxis] + prior_i_perturbations
            )

            assert isinstance(posteriors, np.ndarray)
            if np.all(~np.isfinite(posteriors)):
                # Fail quickly if something went wrong during update
                raise ValueError("All posterior values are non-finite")

        posteriors_da = priors_da.copy(data=posteriors)
        return posteriors_da


class TimeStackingSerialEnKF(EnKF, ABC):
    def __init__(self, update_method="EnSRF"):
        self._update_fn = {"EnSRF": ensrf_update}.get(update_method, ensrf_update)

    def assimilate(
        self,
        priors_da: DataArray,
        obs: list[float],
        psms: list[PhysicalSpacePSM | ReducedSpacePSM],
        seasonalities: list[list[YearAndSeason]],
        loc_matrices: list[ArrayLike] | None = None,
    ) -> DataArray:
        """
        Assimilate multiple time-averaged observations. This method is used by Meng & Hakim (2025).

        The state from all time steps is stacked into one tall state vector, and the EnKF is applied
        to this stacked state. This may have less variability than the time-averaging EnKF.

        Args:
            priors_da: prior states [n_state x n_ens x n_time]
            obs: scalar observations [n_obs]
            psms: list of PSMs corresponding to observations [n_obs]
            seasonalities: seasonalities for each observations [[n_season] x n_obs]
            loc_matrices: list of localization matrices, if applicable [[n_state x 1] x n_obs]

        Returns:
            Posterior state [n_state x n_ens x n_time]
        """
        # Assign prior to posterior in case there are no obs
        posteriors = np.array(priors_da.data)  # [n_state x n_ens x n_time]
        n_state, n_ens, n_time = posteriors.shape

        assert priors_da.dims == ("state", "ens", "time")
        assert len(obs) == len(psms)

        # Make time dimensions the first dimension to facilitate stacking
        # [n_state x n_ens x n_time] -> [n_time x n_state x n_ens]
        posteriors = np.moveaxis(posteriors, -1, 0)

        for i in tqdm(range(len(obs)), unit="obs"):
            priors = posteriors
            # Indexes along time axis of priors that are needed for this obs
            prior_time_idxs_i = [priors_da.indexes["time"].get_loc(s) for s in seasonalities[i]]
            prior_i = priors[
                prior_time_idxs_i, :, :
            ]  # priors corresponding to the seasonality of the obs
            n_time_i = len(seasonalities[i])

            # Calculate time-mean of prior for forward model
            prior_i_mean = prior_i.mean(axis=0)

            # Stack state and time dimensions to obtain 2D array
            # [n_time x n_state x n_ens] -> [(n_time * n_state) x n_ens]
            prior_i_stacked = prior_i.reshape(n_time_i * n_state, n_ens)

            posterior_i_stacked = self._update_fn(
                prior_i_stacked,
                obs[i],
                psms[i].forward(prior_i_mean),
                psms[i].err_std,
                loc_matrices[i] if loc_matrices else None,
            )

            # Unstack state and time dimensions
            # [(n_time * n_state) x n_ens] -> [n_time x n_state x n_ens]
            posteriors[prior_time_idxs_i, :, :] = posterior_i_stacked.reshape(
                n_time_i, n_state, n_ens
            )

            assert isinstance(posteriors, np.ndarray)
            if np.all(~np.isfinite(posteriors)):
                # Fail quickly if something went wrong during update
                raise ValueError("All posterior values are non-finite")

        # Make time dimensions the last dimension again
        # [n_time x n_state x n_ens] -> [n_state x n_ens x n_time]
        posteriors = np.moveaxis(posteriors, 0, -1)

        posteriors_da = priors_da.copy(data=posteriors)
        return posteriors_da


def ensrf_update(
    prior: ArrayLike,
    obs: float,
    obs_est: ArrayLike,
    obs_error_std: float,
    loc_matrix: ArrayLike | None,
) -> ArrayLike:
    """
    Implements the serial ensemble square root filter (EnSRF; Whitaker & Hamill, 2002). Localization is supported
    for the prior covariance matrix in state space.

    Args:
        prior: prior state [n_state x n_ens]
        obs: scalar observation
        obs_est: prior observation estimates [1 x n_ens]
        obs_error_std: observation error std
        loc_matrix: localization matrix [n_state x 1]

    Returns:
        posterior state [n_state x n_ens]
    """
    # Experiments showed that there is no benefit to using scipy's sparse array classes, despite
    # the sparsity of a localized K. Also, numba does not accelerate this function since numpy's
    # SIMD-based functions are already optimal
    Nx, Ne = prior.shape

    x_mean = prior.mean(axis=1, keepdims=True)
    X = prior - x_mean

    y_mean = obs_est.mean()
    Ye = obs_est - y_mean
    var_innovation = obs_est.var(ddof=1) + obs_error_std**2

    # Houtekamer & Mitchell (2001), Eq. 2, 3 + 6
    cov_xy = (X @ Ye.T) / (Ne - 1)
    if loc_matrix is not None:
        cov_xy *= loc_matrix
    K = cov_xy / var_innovation

    # Update mean (Whitaker & Hamill, 2002, Eq. 4)
    x_mean_a = x_mean + K * (obs - y_mean)

    # Update perturbations (Whitaker & Hamill, 2002, Eq. 5 + 13)
    alpha = 1 / (1 + np.sqrt(obs_error_std**2 / var_innovation))
    Xa = X - (alpha * K) @ Ye

    posterior = x_mean_a + Xa

    return posterior


def inflate_covariance(da: ArrayLike, inflation_factor: float) -> ArrayLike:
    """
    Constant covariance inflation for every location by scaling ensemble variance.

    Args:
        da: state [n_state x n_ens]
        inflation_factor: factor by which to increase variance

    Returns:
        inflated state
    """
    da_mean = da.mean(axis=1, keepdims=True)
    da_pert = da - da_mean

    da_pert *= np.sqrt(inflation_factor)
    da_infl = da_mean + da_pert

    return da_infl
