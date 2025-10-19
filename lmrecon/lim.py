from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import dask.array
import numpy as np
import xarray as xr
from numpy.linalg import eig, eigh, eigvals, inv
from tqdm import tqdm

from lmrecon.logger import get_logger
from lmrecon.util import get_timestamp, is_dask_array

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

logger = get_logger(__name__)


class LIM:
    def __init__(self):
        self.tau = None
        self.G_tau: ArrayLike = None
        self.L: ArrayLike = None
        self.Q_evals: ArrayLike = None
        self.Q_evecs: ArrayLike = None
        self.state_coords = None
        self.Q_eval_scale_factor = None
        self.Nx = None  # state length

    def save(self, directory: Path, metadata: dict | None = None):
        directory /= get_timestamp()
        directory.mkdir(parents=True, exist_ok=True)
        outfile = directory / "lim.pkl"
        logger.info(f"Saving LIM to {outfile}")
        pickle.dump(self, outfile.open("wb"))

        properties = {
            "Nx": self.Nx,
            "tau": self.tau,
        }
        if metadata:
            metadata |= properties
        else:
            metadata = properties
        json.dump(metadata, (directory / "metadata.json").open("w"), indent=4)

    @classmethod
    def load(cls, file: Path | str):
        return pickle.load(Path(file).open("rb"))

    def fit(self, data: xr.DataArray):
        time = data.time.data
        taus = np.diff(time)
        self.tau = taus.mean()
        # Must have uniform dt
        assert all(np.isclose(taus, self.tau)), "Sample times must be uniform"

        self.state_coords = data.state
        data = data.data  # extract Dask array
        self.Nx = len(self.state_coords)

        if not (data.mean(axis=1) < 1e-5).all():
            logger.warning("LIM training data must have zero mean")

        data_0 = data[:, :-1]
        data_tau = data[:, 1:]

        C_tau = (data_tau @ data_0.T) / (data_0.shape[1] - 1)
        C_0 = (data_0 @ data_0.T) / (data_0.shape[1] - 1)

        self._fit_dynamics(C_0, C_tau)
        self._fit_noise(C_0)

    def _fit_dynamics(self, C_0, C_tau):
        self.G_tau = C_tau @ inv(C_0)
        if is_dask_array(self.G_tau):
            self.G_tau = self.G_tau.compute()

        G_evals, G_evecs = eig(self.G_tau)
        L_evals = np.log(G_evals) / self.tau
        self.L = G_evecs @ np.diag(L_evals) @ inv(G_evecs)

        if np.any(L_evals.real >= 0):
            positive_L_evals = L_evals[L_evals.real >= 0]
            negative_L_evals = L_evals[L_evals.real < 0]
            logger.debug(f"Positive L eigenvalues:\n{positive_L_evals}")
            logger.debug(f"Negative L eigenvalues:\n{negative_L_evals}")
            raise ValueError("Positive eigenvalues detected in forecast matrix L.")

    def _fit_noise(self, C_0):
        # Adapted from https://github.com/frodre/pyLIM/blob/master/pylim/LIM.py
        # .H instead of .conj().T is not available for np.ndarray, only for np.matrix
        Q = -(self.L @ C_0 + C_0 @ self.L.conj().T)
        if is_dask_array(Q):
            Q = Q.compute()

        # Check if Q is Hermitian
        if not np.isclose(Q, Q.conj().T, atol=1e-10).all():
            raise ValueError("Q is not Hermitian (Q should equal Q.H)")

        Q_evals, Q_evecs = eigh(Q)
        sort_idx = Q_evals.argsort()
        Q_evals = Q_evals[sort_idx][::-1]
        Q_evecs = Q_evecs[:, sort_idx][:, ::-1]
        num_neg = (Q_evals < 0).sum()

        self.Q_eval_scale_factor = 1
        if num_neg > 0:
            num_left = len(Q_evals) - num_neg
            logger.info(
                f"Found {num_neg:d} modes with negative eigenvalues in the noise covariance Q, "
                f"removing them and rescaling {num_left:d} remaining eigenvalues of Q"
            )

            pos_q_evals = Q_evals[Q_evals > 0]
            self.Q_eval_scale_factor = Q_evals.sum() / pos_q_evals.sum()
            assert self.Q_eval_scale_factor <= 1
            logger.info(f"Q eigenvalue rescaling: {self.Q_eval_scale_factor:1.2f}")

            Q_evals = Q_evals[:-num_neg] * self.Q_eval_scale_factor
            Q_evecs = Q_evecs[:, :-num_neg]

        self.Q_evals = Q_evals
        # For some reason, this speeds stochastic forecasts up by a factor 3x, even if it's not Dask
        self.Q_evecs = np.array(Q_evecs)

    def print_properties(self):
        print("G1:", self.G_tau)
        print("Eigenvalues of G:", eigvals(self.G_tau))
        print("L:", self.L)
        print("Eigenvalues of L:", eigvals(self.L))
        print("Eigenvalues of Q:", self.Q_evals)

    def _validate_initial(self, initial: xr.DataArray):
        if initial.dims[0] != "state":
            raise ValueError("Row dimension must be state")

        if not np.array_equal(self.state_coords.data, initial.state.data):
            raise ValueError("Initial state dimension must match training data state dimension")

    def _get_initial_time(self, initial: xr.DataArray) -> float:
        if "time" not in initial.coords or (initial.time.ndim > 0 and len(initial.time) > 1):
            raise ValueError("Initial condition must have a single time coordinate")
        return float(initial.time.item())

    def forecast_deterministic(self, initial: xr.DataArray, n_steps) -> xr.DataArray:
        self._validate_initial(initial)
        if len(initial.dims) > 2:
            raise ValueError("Only one sampling dimension is allowed")
        if len(initial.dims) == 2 and initial.shape[1] > 1:
            raise ValueError("Only a single sample is allowed")

        time = self._get_initial_time(initial) + np.arange(0, (n_steps + 0.9) * self.tau, self.tau)

        initial_np = initial.values
        # Remove time dimension if exists
        if len(initial.dims) > 1:
            initial_np = initial_np.squeeze(axis=-1)

        forecast_np = np.zeros((self.Nx, n_steps + 1))
        forecast_np[:, 0] = initial_np

        #### Actual forecast

        G = self.G_tau
        for i in range(1, n_steps + 1):
            forecast_np[:, i] = G @ initial_np
            G = self.G_tau @ G

        ####

        new_coords = initial.coords.copy()
        new_coords["time"] = time
        forecast = xr.DataArray(
            dask.array.from_array(forecast_np),
            dims=["state", "time"],
            coords=new_coords,
        )

        return forecast

    def forecast_stochastic(
        self,
        initial: xr.DataArray,
        n_steps,
        n_int_steps_per_tau,
        n_ensemble=None,
        progressbar=True,
        seed=None,
    ):
        """
        Forecast using stochastic integration. If n_ensemble > 1 but only a single IC is given, the same IC will be used
        for all members.

        Args:
            initial: Initial conditions
            n_steps: number of tau-length periods to forecast
            n_int_steps_per_tau: number of integration steps per tau-length period
            n_ensemble: number of ensemble members

        Returns:
            Ensemble forecast
        """
        self._validate_initial(initial)
        if len(initial.dims) > 2 and initial.dims[-1] != "time":
            raise ValueError("Last dimension must be time")

        if "ens" in initial.dims:
            if n_ensemble is None:
                n_ensemble = len(initial.ens)
            elif n_ensemble != len(initial.ens):
                raise ValueError(
                    "Conflicting values for n_ensemble and initial condition ens dimension"
                )
        elif n_ensemble is None:
            raise ValueError("Most provide either n_ensemble or initial condition ens dimension")

        initial_np = initial.values
        # Remove time dimension if exists
        if initial.dims[-1] == "time":
            initial_np = initial_np.squeeze(axis=-1)
        # Add ensemble dimension if not present
        if "ens" not in initial.dims:
            initial_np = initial_np[:, np.newaxis]

        time = self._get_initial_time(initial) + np.arange(0, (n_steps + 0.9) * self.tau, self.tau)

        forecast_np = np.zeros((self.Nx, n_ensemble, n_steps + 1))
        # Uses ensemble members as IC if present, or same IC for all ensemble members
        forecast_np[:, :, 0] = np.broadcast_to(initial_np, (self.Nx, n_ensemble))

        #### Actual forecast
        # Adapted from https://github.com/frodre/pyLIM/blob/master/pylim/LIM.py

        rng = np.random.default_rng(seed)
        dt = self.tau / n_int_steps_per_tau
        n_int_steps = int(n_int_steps_per_tau * n_steps)
        num_evals = self.Q_evals.shape[0]

        state_1 = forecast_np[:, :, 0]

        # Do stochastic integration
        for i in tqdm(range(n_int_steps + 1), disable=not progressbar, unit="step"):
            # L * x * dt
            deterministic = (self.L @ state_1) * dt
            # S * eta * dt
            stochastic = self.Q_evecs @ (
                np.sqrt(self.Q_evals[:, np.newaxis] * dt) * rng.normal(size=(num_evals, n_ensemble))
            )
            state_2 = state_1 + deterministic + stochastic
            state_mid = (state_1 + state_2) / 2
            state_1 = state_2
            if i % n_int_steps_per_tau == 0:
                forecast_np[:, :, i // n_int_steps_per_tau] = state_mid.real

        ####

        if "ens" in initial.dims:
            ens_coords = initial.ens
        else:
            ens_coords = range(n_ensemble)
        forecast = xr.DataArray(
            dask.array.from_array(forecast_np),
            dims=["state", "ens", "time"],
            coords={
                "state": self.state_coords,
                "ens": ens_coords,
                "time": time,
            },
        )

        return forecast
