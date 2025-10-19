from __future__ import annotations

import pickle
from enum import Enum, auto
from typing import TYPE_CHECKING

import dask.array
import numpy as np

from lmrecon.logger import get_logger
from lmrecon.stats import estimate_effective_dof
from lmrecon.util import is_dask_array

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import ArrayLike

logger = get_logger(__name__)


class EOFMethod(Enum):
    DASK = auto()
    NUMPY = auto()


class EOF:
    def __init__(self, rank=None, rotation=None, rotation_scale=True, rotation_normalize=True):
        self.rank = rank
        self.rotation = rotation
        self.rotation_scale = rotation_scale
        self.rotation_normalize = rotation_normalize

        self.U: ArrayLike | None = None
        self.S: ArrayLike | None | None = None
        self.Vh: ArrayLike | None | None = None

        self.total_variance: ArrayLike | None = None
        self.variance_per_mode: ArrayLike | None = None
        self.variance_fraction_per_mode: ArrayLike | None = None
        self.variance_retained: ArrayLike | None = None
        self.variance_fraction_retained: ArrayLike | None = None

    def __getstate__(self):
        state = self.__dict__.copy()
        if isinstance(state["U"], dask.array.Array):
            state["U"] = state["U"].compute()
        if isinstance(state["S"], dask.array.Array):
            state["S"] = state["S"].compute()
        if self.Vh is not None and isinstance(state["Vh"], dask.array.Array):
            state["Vh"] = state["Vh"].compute()
        return state

    def save(self, directory: Path, suffix: str | None = None):
        directory.mkdir(parents=True, exist_ok=True)
        outfile = directory / ("eof.pkl" if suffix is None else f"eof-{suffix}.pkl")
        logger.info(f"Saving EOF to {outfile}")
        pickle.dump(self, outfile.open("wb"))

    @classmethod
    def load(cls, file: Path):
        return pickle.load(file.open("rb"))

    def _validate_input_vector(self, data: ArrayLike):
        if not data.ndim <= 2:
            raise ValueError("Stack state vector before applying EOF")
        # assert np.logical_not(np.isnan(data).any()), "nan is not allowed in EOF input data"

    def fit(self, data: ArrayLike, method=EOFMethod.DASK):
        self._validate_input_vector(data)

        input_rank = min(data.shape)
        if self.rank is None:
            self.rank = input_rank
        elif input_rank < self.rank:
            logger.warning(
                f"Insufficient data for EOF with rank {self.rank}, output rank will be {input_rank}"
            )
            self.rank = input_rank

        if method == EOFMethod.DASK and not is_dask_array(data):
            method = EOFMethod.NUMPY

        if method == EOFMethod.DASK:
            logger.debug(f"Calculating EOFs using Dask (rank = {self.rank})")
            # Reduce variation due to random algorithm through power iterations
            U, S, Vh = dask.array.linalg.svd_compressed(
                data, self.rank, n_power_iter=2, seed=2344725412
            )
            # Must be .persist(), .compute() slows down computation
            self.U = U.persist()
            self.S = S.persist()
            self.Vh = Vh.persist()
        elif method == EOFMethod.NUMPY:
            logger.debug(f"Calculating EOFs using NumPy (rank = {self.rank})")
            if is_dask_array(data):
                data = data.compute()
            U, S, Vh = np.linalg.svd(data, full_matrices=False)
            self.U = U[:, : self.rank]
            self.S = S[: self.rank]
            self.Vh = Vh[: self.rank, :]
        else:
            raise ValueError("Unknown EOF method")

        self._compute_statistics(data)
        if self.rotation is not None:
            self._rotate(data)

    def _compute_statistics(self, data: ArrayLike):
        n_samples = data.shape[1]
        # Total variance is sum of variances of all state entries
        self.total_variance = np.array(data.var(ddof=1, axis=1).sum())
        self.variance_per_mode = np.array(self.S**2 / (n_samples - 1))
        # Presumably due to sampling error, the total variance derived from the singular values can
        # be larger than the directly calculated total variance (they converge for arge n)
        self.variance_fraction_per_mode = np.array(self.variance_per_mode / self.total_variance)
        self.variance_retained = np.sum(self.variance_per_mode)
        self.variance_fraction_retained = np.sum(self.variance_fraction_per_mode)
        # North test, gives 68% confidence interval
        # Von Storch and Zwiers, 1999. Eq. (13.44).
        self.variance_fraction_per_mode_ci = self.variance_fraction_per_mode * np.sqrt(
            2 / estimate_effective_dof(data)
        )

    def _rotate(self, data: ArrayLike):
        from factor_analyzer import Rotator

        n_samples = data.shape[1]
        rotator = Rotator(self.rotation, normalize=self.rotation_normalize)

        logger.debug(f"Rotating EOFs with {self.rotation}")
        if self.rotation_scale:
            rotator.fit(self.U @ np.diag(self.S) / np.sqrt(n_samples))
        else:
            rotator.fit(self.U)

        # Inspired by https://github.com/xarray-contrib/xeofs/blob/main/xeofs/models/eof_rotator.py#L120
        self.variance_per_mode = np.array(np.sqrt(np.sum(rotator.loadings_**2, axis=0)))
        idx_by_variance = np.argsort(self.variance_per_mode)[::-1]
        self.variance_per_mode = self.variance_per_mode[idx_by_variance]

        self.variance_fraction_per_mode = np.array(self.variance_per_mode / self.total_variance)
        self.variance_fraction_per_mode_ci = self.variance_fraction_per_mode * np.sqrt(
            2 / estimate_effective_dof(data)
        )

        self.U = rotator.loadings_[:, idx_by_variance] / self.variance_per_mode[np.newaxis, :]
        self.S = np.sqrt(self.variance_per_mode * n_samples)
        # Could also rotate Vh but I do not use it anyways
        self.Vh = None

    def get_component(self, n):
        return self.U[:, n]

    def project_forwards(self, data: ArrayLike) -> ArrayLike:
        self._validate_input_vector(data)
        return self.U.T @ data

    def project_backwards(self, data: ArrayLike) -> ArrayLike:
        self._validate_input_vector(data)
        return self.U @ data
