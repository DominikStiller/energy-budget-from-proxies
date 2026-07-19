from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Self

import dask.array
import numpy as np
import pandas as pd
import xarray as xr

from lmrecon.eof import EOF
from lmrecon.logger import get_logger, logging_disabled
from lmrecon.util import NanMask, stack_state, to_math_order, unstack_state

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

logger = get_logger(__name__)


class SpaceMapper(ABC):
    def __init__(self):
        self.state_coords = None

    def _validate_input(self, data: xr.DataArray):
        if data.dims[0] != "state":
            raise ValueError("Row dimension must be state")
        if len(data.dims) > 2:
            raise ValueError("Only one sampling dimension is allowed")

    @abstractmethod
    def forward(self, data: xr.DataArray) -> xr.DataArray:
        pass

    @abstractmethod
    def backward(self, data: xr.DataArray, force_stepwise=False) -> xr.DataArray:
        pass

    def truncate_dataset(self, ds: xr.Dataset, force_stepwise=False) -> xr.Dataset:
        # Copy dataset into a DataArray field-by-field, setting non-existing fields and nans to zero
        # This is a roundabout way to preserve Dask arrays, which do not support value assignment
        # Nans must be set to zero here because input nan mask could be different from mapper (e.g., different land-sea mask)
        # This method has been validated for the MPI and MRI seasonal anomaly datasets
        ds_full = unstack_state(
            xr.DataArray(
                0, dims=("state", "time"), coords=dict(state=self.state_coords, time=ds.time)
            )
        )
        for field in set(ds.keys()) & set(ds_full.keys()):
            ds_full[field] = ds[field]
        da_full = to_math_order(stack_state(ds_full))
        # Use all here since e.g. instrumental datasets can have time-varying nans
        # This is in contrast to nan mask for EOFs, which cannot handle any nans (in class NanMask)
        nan_mask = np.array(np.isnan(da_full).all(axis=1))[:, np.newaxis]

        if nan_mask.sum() > 0:
            logger.warning(
                "Dataset to truncate contains nans, may distort results if different from mapper nan mask"
            )

        # Truncate and reset nans
        with logging_disabled():
            da_truncated = (
                self.backward(self.forward(da_full.fillna(0)), force_stepwise=force_stepwise)
                .sel(state=da_full.state)
                .where(~nan_mask)
            )
        return unstack_state(da_truncated)[set(ds.variables) & set(ds_full.variables)]


class PhysicalSpaceForecastSpaceMapper(SpaceMapper):
    def __init__(
        self,
        k: int | dict[str, int],
        separate_global_mean: list[str] | None = None,
    ):
        """
        Create the mapper.

        Forward and backward mapping is a linear operation. Therefore, the .backward_matrix can be
        used for efficient backward mapping.

        Args:
            k: EOFs to retain for each field (int for all fields, or dict for per-field values)
            separate_global_mean: list of fields for which to separate global mean from EOFs
        """
        super().__init__()

        self.k = k
        self.separate_global_mean = separate_global_mean or []

        self.fields: list[str] = None
        self.nan_masks: dict[str, NanMask] = {}
        self.eofs_individual: dict[str, EOF] = {}
        self.global_mean_std: dict[str, float] = {}
        self.global_mean_variance: dict[str, float] = {}
        self.eof_std: dict[str, float] = {}
        self.lats: dict[str, ArrayLike] = {}
        self.lat_weights: dict[str, ArrayLike] = {}
        self.state_coords: xr.DataArray = None

    def _get_k_for_field(self, field: str) -> int:
        """Get the number of EOFs to retain for a given field."""
        if isinstance(self.k, dict):
            return self.k[field]
        return self.k

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        outfile = directory / "mapper.pkl"
        logger.info(f"Saving mapper to {outfile}")
        pickle.dump(self, outfile.open("wb"))

    @classmethod
    def load(cls, file: Path | str) -> Self:
        return pickle.load(Path(file).open("rb"))

    def fit_and_forward(self, data: xr.DataArray) -> xr.DataArray:
        self.fit(data)
        return self.forward(data)

    def fit(self, data: xr.DataArray) -> None:
        logger.info("PhysicalSpaceForecastSpaceMapper.fit()")
        self._validate_input(data)
        if "field" not in data.coords:
            raise ValueError("Physical state vector must have field coordinate")

        self.state_coords = data.state
        self.fields = pd.unique(np.array(data.field))

        if isinstance(self.k, dict) and set(self.fields) != self.k.keys():
            raise ValueError("Mismatch between fields and keys of k")

        logger.info("Splitting dataset into Dask arrays")
        data_raw: dict[str, ArrayLike] = {}
        for field in self.fields:
            data_raw[field] = data.sel(field=field).data

        logger.info("Masking nans")
        data_nonan: dict[str, ArrayLike] = {}
        for field in self.fields:
            self.nan_masks[field] = NanMask()
            self.nan_masks[field].fit(data_raw[field])
            data_nonan[field] = self.nan_masks[field].forward(data_raw[field])
            self.lats[field] = self.nan_masks[field].forward(
                self.state_coords.sel(field=field).lat.data
            )[:, np.newaxis]
            self.lat_weights[field] = np.sqrt(np.cos(np.radians(self.lats[field])))

        for i, field in enumerate(self.fields):
            data_field = data_nonan[field]

            if field in self.separate_global_mean:
                # Area-weight by weight^2 for global mean since lat_weights has the sqrt for covariance
                # weighting
                logger.info(f"Separating global mean for {field}")
                data_global_mean = (data_nonan[field] * self.lat_weights[field] ** 2).sum(
                    axis=0, keepdims=True
                ).compute() / np.sum(self.lat_weights[field] ** 2)
                # Avoid in-place assignment when working with Dask arrays since it can lead to subtle bugs (https://github.com/dask/dask/issues/11607)
                data_field = data_field - data_global_mean
                self.global_mean_std[field] = np.std(data_global_mean, ddof=1).item()
                # Variance explained by the global mean in the lat-weighted space.
                # The GM and residual are orthogonal after area-weighting, so total
                # variance = GM variance + residual variance (= EOF variance_total).
                self.global_mean_variance[field] = self.global_mean_std[field] ** 2 * float(
                    np.sum(self.lat_weights[field] ** 2)
                )

            self.eofs_individual[field] = EOF(self._get_k_for_field(field))

            logger.info(f"Fitting EOF for {field} [{i + 1}/{len(self.fields)}]")
            self.eofs_individual[field].fit(data_field * self.lat_weights[field])

        logger.info("Calculating standardization factors")
        for field in self.fields:
            self.eof_std[field] = np.sqrt(float(self.eofs_individual[field].variance_retained))

        # Trigger computation of backward matrix
        _ = self.backward_matrix

    def forward(self, data: xr.DataArray) -> xr.DataArray:
        """
        Map physical state to reduced state.

        This is implemented as stepwise procedure.

        Args:
            data: state in physical space (with or without sampling dimension)

        Returns:
            state in reduced space
        """
        logger.info("PhysicalSpaceForecastSpaceMapper.forward()")

        self._validate_input(data)
        if "field" not in data.coords:
            raise ValueError("Physical state vector must have field coordinate")

        has_sampling_dimension = len(data.dims) > 1

        logger.info("Splitting dataset into Dask arrays")
        data_raw: dict[str, ArrayLike] = {}
        for field in self.fields:
            data_raw[field] = data.sel(field=field).data
            if not has_sampling_dimension:
                # Most operations expect 2D arrays -> add dummy dimension and remove later
                data_raw[field] = data_raw[field][:, np.newaxis]

        logger.info("Masking nans")
        data_nonan: dict[str, ArrayLike] = {}
        for field in self.fields:
            data_nonan[field] = self.nan_masks[field].forward(data_raw[field])

        data_eof_individual: dict[str, ArrayLike] = {}

        for i, field in enumerate(self.fields):
            data_field = data_nonan[field]

            if field in self.separate_global_mean:
                logger.info(f"Separating global mean for {field}")
                data_global_mean = np.array(
                    (data_nonan[field] * self.lat_weights[field] ** 2).sum(axis=0, keepdims=True)
                ) / np.sum(self.lat_weights[field] ** 2)
                data_field = data_field - data_global_mean

            logger.info(f"Projecting EOF for {field} [{i + 1}/{len(self.fields)}]")
            data_eof_individual[field] = self.eofs_individual[field].project_forwards(
                data_field * self.lat_weights[field]
            )

            data_eof_individual[field] = data_eof_individual[field] / self.eof_std[field]

            if field in self.separate_global_mean:
                data_eof_individual[field] = np.vstack(
                    [data_global_mean / self.global_mean_std[field], data_eof_individual[field]]
                )

        logger.info("Stacking EOF projections")
        data_stacked = dask.array.vstack(
            [data_eof_individual[field] for field in self.fields]
        ).rechunk()

        if not has_sampling_dimension:
            # Remove dummy dimension
            data_stacked = np.squeeze(data_stacked, axis=1)

        new_coords = data.drop_vars("state").coords.copy()
        new_coords["state"] = range(data_stacked.shape[0])
        return xr.DataArray(data_stacked, dims=data.dims, coords=new_coords, name=None)

    def backward(self, data: xr.DataArray, force_stepwise=False) -> xr.DataArray:
        """
        Map reduced state to physical state.

        This is implemented as matrix multiplication instead of the stepwise procedure.

        Args:
            data: state in reduced space (with or without sampling dimension)
            force_stepwise: use stepwise procedure instead of matrix multiplication

        Returns:
            state in physical space
        """
        logger.info("PhysicalSpaceForecastSpaceMapper.backward()")

        self._validate_input(data)
        has_sampling_dimension = len(data.dims) > 1

        data_array: ArrayLike = data.data
        if not has_sampling_dimension:
            data_array = data_array[:, np.newaxis]

        if force_stepwise:
            data_array_physical = self._backward_stepwise(data_array)
        else:
            data_array_physical = self.backward_matrix @ data_array
        if not has_sampling_dimension:
            data_array_physical = np.squeeze(data_array_physical, axis=1)

        new_coords = data.drop_vars("state").coords.copy()
        new_coords["state"] = self.state_coords
        data_physical = xr.DataArray(
            data_array_physical, dims=data.dims, coords=new_coords, name=None
        )

        return data_physical

    def _backward_stepwise(self, data: ArrayLike) -> ArrayLike:
        """
        Space mapping is more easily expressed as a sequence of steps. However, this approach is slower than a direct
        matrix multiplication. Therefore, this method is only used to derive the corresponding mapping matrix.
        """
        data_eof_individual: dict[str, ArrayLike] = {}
        global_means: dict[str, ArrayLike] = {}

        logger.info("Splitting EOF projections by field")
        start_row = 0
        for field in self.fields:
            if field in self.separate_global_mean:
                global_means[field] = data[[start_row]] * self.global_mean_std[field]
                start_row += 1
            length = self.eofs_individual[field].rank
            data_eof_individual[field] = data[start_row : start_row + length] * self.eof_std[field]
            start_row += length

        data_nonan: dict[str, ArrayLike] = {}

        for i, field in enumerate(self.fields):
            data_field = data_eof_individual[field]

            logger.info(f"Back-projecting EOF for {field} [{i + 1}/{len(self.fields)}]")
            data_nonan[field] = self.eofs_individual[field].project_backwards(data_field)
            if data_nonan[field].ndim == 1:
                data_nonan[field] = data_nonan[field][:, np.newaxis]

            data_nonan[field] = data_nonan[field] / self.lat_weights[field]

            if field in self.separate_global_mean:
                data_global_mean = global_means[field]
                data_nonan[field] = data_nonan[field] + data_global_mean

        logger.info("Un-masking nans")
        data_raw: dict[str, ArrayLike] = {}
        for field in self.fields:
            data_raw[field] = self.nan_masks[field].backward(data_nonan[field])

        logger.info("Merging fields")
        return np.vstack(list(data_raw.values()))

    @cached_property
    def backward_matrix(self):
        """
        Derive matrix that is equivalent to _backward_stepwise() when left-multiplied to reduced state vector. This is
        useful to estimate the observation in the Kalman filter and accelerate mapping.

        The matrix is derived by backwards-mapping an identity matrix. This is equivalent to backwards-mapping the
        unit vectors in reduced space, i.e., the reduced-space basis.

        Returns:
            Backward matrix
        """
        with logging_disabled():
            return self._backward_stepwise(dask.array.eye(self.n_reduced_state)).compute()

    @cached_property
    def n_reduced_state(self) -> int:
        """
        Length of reduced state.
        """
        n_reduced_state = sum(self.eofs_individual[field].rank for field in self.fields)
        n_reduced_state += len(set(self.separate_global_mean) & set(self.fields))
        return n_reduced_state

    def variance_summary(self) -> pd.DataFrame:
        """
        Summarize variance explained by global mean and retained EOFs for each field.

        All variances are in the lat-weighted space (i.e. after multiplying by
        ``sqrt(cos(lat))``). For fields with a separated global mean the total
        variance is the sum of the global-mean variance and the residual (EOF)
        variance, since the two are orthogonal by construction.

        Returns:
            DataFrame indexed by field with columns: total_variance,
            global_mean_variance, global_mean_fraction, eof_variance_retained,
            eof_fraction_of_total, eof_fraction_of_residual, residual_variance,
            unexplained_fraction, explained_fraction.
        """
        rows = []
        for field in self.fields:
            eof = self.eofs_individual[field]
            eof_var_total = float(eof.variance_total)
            eof_var_retained = float(eof.variance_retained)

            if field in self.separate_global_mean:
                gm_var = self.global_mean_variance[field]
                total_var = gm_var + eof_var_total
            else:
                gm_var = 0.0
                total_var = eof_var_total

            rows.append(
                {
                    "field": field,
                    "total_variance": total_var,
                    "global_mean_variance": gm_var,
                    "global_mean_fraction": gm_var / total_var,
                    "eof_variance_retained": eof_var_retained,
                    "eof_fraction_of_total": eof_var_retained / total_var,
                    "eof_fraction_of_residual": float(eof.variance_fraction_retained),
                    "residual_variance": eof_var_total - eof_var_retained,
                    "unexplained_fraction": (eof_var_total - eof_var_retained) / total_var,
                    "explained_fraction": (gm_var + eof_var_retained) / total_var,
                }
            )
        return pd.DataFrame(rows).set_index("field")

    def get_individual_mode(self, field: str, n: int) -> xr.DataArray:
        """
        Get the physical field corresponding to a unit load of the n-th individual EOF mode.

        Args:
            field: the field for which to get modes
            n: the mode rank

        Returns:
            The physical field corresponding to the n-th mode
        """
        n_eofs = self.eofs_individual[field].U.shape[1]
        if n >= n_eofs:
            raise ValueError(f"Only {n_eofs} for field {field}, but requested {n}")

        reduced_state = np.zeros((n_eofs, 1))
        reduced_state[n, :] = 1
        eof = (
            xr.DataArray(
                self.nan_masks[field].backward(
                    self.eofs_individual[field].project_backwards(reduced_state)
                    / self.lat_weights[field]
                )[:, 0],
                coords=dict(state=self.state_coords.sel(field=field)),
            )
            .unstack("state")
            .sel(field=field)
        )
        return eof
