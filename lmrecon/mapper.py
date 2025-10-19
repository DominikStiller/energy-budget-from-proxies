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
from lmrecon.stats import area_weighted_mean, compute_field_stds, compute_field_stds_by_season
from lmrecon.time import Season, map_season_to_decimal, use_tuple_time_coords
from lmrecon.util import (
    NanMask,
    has_float_timedim,
    list_complement,
    stack_state,
    to_math_order,
    unstack_state,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

logger = get_logger(__name__)


# Do not separate global mean for SIC since it is very localized; otherwise, the EOFs would need
# to offset the global mean in ice-free regions (which is most of the globe)
OMIT_FROM_SEPARATE_GLOBAL_MEAN = ["siconc", "siconcn", "siconcs"]


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
        missing_variables = set(ds_full.keys()) - set(ds.keys())
        if missing_variables and self.not_direct_fields:
            logger.warning(
                f"Missing variables ({', '.join(missing_variables)}) may lead to reduced variance for joint EOFs"
            )
        for field in set(ds.keys()) & set(ds_full.keys()):
            ds_full[field] = ds[field]
        da_full = to_math_order(stack_state(ds_full))
        # Use all here since e.g. instrumental datasets can have time-varying nans
        # This is in contrast to nan mask for EOFs, which cannot handle any nans (in class NanMask)
        nan_mask = np.array(np.isnan(da_full).all(axis=1))[:, np.newaxis]

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
        k: int,
        l: int,
        k_direct: dict[str, int] | None = None,
        standardize_by_season: bool = False,
        separate_global_mean: bool = False,
    ):
        """
        Create the mapper.

        Forward and backward mapping is a linear operation. Therefore, the .backward_matrix can be
        used for efficient backward mapping.

        Args:
            k: EOFs to retain in first step
            l: EOFs to retain in second step
            k_direct: EOFs to retain for fields directly appended to state, keys are field names
            direct_fields: fields to directly append to state instead of including them in joint EOF
                PH20 does this for OHC700m
            standardize_by_season: standardize variance by season or all seasons using same factor
            separate_global_mean: separate global mean from EOFs around that mean
        """
        super().__init__()

        self.k = k
        self.l = l
        self.k_direct = k_direct or {}
        self.direct_fields = list(self.k_direct.keys())
        self.standardize_by_season = standardize_by_season
        self.separate_global_mean = separate_global_mean

        self.fields: list[str] = None
        self.nan_masks: dict[str, NanMask] = {}
        self.eofs_individual: dict[str, EOF] = {}
        self.standard_deviations: xr.Dataset = None
        self.standard_deviations_global_mean: xr.Dataset = None
        self.eof_joint: EOF = EOF(self.l)
        self.lats: dict[str, ArrayLike] = {}
        self.lat_weights: dict[str, ArrayLike] = {}
        self.state_coords: xr.DataArray = None
        self.not_direct_fields: list[str] = None

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        outfile = directory / "mapper.pkl"
        logger.info(f"Saving mapper to {outfile}")
        pickle.dump(self, outfile.open("wb"))

    @classmethod
    def load(cls, file: Path | str) -> Self:
        return pickle.load(Path(file).open("rb"))

    def _validate_input(self, data: xr.DataArray):
        super()._validate_input(data)
        if self.standardize_by_season and (data.dims[1] != "time" or not has_float_timedim(data)):
            raise ValueError(
                "If standardizing by season, second dimension must be float timestamps"
            )

    def fit_and_forward(self, data: xr.DataArray) -> xr.DataArray:
        self.fit(data)
        return self.forward(data)

    def _calculate_field_variances(self, data: xr.DataArray):
        data = unstack_state(data)
        # Doing some duplicate computation by calculating the global mean here and in _fit, but
        # keeps the code cleaner this way
        global_mean = area_weighted_mean(data).compute()
        for field in set(OMIT_FROM_SEPARATE_GLOBAL_MEAN) & set(self.fields):
            global_mean[field] = xr.zeros_like(global_mean[field])

        if self.separate_global_mean:
            if self.standardize_by_season:
                self.standard_deviations_global_mean = (
                    use_tuple_time_coords(global_mean).groupby("season").std()
                )
                self.standard_deviations = compute_field_stds_by_season(data - global_mean)
            else:
                self.standard_deviations_global_mean = global_mean.std()
                self.standard_deviations = compute_field_stds(data - global_mean)
        else:
            if self.standardize_by_season:
                self.standard_deviations = compute_field_stds_by_season(data)
            else:
                self.standard_deviations = compute_field_stds(data)

    def fit(self, data: xr.DataArray) -> None:
        logger.info("PhysicalSpaceForecastSpaceMapper.fit()")
        self._validate_input(data)
        if "field" not in data.coords:
            raise ValueError("Physical state vector must have field coordinate")

        self.state_coords = data.state
        self.fields = pd.unique(np.array(data.field))
        self.not_direct_fields = list_complement(self.fields, self.direct_fields)

        logger.info("Calculating field variances")
        self._calculate_field_variances(data)

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

        data_eof_individual: dict[str, ArrayLike] = {}

        if self.standardize_by_season:
            seasons = use_tuple_time_coords(data)["season"]

        for i, field in enumerate(self.fields):
            data_field = data_nonan[field]

            if self.separate_global_mean and field not in OMIT_FROM_SEPARATE_GLOBAL_MEAN:
                # Area-weight by weight^2 for global mean since lat_weights has the sqrt for covariance
                # weighting
                logger.info(f"Separating global mean for {field}")
                data_global_mean = (data_nonan[field] * self.lat_weights[field] ** 2).sum(
                    axis=0, keepdims=True
                ).compute() / np.sum(self.lat_weights[field] ** 2)
                # Avoid in-place assignment when working with Dask arrays since it can lead to subtle bugs (https://github.com/dask/dask/issues/11607)
                data_field = data_field - data_global_mean
                if self.standardize_by_season:
                    data_global_mean = (
                        data_global_mean
                        / self.standard_deviations_global_mean[field].sel(season=seasons).data
                    )
                else:
                    data_global_mean = (
                        data_global_mean / self.standard_deviations_global_mean[field].data
                    )

            data_field = data_field * self.lat_weights[field]
            if self.standardize_by_season:
                data_field = data_field / self.standard_deviations[field].sel(season=seasons).data
            else:
                # Theoretically do not need to standardize before individual EOFs if not doing it by
                # season, but still do it for consistency
                data_field = data_field / self.standard_deviations[field].data

            self.eofs_individual[field] = EOF(
                self.k_direct[field] if field in self.direct_fields else self.k
            )

            logger.info(f"Fitting EOF for {field} [{i + 1}/{len(self.fields)}]")
            self.eofs_individual[field].fit(data_field)
            logger.info(f"Projecting EOF for {field}")
            data_eof_individual[field] = self.eofs_individual[field].project_forwards(data_field)

            # Standardize by retained variance, which is only relevant if doing joint EOFs
            data_eof_individual[field] = data_eof_individual[field] / np.sqrt(
                self.eofs_individual[field].variance_retained
            )

            if self.separate_global_mean and field not in OMIT_FROM_SEPARATE_GLOBAL_MEAN:
                # Stacking with the global mean here means that the global mean will be part
                # of the joint EOFs, maybe this is not a good idea?
                data_eof_individual[field] = np.vstack(
                    [data_global_mean, data_eof_individual[field]]
                )

        if len(self.not_direct_fields) > 0:
            data_stacked_for_joint_eof = dask.array.vstack(
                [data_eof_individual[field] for field in self.not_direct_fields]
            )

            logger.info(f"Fitting joint EOF for {', '.join(self.not_direct_fields)}")
            self.eof_joint.fit(data_stacked_for_joint_eof)

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

        if self.standardize_by_season:
            seasons = use_tuple_time_coords(data)["season"]

        for i, field in enumerate(self.fields):
            data_field = data_nonan[field]

            if self.separate_global_mean and field not in OMIT_FROM_SEPARATE_GLOBAL_MEAN:
                logger.info(f"Separating global mean for {field}")
                data_global_mean = np.array(
                    (data_nonan[field] * self.lat_weights[field] ** 2).sum(axis=0, keepdims=True)
                ) / np.sum(self.lat_weights[field] ** 2)
                data_field = data_field - data_global_mean
                if self.standardize_by_season:
                    data_global_mean = (
                        data_global_mean
                        / self.standard_deviations_global_mean[field].sel(season=seasons).data
                    )
                else:
                    data_global_mean = (
                        data_global_mean / self.standard_deviations_global_mean[field].data
                    )

            data_field = data_field * self.lat_weights[field]
            if self.standardize_by_season:
                data_field = data_field / self.standard_deviations[field].sel(season=seasons).data
            else:
                data_field = data_field / self.standard_deviations[field].data

            logger.info(f"Projecting EOF for {field} [{i + 1}/{len(self.fields)}]")
            data_eof_individual[field] = self.eofs_individual[field].project_forwards(data_field)

            data_eof_individual[field] = data_eof_individual[field] / np.sqrt(
                self.eofs_individual[field].variance_retained
            )
            if self.separate_global_mean and field not in OMIT_FROM_SEPARATE_GLOBAL_MEAN:
                data_eof_individual[field] = np.vstack(
                    [data_global_mean, data_eof_individual[field]]
                )

        if len(self.not_direct_fields) > 0:
            logger.info(f"Projecting joint EOF for {', '.join(self.not_direct_fields)}")
            data_stacked_for_joint_eof = dask.array.vstack(
                [data_eof_individual[field] for field in self.not_direct_fields]
            )
            data_eof_joint = self.eof_joint.project_forwards(data_stacked_for_joint_eof)

            logger.info(f"Appending direct fields for {', '.join(self.direct_fields)}")
            data_eof_joint_and_direct = dask.array.vstack(
                [data_eof_joint] + [data_eof_individual[field] for field in self.direct_fields]
            ).rechunk()
        else:
            logger.info(f"Stacking direct fields for {', '.join(self.direct_fields)}")
            data_eof_joint_and_direct = dask.array.vstack(
                [data_eof_individual[field] for field in self.direct_fields]
            ).rechunk()

        if not has_sampling_dimension:
            # Remove dummy dimension
            data_eof_joint_and_direct = np.squeeze(data_eof_joint_and_direct, axis=1)

        new_coords = data.drop_vars("state").coords.copy()
        new_coords["state"] = range(data_eof_joint_and_direct.shape[0])
        return xr.DataArray(data_eof_joint_and_direct, dims=data.dims, coords=new_coords, name=None)

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
            if self.standardize_by_season:
                data_array_physical = self._backward_stepwise(
                    data_array, use_tuple_time_coords(data)["season"]
                )
            else:
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

        if not force_stepwise and self.standardize_by_season:
            # Adjust seasonal variances (matrix-based backward mapping uses DJF)
            seasons = use_tuple_time_coords(data)["season"]
            for field in self.fields:
                std_factors = (
                    self.standard_deviations[field].sel(season=seasons).data
                    / self.standard_deviations[field].sel(season=Season.DJF).item()
                )
                data_physical.loc[dict(field=field)] = (
                    data_physical.loc[dict(field=field)].data * std_factors
                )

        return data_physical

    def _backward_stepwise(self, data: ArrayLike, seasons: list[Season] | None = None) -> ArrayLike:
        """
        Space mapping is more easily expressed as a sequence of steps. However, this approach is slower than a direct
        matrix multiplication. Therefore, this method is only used to derive the corresponding mapping matrix.
        """
        data_eof_individual: dict[str, ArrayLike] = {}
        global_means: dict[str, ArrayLike] = {}

        logger.info(f"Splitting direct fields for {', '.join(self.direct_fields)}")
        start_row = self.eof_joint.rank
        for field in self.direct_fields:
            if self.separate_global_mean and field not in OMIT_FROM_SEPARATE_GLOBAL_MEAN:
                global_means[field] = data[[start_row]]
                start_row += 1
            length = self.eofs_individual[field].rank
            data_eof_individual[field] = data[start_row : start_row + length]
            start_row += length

        if len(self.not_direct_fields) > 0:
            logger.info(f"Back-projecting joint EOF for {', '.join(self.not_direct_fields)}")
            data_eof_joint = data[: self.eof_joint.rank]
            data_stacked_for_joint_eof = self.eof_joint.project_backwards(data_eof_joint)

            start_row = 0
            for field in self.not_direct_fields:
                if self.separate_global_mean and field not in OMIT_FROM_SEPARATE_GLOBAL_MEAN:
                    global_means[field] = data[[start_row]]
                    start_row += 1
                length = self.eofs_individual[field].rank
                data_eof_individual[field] = data_stacked_for_joint_eof[
                    start_row : start_row + length
                ]
                start_row += length

        data_nonan: dict[str, ArrayLike] = {}

        for i, field in enumerate(self.fields):
            data_field = data_eof_individual[field]

            data_field = data_field * np.sqrt(self.eofs_individual[field].variance_retained)

            logger.info(f"Back-projecting EOF for {field} [{i + 1}/{len(self.fields)}]")
            data_nonan[field] = self.eofs_individual[field].project_backwards(data_field)
            if data_nonan[field].ndim == 1:
                data_nonan[field] = data_nonan[field][:, np.newaxis]

            if self.standardize_by_season:
                data_nonan[field] = (
                    data_nonan[field] * self.standard_deviations[field].sel(season=seasons).data
                )
            else:
                data_nonan[field] = data_nonan[field] * self.standard_deviations[field].data
            data_nonan[field] = data_nonan[field] / self.lat_weights[field]

            if self.separate_global_mean and field not in OMIT_FROM_SEPARATE_GLOBAL_MEAN:
                data_global_mean = global_means[field]
                if self.standardize_by_season:
                    data_global_mean = data_global_mean * (
                        self.standard_deviations_global_mean[field].sel(season=seasons).data
                    )
                else:
                    data_global_mean = (
                        data_global_mean * self.standard_deviations_global_mean[field].data
                    )
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

        If standardization is done by season, the backward matrix will be for DJF and results need to be scaled
        appropriately.

        Returns:
            Backward matrix
        """
        with logging_disabled():
            return self._backward_stepwise(
                dask.array.eye(self.n_reduced_state), [Season.DJF] * self.n_reduced_state
            ).compute()

    @cached_property
    def n_reduced_state(self) -> int:
        """
        Length of reduced state.
        """
        n_reduced_state = self.l + sum(self.k_direct.values())
        if self.separate_global_mean:
            n_reduced_state += len(self.k_direct) - len(
                set(OMIT_FROM_SEPARATE_GLOBAL_MEAN) & set(self.fields)
            )
        return n_reduced_state

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

    def get_joint_mode(self, n: int) -> xr.Dataset:
        """
        Get the physical field corresponding to a unit load of the n-th joint EOF mode, mapped to all fields.

        Args:
            n: the mode rank

        Returns:
            The physical fields corresponding to the n-th mode
        """
        n_eofs = self.l + sum(self.k_direct.values())
        if self.standardize_by_season:
            # Get modes for all four seasons
            reduced_state = np.zeros((n_eofs, 4))
            reduced_state[n, :] = 1
            eof = unstack_state(
                self.backward(
                    xr.DataArray(
                        reduced_state,
                        dims=["state", "time"],
                        coords=dict(
                            state=range(n_eofs),
                            time=list(map(map_season_to_decimal, Season.ANNUAL)),
                        ),
                    ),
                    force_stepwise=True,
                )
            ).rename(time="season")
        else:
            reduced_state = np.zeros(n_eofs)
            reduced_state[n] = 1
            eof = unstack_state(
                self.backward(xr.DataArray(reduced_state, coords=dict(state=range(n_eofs))))
            )
        return eof
