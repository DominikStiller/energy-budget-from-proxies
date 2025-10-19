from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cache
from typing import TYPE_CHECKING

import cf_xarray as cfxr
import numpy as np
import pandas as pd
import xarray as xr
from xarray import DataArray

from lmrecon.kf import (
    InstantaneousSerialEnKF,
    TimeAveragingSerialEnKF,
    TimeStackingSerialEnKF,
    inflate_covariance,
)
from lmrecon.logger import get_logger, logging_disabled
from lmrecon.psm import PSM, PhysicalSpacePSM, ReducedSpacePSM
from lmrecon.stats import localize_gc5thorder
from lmrecon.time import (
    Season,
    YearAndSeason,
    add_season_coords,
    convert_decimal_year_to_tuple,
    convert_tuple_to_decimal_year,
    format_tuple_time,
    split_seasonality,
    use_decimal_year_time_coords,
    use_tuple_time_coords,
)
from lmrecon.util import (
    get_spherical_distance,
    get_state_index,
    stack_state,
    to_math_order,
    unstack_state,
)

if TYPE_CHECKING:
    from pathlib import Path

    from cfr import ProxyDatabase
    from numpy.typing import ArrayLike

    from lmrecon.lim import LIM
    from lmrecon.mapper import PhysicalSpaceForecastSpaceMapper

logger = get_logger(__name__)


def create_initial_ensemble_from_perturbations(initial, n_ens, year_start, std=0.1):
    should_unstack = False
    if "state" not in initial.dims:
        initial = stack_state(initial)
        should_unstack = True

    rng = np.random.default_rng(562151)

    ens = []
    for i in range(n_ens):
        member = initial * rng.normal(1, std, size=initial.shape)
        member = member.expand_dims("ens").assign_coords(ens=[i])
        ens.append(member)

    ens = xr.concat(ens, "ens")
    if should_unstack:
        ens = unstack_state(ens)
    ens = ens.expand_dims("time").assign_coords(time=[year_start]).squeeze()

    return ens


def create_initial_ensemble_from_sample(
    ds_all, n_ens, year_start, season_start=Season.DJF, seed_offset=0
):
    ds_all = ds_all.sel(time=add_season_coords(ds_all).season == Season.DJF).drop("season")
    idx_sample = np.random.default_rng(682652 + seed_offset).choice(
        np.arange(len(ds_all.time)), n_ens, replace=False
    )
    ens = ds_all.isel(time=idx_sample)
    logger.info(
        "Drawing initial ensemble from years "
        + ", ".join(f"{v:.1f}" for v in sorted(ens.time.values))
    )
    ens = ens.rename(time="ens").assign_coords(ens=range(n_ens))
    ens = (
        ens.expand_dims("time")
        .assign_coords(time=[convert_tuple_to_decimal_year((year_start, season_start))])
        .squeeze()
    )
    return ens


class StateManager:
    """
    Manages the states at different times. The most recent ones are retained in memory since they are needed for DA,
    older states are written to disk.
    """

    def __init__(self, outdir: Path, prefix: str, max_inmemory_states: int, reduce_precision=True):
        """
        Constructor.

        Args:
            outdir: output directory
            prefix: sub-directory (e.g., prior or posterior)
            max_inmemory_states: number of states to keep in memory before writing to disk
            reduce_precision: write as float32 instead of float64
        """
        self.outdir = outdir
        self.prefix = prefix
        self.max_inmemory_states = max_inmemory_states
        self.reduce_precision = reduce_precision

        # Keys are (year, season), which allow chronological comparison
        self._states_inmemory: dict[YearAndSeason, DataArray] = {}
        self._states_disk: dict[YearAndSeason, Path] = {}

    def _write(self, time: YearAndSeason):
        state = self._states_inmemory[time]
        if self.reduce_precision:
            state = state.astype(np.float32)

        state = state.to_dataset(name="data")
        if isinstance(state.indexes["state"], pd.MultiIndex):
            state = cfxr.encode_multi_index_as_compress(state, "state")

        # Write to disk
        state_output_path = self.outdir / self.prefix / f"{time[0]:04}-{time[1].name}.nc"
        assert not state_output_path.exists()
        state.to_netcdf(state_output_path)
        self._states_disk[time] = state_output_path

        del self._states_inmemory[time]

    def has_time(self, time: float | YearAndSeason):
        if isinstance(time, float):
            time = convert_decimal_year_to_tuple(time)

        return time in self._states_inmemory.keys() | self._states_disk.keys()

    def insert(self, time: float | YearAndSeason, state: DataArray):
        if isinstance(time, float):
            time = convert_decimal_year_to_tuple(time)

        if self._states_inmemory and time < min(self._states_inmemory.keys()):
            raise ValueError(
                "Can only insert states that are newer than the earliest in-memory state"
            )

        self._states_inmemory[time] = state

        if len(self._states_inmemory) > self.max_inmemory_states:
            earliest_time = min(self._states_inmemory.keys())
            self._write(earliest_time)

    def get(self, time: float | YearAndSeason) -> DataArray:
        if isinstance(time, float):
            time = convert_decimal_year_to_tuple(time)

        if time in self._states_inmemory:
            return self._states_inmemory[time]
        elif time in self._states_disk:
            raise NotImplementedError("Re-loading of old states not supported")
        else:
            raise ValueError(f"Unknown time {time}")

    def flush(self):
        """Write all in-memory states to disk, e.g., when exiting program"""
        # Convert to list to create a copy of the keys since we're deleting from the dictionary that we're iterating
        for time in list(self._states_inmemory.keys()):
            self._write(time)
        assert len(self._states_inmemory) == 0


class DataAssimilation(ABC):
    def __init__(self, **kwargs):
        self.outdir: Path = kwargs["outdir"]
        self.pdb: ProxyDatabase = kwargs["pdb"]
        self.psms: dict[str, PSM | dict[Season, PSM]] = kwargs["psms"]
        self.mapper: PhysicalSpaceForecastSpaceMapper = kwargs["mapper"]
        self.inflation_factor: float | None = kwargs["inflation_factor"]

        self._state_coords = self.mapper.state_coords.indexes["state"]
        self._prior_state_manager = StateManager(self.outdir, "prior", 0)
        self._posterior_state_manager = StateManager(
            self.outdir, "posterior", self._required_history_length
        )

    @property
    @abstractmethod
    def _required_history_length(self):
        """Number of states before current time needed for DA, necessary for time averaging"""

    @cache
    def _get_psm_input_idx(self, psm: PSM) -> int:
        # Get state vector index corresponding to closest proxy location and proxy input field
        return get_state_index(
            self._state_coords,
            psm.field,
            psm.lat,
            psm.lon,
        )

    def cycle(self, prior: DataArray, year_start: int, year_end: int, tau: float):
        # First timestep is DJF of year_start, last timestep is SON of year_end
        times = np.arange(year_start + 0.5 / 12, year_end + 1, tau)
        prior = prior.compute()

        for time in times:
            logger.info(f"===== {format_tuple_time(time)} =====")

            # Check that forecast is in sync with DA
            assert convert_decimal_year_to_tuple(time) == convert_decimal_year_to_tuple(
                prior.time.item()
            )

            self._prior_state_manager.insert(time, prior)

            # select required posteriors from previous times and current prior
            priors = to_math_order(
                xr.concat(
                    [
                        self._posterior_state_manager.get(t)
                        for t in self._get_times_for_history(time, tau)
                    ]
                    + [prior],
                    dim="time",
                )
            )
            posteriors, prior = self._assimilate_and_forecast(time, priors)
            if self.inflation_factor is not None:
                prior[:] = inflate_covariance(prior.data, self.inflation_factor)

            if posteriors.time.ndim == 0:
                # Single posterior
                self._posterior_state_manager.insert(posteriors.time.item(), posteriors)
            else:
                for posterior_time in posteriors.time:
                    # Update past posteriors too
                    self._posterior_state_manager.insert(
                        posterior_time.item(), posteriors.sel(time=posterior_time)
                    )

        self._prior_state_manager.flush()
        self._posterior_state_manager.flush()

    def _get_times_for_history(self, time: float, tau: float) -> list[YearAndSeason]:
        # Return _required_history_length - 1 since the current prior is also added to pist
        return [
            convert_decimal_year_to_tuple(t)
            for t in np.arange(time - (self._required_history_length - 1) * tau, time, tau)
            if self._posterior_state_manager.has_time(t)
        ]

    @abstractmethod
    def _assimilate_and_forecast(
        self, current_time: float, prior: DataArray
    ) -> tuple[DataArray, DataArray]:
        """
        Assimilate current proxies and possibly forecast. The prior can contain multiple times if time-averaging DA
        is used, e.g., to assimilate seasonally sensitive proxies.

        Args:
            current_time: current time
            prior: prior state(s)

        Returns:
            Tuple of posterior state(s) and next prior.
        """


class PhysicalSpaceDataAssimilation(DataAssimilation, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loc_rad = kwargs["loc_rad"]

    @cache
    def _get_physical_psm(self, pid: str, season: Season) -> PhysicalSpacePSM:
        # Wrap PSMs in PhysicalSpacePSM
        psm_or_dict = self.psms[pid]
        if isinstance(psm_or_dict, dict):
            psm = psm_or_dict[season]
        else:
            psm = psm_or_dict
        return PhysicalSpacePSM(psm, self._get_psm_input_idx(psm))

    @cache
    def _get_localization_matrix(self, pid: str) -> ArrayLike | None:
        if self.loc_rad is None:
            return None

        # Could also be based on exact proxy location instead of nearest grid point (but may lead to asymmetric results)
        lat_base, lon_base = self._state_coords[self._get_psm_input_idx(pid)][1:]
        return localize_gc5thorder(
            get_spherical_distance(
                # Obs location
                lat_base,
                lon_base,
                # State locations
                self._state_coords.get_level_values(1),
                self._state_coords.get_level_values(2),
            ),
            self.loc_rad,
        )[:, np.newaxis]


class ReducedSpaceDataAssimilation(DataAssimilation, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @cache
    def _get_reduced_psm(self, pid: str, season: Season) -> ReducedSpacePSM:
        # Wrap PSMs in ReducedSpacePSM
        psm_or_dict = self.psms[pid]
        if isinstance(psm_or_dict, dict):
            psm = psm_or_dict[season]
        else:
            psm = psm_or_dict
        return ReducedSpacePSM(psm, self._get_psm_input_idx(psm), self.mapper)


class InstantaneousDataAssimilation(DataAssimilation, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def _required_history_length(self):
        return 0

    def _get_proxies_to_assimilate(self, current_time: float) -> dict[str, float]:
        records = {}

        # Select proxies, 3-month window centered at time to select obs for season
        for pid, pobj in self.pdb.slice(
            [current_time - 1.5 / 12, current_time + 1.49 / 12]
        ).records.items():
            if len(pobj.value) == 0:
                # No data for this proxy at current timestep
                continue

            if len(pobj.value) > 1:
                # Will possibly need to take mean if there are multiple proxy values per assimilation time step
                # Need to adjust obs error accordingly (https://github.com/fzhu2e/cfr/blob/main/cfr/da/enkf.py#L170)
                raise NotImplementedError()

            obs_i = pobj.value.item()
            if np.isfinite(obs_i):
                records[pid] = obs_i

        return records


class TimeWindowDataAssimilation(DataAssimilation, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        match kwargs.get("window_mode", "average"):
            case "average":
                self.kf = TimeAveragingSerialEnKF("EnSRF")
            case "stack":
                self.kf = TimeStackingSerialEnKF("EnSRF")

    @property
    def _required_history_length(self):
        # Maximum proxy seasonality contains four seasons
        return 4

    def _get_proxies_to_assimilate(
        self, current_time: float, priors: DataArray
    ) -> dict[str, tuple[float, list[YearAndSeason]]]:
        records = {}
        current_year, current_season = convert_decimal_year_to_tuple(current_time)
        prior_times = list(priors["time"].data)

        for pid, pobj in self.pdb.records.items():
            psm_or_dict = self.psms[pid]
            is_seasonally_resolved = isinstance(psm_or_dict, dict)

            if is_seasonally_resolved and current_season not in psm_or_dict:
                # Do not assimilate seasonally resolved proxy if there is no PSM for current season
                # since this means that the calibration was unsuccessful for this season.
                continue
            if not is_seasonally_resolved and psm_or_dict.seasonality[-1] != current_season:
                # Do not assimilate annually resolved proxies whose seasonality does not end with
                # the current season
                continue

            if not is_seasonally_resolved:
                # Annually resolved
                seasons_previous_year, seasons_current_year = split_seasonality(
                    psm_or_dict.seasonality
                )
                seasonality_i = [(current_year - 1, s) for s in seasons_previous_year] + [
                    (current_year, s) for s in seasons_current_year
                ]
                pobj = pobj.slice((current_year - 0.5, current_year + 0.5))
            else:
                # Seasonally resolved
                seasonality_i = [(current_year, current_season)]
                pobj = pobj.slice((current_time - 1.5 / 12, current_time + 1.49 / 12))

            if not all(s in prior_times for s in seasonality_i):
                # Not all necessary seasons in prior, e.g., at start of DA
                logger.warning(
                    f"Not all necessary timesteps for seasonality of proxy {pid} in prior"
                )
                continue

            if len(pobj.value) == 0:
                # No data for this proxy at current timestep
                continue

            if len(pobj.value) > 1:
                # Will possibly need to take mean if there are multiple proxy values per assimilation time step
                # Need to adjust obs error accordingly (https://github.com/fzhu2e/cfr/blob/main/cfr/da/enkf.py#L170)
                raise NotImplementedError()

            obs_i = pobj.value.item()
            if np.isfinite(obs_i):
                records[pid] = (obs_i, seasonality_i)

        return records


class LIMandInstantaneousEnSRFinPhysicalSpaceDataAssimilation(
    PhysicalSpaceDataAssimilation, InstantaneousDataAssimilation
):
    """
    Data assimilation scheme with LIM for forecasting and EnSRF for assimilation. Observations are treated as
    instantaneous. The state is updated in physical space, which enables localization.
    """

    def __init__(
        self,
        outdir: Path,
        pdb: ProxyDatabase,
        psms: dict[str, PSM | dict[Season, PSM]],
        mapper: PhysicalSpaceForecastSpaceMapper,
        loc_rad: float,
        lim: LIM,
        inflation_factor: float | None = None,
    ):
        super().__init__(
            outdir=outdir,
            pdb=pdb,
            psms=psms,
            mapper=mapper,
            loc_rad=loc_rad,
            inflation_factor=inflation_factor,
        )
        self.lim = lim
        self.n_int_steps_per_tau = int(1440 * lim.tau)

        self.kf = InstantaneousSerialEnKF("EnSRF")

        # This DA scheme only supports seasonally resolved proxies
        assert np.all([0.24 < p.dt < 0.26 for p in self.pdb.records.values()])

    def _assimilate_and_forecast(
        self, current_time: float, prior: DataArray
    ) -> tuple[DataArray, DataArray]:
        # Squeeze time dimension
        prior = prior.sel(time=current_time)
        _, current_season = convert_decimal_year_to_tuple(current_time)

        proxies_to_assimilate = self._get_proxies_to_assimilate(current_time)
        obs = list(proxies_to_assimilate.values())
        psms = [self._get_physical_psm(pid, current_season) for pid in proxies_to_assimilate]
        loc_matrices = [self._get_localization_matrix(pid) for pid in proxies_to_assimilate]

        logger.info("Assimilating observations")
        posterior = self.kf.assimilate(prior, obs, psms, loc_matrices)

        with logging_disabled():
            initial_condition = self.mapper.forward(posterior)

        logger.info("Forecasting in reduced space")
        forecast = self.lim.forecast_stochastic(
            initial_condition, 1, self.n_int_steps_per_tau
        ).isel(time=-1)

        with logging_disabled():
            next_prior = self.mapper.backward(forecast)

        return posterior, next_prior


class LIMandTimeWindowEnSRFinPhysicalSpaceDataAssimilation(
    PhysicalSpaceDataAssimilation, TimeWindowDataAssimilation
):
    """
    Data assimilation scheme with LIM for forecasting and EnSRF for assimilation. Observations are treated as
    time averages. The state is updated in physical space, which enables localization.
    """

    def __init__(
        self,
        outdir: Path,
        pdb: ProxyDatabase,
        psms: dict[str, PSM | dict[Season, PSM]],
        mapper: PhysicalSpaceForecastSpaceMapper,
        lim: LIM,
        loc_rad: float,
        window_mode: str = "average",
        inflation_factor: float | None = None,
    ):
        super().__init__(
            outdir=outdir,
            pdb=pdb,
            psms=psms,
            mapper=mapper,
            loc_rad=loc_rad,
            inflation_factor=inflation_factor,
            window_mode=window_mode,
        )
        self.lim = lim
        self.n_int_steps_per_tau = int(1440 * lim.tau)

        # This DA scheme only supports seasonally or annually resolved proxies
        assert np.all([0.24 < p.dt < 0.26 or 0.99 < p.dt < 1.01 for p in self.pdb.records.values()])

    def _assimilate_and_forecast(
        self, current_time: float, priors: DataArray
    ) -> tuple[DataArray, DataArray]:
        priors = use_tuple_time_coords(priors)
        _, current_season = convert_decimal_year_to_tuple(current_time)

        proxies_to_assimilate = self._get_proxies_to_assimilate(current_time, priors)
        obs = [p[0] for p in proxies_to_assimilate.values()]
        psms = [self._get_physical_psm(pid, current_season) for pid in proxies_to_assimilate]
        seasonalities = [p[1] for p in proxies_to_assimilate.values()]
        loc_matrices = [self._get_localization_matrix(pid) for pid in proxies_to_assimilate]

        logger.info("Assimilating observations")
        posteriors = self.kf.assimilate(priors, obs, psms, seasonalities, loc_matrices)

        with logging_disabled():
            # Only forecast posterior of latest timestep
            initial_condition = self.mapper.forward(
                use_decimal_year_time_coords(
                    posteriors.sel(time=convert_decimal_year_to_tuple(current_time))
                )
            ).compute()

        logger.info("Forecasting in reduced space")
        forecast = (
            self.lim.forecast_stochastic(initial_condition, 1, self.n_int_steps_per_tau)
            .isel(time=-1)
            .compute()
        )

        with logging_disabled():
            next_prior = self.mapper.backward(forecast)

        posteriors = use_decimal_year_time_coords(posteriors)
        return posteriors, next_prior


class LIMandTimeWindowEnSRFinReducedSpaceDataAssimilation(
    ReducedSpaceDataAssimilation, TimeWindowDataAssimilation
):
    """
    Data assimilation scheme with LIM for forecasting and EnSRF for assimilation. Observations are treated as
    time averages. The state is updated in reduced space.
    """

    def __init__(
        self,
        outdir: Path,
        pdb: ProxyDatabase,
        psms: dict[str, PSM | dict[Season, PSM]],
        mapper: PhysicalSpaceForecastSpaceMapper,
        lim: LIM,
        window_mode: str = "average",
        inflation_factor: float | None = None,
    ):
        super().__init__(
            outdir=outdir,
            pdb=pdb,
            psms=psms,
            mapper=mapper,
            inflation_factor=inflation_factor,
            window_mode=window_mode,
        )
        self.lim = lim
        self.n_int_steps_per_tau = int(1440 * lim.tau)  # 1440 steps per year -> ~6 h steps

        # This DA scheme only supports seasonally or annually resolved proxies
        # assert np.all(
        #     list(0.24 < p.dt < 0.26 or 0.99 < p.dt < 1.01 for p in self.pdb.records.values())
        # )

    def _assimilate_and_forecast(
        self, current_time: float, priors: DataArray
    ) -> tuple[DataArray, DataArray]:
        priors = use_tuple_time_coords(priors)
        _, current_season = convert_decimal_year_to_tuple(current_time)

        proxies_to_assimilate = self._get_proxies_to_assimilate(current_time, priors)
        obs = [p[0] for p in proxies_to_assimilate.values()]
        psms = [self._get_reduced_psm(pid, current_season) for pid in proxies_to_assimilate]
        seasonalities = [p[1] for p in proxies_to_assimilate.values()]

        logger.info("Assimilating observations")
        posteriors = self.kf.assimilate(priors, obs, psms, seasonalities)

        # Only forecast posterior of latest timestep
        initial_condition = use_decimal_year_time_coords(
            posteriors.sel(time=convert_decimal_year_to_tuple(current_time))
        )

        logger.info("Forecasting in reduced space")
        next_prior = self.lim.forecast_stochastic(
            initial_condition, 1, self.n_int_steps_per_tau
        ).isel(time=-1)

        posteriors = use_decimal_year_time_coords(posteriors)
        return posteriors, next_prior
