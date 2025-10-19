from __future__ import annotations

from enum import Enum
from functools import total_ordering
from typing import TYPE_CHECKING, ClassVar, Self

import cftime
import numpy as np
import xarray as xr
from cfr.utils import datetime2year_float

from lmrecon.util import (
    has_cftime_timedim,
    has_float_timedim,
    has_int_timedim,
    has_npdatetime_timedim,
    has_tuple_timedim,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from xarray.core.types import T_Xarray

MONTH_ABBREVIATIONS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

MONTH_LENGTHS = xr.DataArray(
    [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    dims=["month"],
    coords=dict(month=np.arange(1, 13)),
)


@total_ordering
class Season(Enum):
    """
    Enum for meteorological seasons. They are comparable according to their int value, i.e., DJF < MAM.
    """

    DJF = 0
    MAM = 1
    JJA = 2
    SON = 3
    ANNUAL: ClassVar[list[int]] = [DJF, MAM, JJA, SON]

    def __lt__(self, other):
        # From https://stackoverflow.com/a/39269589
        # Implement comparison ourselves instead of using IntEnum since they would be automatically converted to int
        # when used as xarray index
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplementedError()

    @classmethod
    def to_str_list(cls, l: list[Self]) -> list[str]:
        return [s.name for s in l]

    @classmethod
    def from_str_list(cls, l: list[str]) -> list[Self]:
        return [Season[s] for s in l]


type YearAndSeason = tuple[int, Season]


def month_name(time: xr.DataArray):
    return [MONTH_ABBREVIATIONS[n - 1] for n in time.dt.month.values]


def map_season_to_decimal(season: Season) -> float:
    """
    Map a season label to a decimal number. Assign middle of season as timestamp.

    Args:
        season: DJF, MAM, JJA or SON

    Returns:
        decimal representation of season
    """
    return {
        Season.DJF: 0.5 / 12,
        Season.MAM: 3.5 / 12,
        Season.JJA: 6.5 / 12,
        Season.SON: 9.5 / 12,
    }[season]


def map_decimal_to_season(time: float) -> Season:
    """
    Map the decimal representation of a season to its label.

    Args:
        time: decimal representation of season

    Returns:
        DJF, MAM, JJA or SON
    """
    decimal = (time % 1) * 12
    if -1 < decimal < 2 or 11 < decimal < 12:
        return Season.DJF
    elif 2 < decimal < 5:
        return Season.MAM
    elif 5 < decimal < 8:
        return Season.JJA
    elif 8 < decimal < 11:
        return Season.SON
    else:
        raise ValueError(f"Invalid decimal representation of season: {decimal:f}")


def round_to_nearest_season(time: ArrayLike) -> ArrayLike:
    """
    Round decimal timestamps to the nearest season. Assigns the middle of the season as timestamp.

    Args:
        time: decimal timestamps

    Returns:
        Decimal timestamps of nearest season
    """
    year = time.astype(int)
    month = (time - year) * 12
    # Assign Dec to DJF of next year
    year = np.where(11 <= month, year + 1, year)

    # Assign middle of season
    month[((-1 < month) & (month < 2)) | ((11 <= month) & (month < 12))] = 0.5
    month[(2 <= month) & (month < 5)] = 3.5
    month[(5 <= month) & (month < 8)] = 6.5
    month[(8 <= month) & (month < 11)] = 9.5

    return year + month / 12


def map_decimal_to_month(time: float) -> int:
    """
    Map the decimal representation of a month to its number.

    Args:
        time: decimal representation of number

    Returns:
        number of month
    """
    decimal = (time % 1) * 12
    return 1 + int(np.round(decimal))


def convert_decimal_year_to_tuple(time: float) -> YearAndSeason:
    return int(time), map_decimal_to_season(time)


def convert_tuple_to_decimal_year(time: YearAndSeason) -> float:
    if isinstance(time, float):
        return time
    return time[0] + map_season_to_decimal(time[1])


def convert_decimal_year_to_datetime(decimal_year: np.array, leap_years=False) -> np.array:
    if not leap_years:
        days_since_epoch = (np.array(decimal_year) - 1) * 365

        return cftime.num2date(
            days_since_epoch,
            "days since 0001-01-01",
            calendar="noleap",
        )
    else:
        year = np.array(decimal_year).astype(int)
        year_fraction = decimal_year - year
        max_year = np.max(year)
        first_day_of_year = np.array(
            np.arange(0, max_year + 2).astype(str), dtype="datetime64[D]"
        )  # indexed by year
        year_length_in_days = (first_day_of_year[1:] - first_day_of_year[:-1]).astype(int)
        days_since_epoch = np.zeros(max_year + 1)  # indexed by year
        days_since_epoch[2:] = np.cumsum(year_length_in_days[1:-1])  # epoch being 0001-01-01

        return cftime.num2date(
            days_since_epoch[year] + year_length_in_days[year] * year_fraction,
            "days since 0001-01-01",
            calendar="proleptic_gregorian",
        )


def convert_datetime_to_decimal_year(da: xr.DataArray, equal_month_lengths=False) -> np.array:
    if equal_month_lengths:
        return da.dt.year.data + (da.dt.month.data - 0.5) / 12
    else:
        return datetime2year_float(da.data, resolution="day")


def use_tuple_time_coords(da: T_Xarray, force_seasonal=False) -> T_Xarray:
    if "year" in da.coords and "season" in da.coords:
        return da
    if has_float_timedim(da):
        if is_seasonal_resolution(da) or force_seasonal:
            return da.assign_coords(
                year=("time", da.time.data.astype(int)),
                season=("time", list(map(map_decimal_to_season, da.time))),
            ).set_xindex(["year", "season"])
    elif has_npdatetime_timedim(da) or has_cftime_timedim(da):
        if is_seasonal_resolution(da) or force_seasonal:
            return use_tuple_time_coords(
                use_decimal_year_time_coords(da), force_seasonal=force_seasonal
            )
        elif is_monthly_resolution(da):
            return da.assign_coords(
                year=("time", da.time.dt.year.data),
                month=("time", da.time.dt.month.data),
            ).set_xindex(["year", "month"])

    print(da.time)
    raise NotImplementedError()


def use_decimal_year_time_coords(da: T_Xarray, equal_month_lengths=False) -> T_Xarray:
    if "time" in da.coords and (has_float_timedim(da) or has_int_timedim(da)):
        return da
    if has_tuple_timedim(da):
        if is_seasonal_resolution(da):
            if da.year.ndim == 0 and da.season.ndim == 0:
                time = da.year.data.item() + map_season_to_decimal(da.season.data.item())
                return da.assign_coords(time=time).drop_vars(["year", "season"])
            elif da.year.ndim == 0:
                time = da.year.data.item() + list(map(map_season_to_decimal, da.season.data))
                return da.assign_coords(time=time).drop_vars(["year"])
            elif da.season.ndim == 0:
                time = da.year.data + map_season_to_decimal(da.season.data.item())
                return da.assign_coords(time=time).drop_vars(["season"])
            else:
                time = da.year.data + list(map(map_season_to_decimal, da.season.data))
                return da.assign_coords(time=("time", time))
        elif is_monthly_resolution(da):
            if equal_month_lengths:
                return da.assign_coords(time=(da.year + (da.month - 0.5) / 12).data)
            else:
                raise NotImplementedError()
    elif has_cftime_timedim(da) or has_npdatetime_timedim(da):
        time = convert_datetime_to_decimal_year(da.time, equal_month_lengths=equal_month_lengths)
        return da.assign_coords(time=time)

    print(da.time)
    raise NotImplementedError()


def add_season_coords(da: T_Xarray) -> T_Xarray:
    if "season" in da.coords:
        return da
    return da.assign_coords(
        season=("time", list(map(map_decimal_to_season, da.time))),
    )


def add_month_coords(da: T_Xarray) -> T_Xarray:
    if "month" in da.coords:
        return da
    return da.assign_coords(
        month=("time", list(map(map_decimal_to_month, da.time))),
    )


def use_datetime_time_coords(da: T_Xarray) -> T_Xarray:
    if "time" in da.coords and (has_cftime_timedim(da) or has_npdatetime_timedim(da)):
        return da
    if has_tuple_timedim(da):
        da = da.assign_coords(time=convert_tuple_to_decimal_year(da.time))
    return da.assign_coords(time=convert_decimal_year_to_datetime(da.time))


def use_string_season_coords(da: T_Xarray) -> T_Xarray:
    return da.assign_coords(season=Season.to_str_list(da["season"].to_numpy()))


def use_enum_season_coords(da: T_Xarray) -> T_Xarray:
    return da.assign_coords(season=Season.from_str_list(da.season.values))


def use_monthly_npdatetime_time_coords(da: T_Xarray) -> T_Xarray:
    return da.assign_coords(time=da["time"].astype("datetime64[M]"))


def is_annual_resolution(da: T_Xarray) -> bool:
    if has_tuple_timedim(da):
        return False
    elif has_npdatetime_timedim(da) or has_cftime_timedim(da):
        # Assume that datetimes are monthly, not annual
        return False
    elif has_int_timedim(da):
        return True
    elif has_float_timedim(da):
        dt = np.median(np.diff(da.time))
        return np.isclose(dt, 1)
    else:
        raise NotImplementedError()


def is_seasonal_resolution(da: T_Xarray) -> bool:
    if has_tuple_timedim(da):
        return isinstance(da["time"].values.flat[0][1], Season)
    elif has_int_timedim(da):
        return False
    elif has_npdatetime_timedim(da) or has_cftime_timedim(da):
        # Assume that datetimes are monthly, not seasonal
        return False
    elif has_float_timedim(da):
        if len(da.time) == 1:
            return True
        dt = np.median(np.diff(da.time))
        return np.isclose(dt, 1 / 4, rtol=0.05)
    else:
        raise NotImplementedError()


def is_monthly_resolution(da: T_Xarray) -> bool:
    if has_tuple_timedim(da):
        return isinstance(da["time"].values.flat[0][1], int)
    elif has_int_timedim(da):
        return False
    elif has_npdatetime_timedim(da) or has_cftime_timedim(da):
        # maybe a bad assumption
        return True
    elif has_float_timedim(da):
        dt = np.median(np.diff(use_decimal_year_time_coords(da).time))
        return np.isclose(dt, 1 / 12, rtol=0.05)
    else:
        raise NotImplementedError()


def format_tuple_time(time: float, year_first=True):
    if year_first:
        return f"{int(time)}-{map_decimal_to_season(time).name}"
    else:
        return f"{map_decimal_to_season(time).name} {int(time)}"


def split_seasonality(seasonality: list[Season]) -> tuple[list[Season], list[Season]]:
    """
    Splits a seasonality list into the seasons belonging to the previous and current year. The split occurs at the DJF.

    Args:
        seasonality: list of seasons

    Returns:
        Tuple with list of previous year's seasons and list of current year's seasons
    """
    if len(seasonality) != len(set(seasonality)):
        raise ValueError("Seasons must be unique, can either belong to previous or current year")

    # Check if seasonality crosses the year border
    if Season.DJF not in seasonality or seasonality.index(Season.DJF) == 0:
        # Seasonality is all within current year
        seasons_current_year = seasonality
        seasons_previous_year = []
    else:
        # Some seasons belong to previous year, some to current
        # Split at DJF
        djf_idx = seasonality.index(Season.DJF)
        seasons_current_year = seasonality[djf_idx:]
        seasons_previous_year = seasonality[:djf_idx]

    return seasons_previous_year, seasons_current_year


def shift_to_year_middle(ds: T_Xarray) -> T_Xarray:
    if not is_annual_resolution(ds):
        raise ValueError("Input is not at annual resolution")
    return ds.assign_coords(time=np.floor(ds.time) + 0.5)
