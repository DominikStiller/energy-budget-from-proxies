from __future__ import annotations

from unittest import TestCase

import cftime
import numpy as np

from lmrecon.time import (
    Season,
    convert_decimal_year_to_datetime,
    map_decimal_to_season,
    map_season_to_decimal,
    round_to_nearest_season,
)


class TestTime(TestCase):
    def test_map_decimal_to_season(self):
        self.assertEqual(map_decimal_to_season(map_season_to_decimal(Season.DJF)), Season.DJF)
        self.assertEqual(map_decimal_to_season(map_season_to_decimal(Season.MAM)), Season.MAM)
        self.assertEqual(map_decimal_to_season(map_season_to_decimal(Season.JJA)), Season.JJA)
        self.assertEqual(map_decimal_to_season(map_season_to_decimal(Season.SON)), Season.SON)

    def test_round_to_nearest_season(self):
        # Monthly with some jitter
        time = 100 + np.arange(0.5 / 12, 3, 1 / 12)
        time += 0.01 * np.random.randn(*time.shape)

        # Naive method
        year = np.array(time).astype(int)
        decimal_month = time - year
        # Assign Dec to DJF of next year
        year = np.where(11 / 12 <= decimal_month, year + 1, year)
        # Not the cleanest but this is the easiest way to truncate months to seasons
        desired = year + [map_season_to_decimal(map_decimal_to_season(t)) for t in decimal_month]

        np.testing.assert_allclose(round_to_nearest_season(time), desired)

    def test_convert_decimal_year_to_datetime(self):
        np.testing.assert_equal(
            convert_decimal_year_to_datetime(
                [
                    1,
                    1 + 364 / 365,
                    100,
                    2013 + (31 + 28) / 365,
                ]
            ),
            [
                cftime.datetime(1, 1, 1, calendar="noleap"),
                cftime.datetime(1, 12, 31, calendar="noleap"),
                cftime.datetime(100, 1, 1, calendar="noleap"),
                cftime.datetime(2013, 3, 1, calendar="noleap"),
            ],
        )
        np.testing.assert_equal(
            convert_decimal_year_to_datetime(
                [
                    1,
                    1 + 364 / 365,
                    100,
                    2012 + 2 / 366,  # leap year
                    2012 + (31 + 28) / 366,  # leap year
                    2013 + (31 + 28) / 365,
                ],
                leap_years=True,
            ),
            [
                cftime.datetime(1, 1, 1, calendar="proleptic_gregorian"),
                cftime.datetime(1, 12, 31, calendar="proleptic_gregorian"),
                cftime.datetime(100, 1, 1, calendar="proleptic_gregorian"),
                cftime.datetime(2012, 1, 3, calendar="proleptic_gregorian"),
                cftime.datetime(2012, 2, 29, calendar="proleptic_gregorian"),
                cftime.datetime(2013, 3, 1, calendar="proleptic_gregorian"),
            ],
        )


class TestSeason(TestCase):
    def test_ordering(self):
        self.assertLess(Season.DJF, Season.MAM)
        self.assertLess(Season.MAM, Season.JJA)
        self.assertLess(Season.JJA, Season.SON)

        self.assertLess((1, Season.DJF), (1, Season.MAM))
        self.assertLess((1, Season.DJF), (2, Season.DJF))
        self.assertLess((1, Season.MAM), (2, Season.DJF))
        self.assertLess((1, Season.SON), (2, Season.DJF))
