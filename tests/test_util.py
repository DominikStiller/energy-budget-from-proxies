from __future__ import annotations

from unittest import TestCase

import numpy as np

from lmrecon.util import get_closest_gridpoint


class TestUtil(TestCase):
    def test_get_closest_gridpoint_exactmatch(self):
        # Exact match
        np.testing.assert_almost_equal(
            get_closest_gridpoint(30, 45, np.array([-30, 0, 30, 60]), np.array([15, 30, 45, 60])),
            [30, 45],
        )

    def test_get_closest_gridpoint_realistic(self):
        # Exact match
        np.testing.assert_almost_equal(
            get_closest_gridpoint(35, 20, np.array([-30, 0, 30, 60]), np.array([15, 30, 45, 60])),
            [30, 45],
        )

    def test_get_closest_gridpoint_almosthalfway(self):
        # Almost halfway between two points
        np.testing.assert_almost_equal(
            get_closest_gridpoint(30, 14.999, np.array([30, 30, 30]), np.array([10, 20, 30])),
            [30, 10],
        )
