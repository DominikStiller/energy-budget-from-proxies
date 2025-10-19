from __future__ import annotations

from unittest import TestCase

import numpy as np

from lmrecon.psm import LinearPSM


class TestLinearPSM(TestCase):
    def test_calibrate(self):
        rng = np.random.default_rng(52561561)

        slope = 5
        intercept = 3
        err_std = 0.8
        n_sample = 1000
        x = rng.uniform(0, 10, size=(1, n_sample))
        y_true = intercept + x * slope
        y = y_true + rng.normal(loc=0, scale=err_std, size=n_sample)

        psm = LinearPSM(None, "tas", None, None, None)
        psm.calibrate(x, y)

        self.assertAlmostEqual(intercept, np.squeeze(psm._model.intercept_), 1)
        self.assertAlmostEqual(slope, np.squeeze(psm._model.coef_), 1)
        self.assertAlmostEqual(err_std, psm.err_std, 1)

    def test_forward_multiple(self):
        n_sample = 20
        x = np.linspace(0, 10, n_sample)[np.newaxis, :]  # 1 x n_sample
        y = 3 + x * 5  # 1 x n_sample

        psm = LinearPSM(None, "tas", None, None, None)
        psm.calibrate(x, y)

        yhat = psm.forward(x)

        self.assertEqual(yhat.ndim, 2)
        self.assertEqual(yhat.shape[0], 1)
        self.assertEqual(yhat.shape[1], n_sample)
        np.testing.assert_almost_equal(yhat, y)
