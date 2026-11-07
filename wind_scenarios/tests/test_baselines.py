import unittest

import numpy as np
import pandas as pd

from tests.helpers import small_windows
from windgen.baselines import ClimatologyBaseline, PersistenceBaseline, RidgeResidualBaseline
from windgen.data import Normalizer, build_features, make_synthetic_raw, make_windows


class TestBaselines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        feats = build_features(make_synthetic_raw(24 * 365 * 2, seed=3))
        cls.train = make_windows(feats, end="2019-07-01", stride=3)
        cls.test = make_windows(feats, start="2019-07-01", stride=25)

    def test_persistence(self):
        s = PersistenceBaseline().fit(self.train).scenarios(self.test)
        self.assertEqual(s.shape, (len(self.test), 1, 24))
        np.testing.assert_allclose(s[:, 0, :], self.test.hist[:, :, 0], rtol=1e-6)

    def test_persistence_needs_enough_lookback(self):
        feats = build_features(make_synthetic_raw(24 * 20))
        w = make_windows(feats, lookback=6, horizon=24)
        with self.assertRaises(ValueError):
            PersistenceBaseline().scenarios(w)

    def test_climatology_samples_from_training_pool(self):
        b = ClimatologyBaseline(min_group=5).fit(self.train)
        s = b.scenarios(self.test, n=7, seed=0)
        self.assertEqual(s.shape, (len(self.test), 7, 24))
        pool = {tuple(np.round(r, 5)) for r in self.train.target.astype(np.float64)}
        for r in s[0]:
            self.assertIn(tuple(np.round(r, 5)), pool)
        np.testing.assert_array_equal(s, b.scenarios(self.test, n=7, seed=0))
        self.assertTrue(np.isfinite(s).all())

    def test_climatology_requires_fit(self):
        with self.assertRaises(RuntimeError):
            ClimatologyBaseline().scenarios(self.test)

    def test_climatology_fallback_for_unseen_month(self):
        b = ClimatologyBaseline(min_group=10**9).fit(self.train)  # force fallback to hour pools
        s = b.scenarios(self.test, n=2)
        self.assertEqual(s.shape, (len(self.test), 2, 24))

    def test_ridge_beats_climatology_mean_on_coupled_data(self):
        norm = Normalizer().fit(self.train)
        ridge = RidgeResidualBaseline(norm).fit(self.train)
        clim = ClimatologyBaseline(min_group=5).fit(self.train)
        obs = self.test.target
        e_r = np.abs(ridge.scenarios(self.test, 30).mean(1) - obs).mean()
        e_c = np.abs(clim.scenarios(self.test, 30).mean(1) - obs).mean()
        self.assertLess(e_r, e_c)

    def test_ridge_recovers_linear_relation(self):
        _, w = small_windows(24 * 200)
        norm = Normalizer().fit(w)
        # make target an exact linear function of the conditioning vector
        X = norm.cond(w.hist, w.cal)
        rng = np.random.default_rng(0)
        coef = rng.standard_normal((X.shape[1], 24)) * 0.05
        w.target = norm.denorm_target(X @ coef).astype(np.float32)
        r = RidgeResidualBaseline(norm, alpha=1e-6).fit(w)
        self.assertLess(np.abs(r.resid).max(), 1e-3)

    def test_ridge_requires_fit_and_nonneg(self):
        norm = Normalizer().fit(self.train)
        r = RidgeResidualBaseline(norm)
        with self.assertRaises(RuntimeError):
            r.scenarios(self.test)
        s = r.fit(self.train).scenarios(self.test, 5)
        self.assertTrue((s >= 0).all())


if __name__ == "__main__":
    unittest.main()
