import unittest

import numpy as np

from tests.helpers import small_windows  # noqa: F401
from windgen import metrics as M


class TestScores(unittest.TestCase):
    def test_crps_single_member_is_mae(self):
        rng = np.random.default_rng(0)
        obs = rng.random((10, 6))
        ens = rng.random((10, 1, 6))
        self.assertAlmostEqual(M.crps_ensemble(obs, ens), np.abs(ens[:, 0] - obs).mean())

    def test_crps_known_value(self):
        # ensemble {0, 1}, obs 0.5: E|X-y| = 0.5 ; E|X-X'| = 0.5 -> crps = 0.5 - 0.25
        self.assertAlmostEqual(M.crps_ensemble(np.array([[0.5]]), np.array([[[0.0], [1.0]]])), 0.25)

    def test_crps_perfect_and_nonnegative(self):
        obs = np.random.default_rng(1).random((5, 4))
        ens = np.repeat(obs[:, None, :], 7, axis=1)
        self.assertAlmostEqual(M.crps_ensemble(obs, ens), 0.0)
        rng = np.random.default_rng(2)
        self.assertGreaterEqual(M.crps_ensemble(rng.random((20, 4)), rng.random((20, 9, 4))), 0)

    def test_crps_matches_bruteforce(self):
        rng = np.random.default_rng(3)
        obs, ens = rng.random((6, 3)), rng.random((6, 5, 3))
        t1 = np.abs(ens - obs[:, None]).mean(1)
        t2 = np.abs(ens[:, :, None] - ens[:, None]).mean((1, 2))
        self.assertAlmostEqual(M.crps_ensemble(obs, ens), (t1 - 0.5 * t2).mean())

    def test_energy_score(self):
        obs = np.zeros((1, 2))
        ens = np.array([[[3.0, 4.0]]])
        self.assertAlmostEqual(M.energy_score(obs, ens), 5.0)  # single member: Euclidean error
        ens2 = np.repeat(obs[:, None, :], 4, axis=1)
        self.assertAlmostEqual(M.energy_score(obs, ens2), 0.0)

    def test_point_errors(self):
        r = M.point_errors(np.zeros((2, 2)), np.ones((2, 3, 2)))
        self.assertAlmostEqual(r["mae"], 1.0)
        self.assertAlmostEqual(r["rmse"], 1.0)

    def test_shape_and_nan_validation(self):
        with self.assertRaises(ValueError):
            M.crps_ensemble(np.zeros((3, 4)), np.zeros((3, 4)))
        with self.assertRaises(ValueError):
            M.crps_ensemble(np.zeros((3, 4)), np.zeros((2, 5, 4)))
        with self.assertRaises(ValueError):
            M.crps_ensemble(np.full((3, 4), np.nan), np.zeros((3, 5, 4)))


class TestCalibration(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.obs = rng.standard_normal((400, 12))
        self.good = rng.standard_normal((400, 19, 12))          # same distribution
        self.narrow = 0.2 * rng.standard_normal((400, 19, 12))  # over-confident

    def test_rank_histogram_flat_for_calibrated(self):
        h = M.rank_histogram(self.obs, self.good)
        self.assertAlmostEqual(h.sum(), 1.0)
        self.assertLess(np.abs(h - 1 / 20).max(), 0.01)

    def test_rank_histogram_u_shaped_for_overconfident(self):
        h = M.rank_histogram(self.obs, self.narrow)
        self.assertGreater(h[0] + h[-1], 0.5)

    def test_coverage_and_calibration_error(self):
        self.assertAlmostEqual(M.coverage(self.obs, self.good, 0.9), 0.9, delta=0.03)
        self.assertLess(M.calibration_error(self.obs, self.good), 0.04)
        self.assertGreater(M.calibration_error(self.obs, self.narrow), 0.2)
        with self.assertRaises(ValueError):
            M.coverage(self.obs, self.good, 1.5)

    def test_spread_skill(self):
        self.assertAlmostEqual(M.spread_skill(self.obs, self.good), 1.0, delta=0.1)
        self.assertLess(M.spread_skill(self.obs, self.narrow), 0.4)
        self.assertTrue(np.isnan(M.spread_skill(self.obs, self.good[:, :1])))

    def test_ties_handled_in_pit(self):
        obs = np.zeros((50, 4))
        ens = np.zeros((50, 9, 4))
        r = M.pit_ranks(obs, ens)
        self.assertTrue(((r >= 0) & (r <= 9)).all())
        self.assertGreater(len(np.unique(r)), 5)


class TestSimilarityAndDiversity(unittest.TestCase):
    def test_ks_wasserstein_identical_and_shifted(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((100, 10))
        same = M.ks_wasserstein(x, x[:, None, :])
        self.assertAlmostEqual(same["ks"], 0.0)
        self.assertAlmostEqual(same["wasserstein"], 0.0)
        shifted = M.ks_wasserstein(x, (x + 1.0)[:, None, :])
        self.assertAlmostEqual(shifted["wasserstein"], 1.0, places=6)
        self.assertGreater(shifted["ks"], 0.3)

    def test_acf_of_ar1_and_noise(self):
        rng = np.random.default_rng(1)
        n, H = 400, 24
        e = rng.standard_normal((n, H))
        ar = np.zeros((n, H))
        for t in range(1, H):
            ar[:, t] = 0.9 * ar[:, t - 1] + e[:, t]
        self.assertAlmostEqual(M.acf(ar, 1)[0], 0.9, delta=0.05)
        self.assertAlmostEqual(M.acf(e, 1)[0], 0.0, delta=0.05)
        self.assertGreater(M.acf_distance(ar, e[:, None, :]), 0.2)
        self.assertAlmostEqual(M.acf(np.ones((5, 8)), 2)[0], 0.0)  # constant -> 0, not NaN

    def test_ramp_wasserstein(self):
        rng = np.random.default_rng(2)
        smooth = np.cumsum(0.1 * rng.standard_normal((50, 24)), axis=1)
        rough = 2.0 * rng.standard_normal((50, 24))
        self.assertGreater(M.ramp_wasserstein(smooth, rough[:, None, :]), 0.5)
        self.assertAlmostEqual(M.ramp_wasserstein(smooth, smooth[:, None, :]), 0.0)

    def test_diversity(self):
        rng = np.random.default_rng(3)
        same = np.ones((4, 5, 6))
        self.assertAlmostEqual(M.diversity(same), 0.0)
        self.assertEqual(M.diversity(same[:, :1]), 0.0)
        ens = rng.standard_normal((200, 6, 10))
        # E|Xi - Xj| RMS over hours ~ sqrt(2) for independent unit normals
        self.assertAlmostEqual(M.diversity(ens), np.sqrt(2) * 0.97, delta=0.12)

    def test_evaluate_scenarios_keys_and_none_for_single_member(self):
        rng = np.random.default_rng(4)
        obs = rng.random((30, 24))
        full = M.evaluate_scenarios(obs, rng.random((30, 8, 24)))
        single = M.evaluate_scenarios(obs, rng.random((30, 1, 24)))
        for k in ("crps", "energy_score", "mae", "rmse", "ks", "wasserstein", "acf_distance",
                  "ramp_wasserstein", "diversity", "spread_skill", "calibration_error", "coverage_90"):
            self.assertIn(k, full)
            self.assertIsNotNone(full[k])
        self.assertIsNone(single["calibration_error"])
        self.assertIsNone(single["spread_skill"])


if __name__ == "__main__":
    unittest.main()
