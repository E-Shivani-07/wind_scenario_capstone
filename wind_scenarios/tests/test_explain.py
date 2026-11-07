import unittest

import numpy as np

from tests.helpers import small_windows
from windgen.data import FEATURES, GROUPS, Normalizer
from windgen.diffusion import ConditionalDDPM, ScenarioGenerator
from windgen.explain import GROUP_NAMES, ScenarioExplainer, rank_agreement, shift_group

P = FEATURES.index("sp")


class StubGenerator:
    """Scenario mean = 3 + 0.01*(pressure of the LAST history hour - 100000)/10 ; no other input matters."""

    def scenarios_from_arrays(self, hist, cal, n=4, seed=0, common_noise=False):
        hist = np.asarray(hist)
        base = 3.0 + (hist[:, -1, P] - 100000.0) / 1000.0
        noise = np.random.default_rng(seed).standard_normal((n, 24)) * 0.1
        return np.clip(base[:, None, None] + noise[None], 0, None)


class TestShiftGroup(unittest.TestCase):
    def setUp(self):
        _, w = small_windows()
        self.h = w.hist[:3].astype(np.float64)

    def test_each_group_changes_only_its_columns(self):
        for g, cols in GROUPS.items():
            out = shift_group(self.h, g, 0.1 if g in ("wind_speed", "radiation") else 5.0)
            other = [c for c in range(len(FEATURES)) if c not in cols]
            np.testing.assert_allclose(out[..., other], self.h[..., other])
            self.assertFalse(np.allclose(out[..., cols], self.h[..., cols]) and g != "precipitation")

    def test_values(self):
        out = shift_group(self.h, "pressure", -500.0)
        np.testing.assert_allclose(out[..., P], self.h[..., P] - 500.0)
        out = shift_group(self.h, "wind_speed", 0.1)
        np.testing.assert_allclose(out[..., 0], self.h[..., 0] * 1.1)
        out = shift_group(self.h, "humidity", 1000.0)
        self.assertTrue((out[..., 4] <= 100).all())
        out = shift_group(self.h, "wind_direction", 90.0)
        np.testing.assert_allclose(out[..., 1] ** 2 + out[..., 2] ** 2, 1.0, atol=1e-6)

    def test_input_not_modified_and_bad_group(self):
        before = self.h.copy()
        shift_group(self.h, "temperature", 3.0)
        np.testing.assert_array_equal(before, self.h)
        with self.assertRaises(ValueError):
            shift_group(self.h, "nonsense", 1.0)


class TestExplainerWithStub(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.w = small_windows()
        cls.ex = ScenarioExplainer(StubGenerator(), cls.w, n_scenarios=4, seed=0)

    def test_shapley_efficiency(self):
        i = 10
        r = self.ex.shapley(self.w.hist[i], self.w.cal[i], n_perm=8)
        self.assertEqual(len(r["phi"]), len(GROUP_NAMES))
        self.assertAlmostEqual(r["phi"].sum(), r["prediction"] - r["base_value"], places=9)

    def test_shapley_identifies_the_only_relevant_group(self):
        r = self.ex.shapley(self.w.hist[5], self.w.cal[5], n_perm=12)
        phi = dict(zip(r["names"], r["phi"]))
        for g, v in phi.items():
            if g != "pressure":
                self.assertAlmostEqual(v, 0.0, places=9)
        self.assertNotAlmostEqual(phi["pressure"], 0.0, places=3)

    def test_global_importance(self):
        imp = self.ex.global_importance(self.w, n_instances=6, n_perm=6)
        self.assertEqual(imp["names"][int(np.argmax(imp["share"]))], "pressure")
        self.assertAlmostEqual(imp["share"].sum(), 1.0)
        self.assertEqual(imp["phi_all"].shape, (6, len(GROUP_NAMES)))

    def test_temporal_attribution_peaks_at_last_lag(self):
        t = self.ex.temporal_attribution(self.w, n_instances=5)
        self.assertEqual(len(t["importance"]), 24)
        self.assertEqual(int(np.argmax(t["importance"])), 23)  # lag index 23 = hour before origin
        self.assertEqual(int(t["lag_hours_before_origin"][23]), 1)
        self.assertAlmostEqual(t["importance"][:23].sum(), 0.0, places=9)
        self.assertAlmostEqual(t["share"].sum(), 1.0)

    def test_deletion_faithfulness_and_rank_agreement(self):
        imp = self.ex.global_importance(self.w, n_instances=4, n_perm=6)
        f = self.ex.deletion_faithfulness(self.w, imp)
        self.assertGreater(f["top1_deletion_ratio"], 3.0)  # only pressure matters for the stub
        self.assertGreater(f["spearman"], 0.0)
        self.assertAlmostEqual(rank_agreement([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(rank_agreement([1, 2, 3], [3, 2, 1]), -1.0)

    def test_counterfactual_direction_and_irrelevant_group(self):
        h, c = self.w.hist[0], self.w.cal[0]
        up = self.ex.counterfactual(h, c, "pressure", +1000.0)
        down = self.ex.counterfactual(h, c, "pressure", -1000.0)
        self.assertAlmostEqual(up["mean_shift"], 1.0, places=6)
        self.assertAlmostEqual(down["mean_shift"], -1.0, places=6)
        self.assertGreater(up["wasserstein"], 0.5)
        none = self.ex.counterfactual(h, c, "temperature", 3.0)
        self.assertAlmostEqual(none["mean_shift"], 0.0, places=9)
        self.assertEqual(none["base"].shape, (4, 24))

    def test_find_counterfactual(self):
        h, c = self.w.hist[0], self.w.cal[0]
        r = self.ex.find_counterfactual(h, c, "pressure", target_shift=0.3)
        self.assertIsNotNone(r)
        self.assertGreaterEqual(r["mean_shift"], 0.3)
        self.assertGreater(r["delta"], 0)
        self.assertLess(abs(r["delta"]), 500)  # 300 Pa needed; grid step 100
        self.assertIsNone(self.ex.find_counterfactual(h, c, "temperature", 1.0))  # impossible
        with self.assertRaises(ValueError):
            self.ex.find_counterfactual(h, c, "pressure", 0.0)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            self.ex.shapley(self.w.hist[:2], self.w.cal[:2])


class TestExplainerWithRealModel(unittest.TestCase):
    def test_runs_end_to_end_and_efficiency_holds(self):
        _, w = small_windows(24 * 150)
        norm = Normalizer().fit(w)
        m = ConditionalDDPM(24, 24 * 8 + 4, T=8, hidden=16, n_layers=1)
        m.fit(norm.norm_target(w.target), norm.cond(w.hist, w.cal), steps=20, batch_size=32,
              log_every=10, verbose=False)
        ex = ScenarioExplainer(ScenarioGenerator(m, norm), w, n_scenarios=3, n_background=8)
        r = ex.shapley(w.hist[0], w.cal[0], n_perm=3)
        self.assertAlmostEqual(r["phi"].sum(), r["prediction"] - r["base_value"], places=8)
        np.testing.assert_allclose(ex.predict_mean(w.hist[:1], w.cal[:1])[0], r["prediction"], rtol=1e-6)
        cf = ex.counterfactual(w.hist[0], w.cal[0], "pressure", -500.0)
        self.assertTrue(np.isfinite(cf["mean_shift"]))
        # deterministic (common random numbers)
        a = ex.predict_mean(w.hist[:2], w.cal[:2])
        b = ex.predict_mean(w.hist[:2], w.cal[:2])
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
