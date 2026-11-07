import os
import tempfile
import unittest

import numpy as np

from tests.helpers import small_windows
from windgen.data import Normalizer
from windgen.diffusion import ConditionalDDPM, ScenarioGenerator, cosine_betas, time_embedding


def toy_data(n=3000, seed=0):
    """y = [c0, 2*c0] + noise: a trajectory fully determined by the condition up to noise."""
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((n, 2))
    y = np.stack([c[:, 0], -c[:, 0]], axis=1) + 0.1 * rng.standard_normal((n, 2))
    return y, c


class TestSchedule(unittest.TestCase):
    def test_schedule_properties(self):
        m = ConditionalDDPM(4, 3, T=50, hidden=8)
        self.assertTrue((m.betas > 0).all() and (m.betas < 1).all())
        self.assertTrue((np.diff(m.ab) < 0).all())
        self.assertTrue(0 < m.ab[-1] < 0.01)
        self.assertAlmostEqual(m.coef_x0[0], 1.0, places=6)
        self.assertAlmostEqual(m.coef_xt[0], 0.0, places=6)
        self.assertAlmostEqual(m.post_var[0], 0.0, places=9)
        self.assertEqual(len(cosine_betas(10)), 10)

    def test_time_embedding(self):
        e = time_embedding(np.arange(5), 8)
        self.assertEqual(e.shape, (5, 8))
        self.assertTrue(np.isfinite(e).all())
        with self.assertRaises(ValueError):
            time_embedding(1, 7)

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            ConditionalDDPM(0, 3)
        with self.assertRaises(ValueError):
            ConditionalDDPM(3, 3, T=1)


class TestTrainingAndSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        y, c = toy_data()
        cls.y, cls.c = y, c
        cls.m = ConditionalDDPM(2, 2, T=30, hidden=64, n_layers=2, seed=0)
        cls.hist = cls.m.fit(y, c, steps=1500, batch_size=128, lr=2e-3, log_every=500,
                             val=(y[:300], c[:300]), verbose=False)

    def test_loss_decreases(self):
        self.assertLess(self.hist["train_loss"][-1], self.hist["train_loss"][0])
        self.assertTrue(np.isfinite(self.hist["val_loss"]).all())

    def test_sample_shape_and_finite(self):
        s = self.m.sample(self.c[:7], n_samples=5, seed=1)
        self.assertEqual(s.shape, (7, 5, 2))
        self.assertTrue(np.isfinite(s).all())

    def test_learns_conditional_mean_and_spread(self):
        rng = np.random.default_rng(5)
        c = rng.standard_normal((60, 2))
        s = self.m.sample(c, n_samples=40, seed=2)
        mean = s.mean(axis=1)
        corr0 = np.corrcoef(mean[:, 0], c[:, 0])[0, 1]
        corr1 = np.corrcoef(mean[:, 1], -c[:, 0])[0, 1]
        self.assertGreater(corr0, 0.95)
        self.assertGreater(corr1, 0.95)
        self.assertLess(np.abs(mean[:, 0] - c[:, 0]).mean(), 0.15)
        spread = s.std(axis=1).mean()
        self.assertTrue(0.03 < spread < 0.4, spread)  # true noise sd = 0.1: not collapsed, not huge

    def test_deterministic_given_seed(self):
        a = self.m.sample(self.c[:4], 3, seed=9)
        b = self.m.sample(self.c[:4], 3, seed=9)
        c = self.m.sample(self.c[:4], 3, seed=10)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.allclose(a, c))

    def test_scenarios_differ_within_a_condition(self):
        s = self.m.sample(self.c[:2], 6, seed=3)
        self.assertGreater(s.std(axis=1).min(), 1e-4)

    def test_common_noise_same_noise_for_equal_conditions(self):
        cond = np.repeat(self.c[:1], 3, axis=0)
        s = self.m.sample(cond, 4, seed=4, common_noise=True)
        np.testing.assert_allclose(s[0], s[1], atol=1e-6)
        np.testing.assert_allclose(s[0], s[2], atol=1e-6)

    def test_weights_restored_after_sampling_with_ema(self):
        before = [p.copy() for p in self.m.net.params()]
        self.m.sample(self.c[:2], 2, use_ema=True)
        for b, p in zip(before, self.m.net.params()):
            np.testing.assert_array_equal(b, p)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.npz")
            self.m.save(path, extra={"foo": np.arange(3)})
            m2 = ConditionalDDPM.load(path)
        a = self.m.sample(self.c[:3], 2, seed=7)
        b = m2.sample(self.c[:3], 2, seed=7)
        np.testing.assert_allclose(a, b, atol=1e-6)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            self.m.sample(np.zeros((3, 5)), 2)
        with self.assertRaises(ValueError):
            self.m.sample(self.c[:2], 0)
        bad = self.c[:2].copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.m.sample(bad, 2)
        with self.assertRaises(ValueError):
            self.m.fit(self.y[:10], self.c[:9], steps=1)
        with self.assertRaises(ValueError):
            self.m.fit(np.full((10, 2), np.nan), self.c[:10], steps=1)


class TestAttentionBackbone(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, w = small_windows()
        cls.w = w
        cls.norm = Normalizer().fit(w)
        cls.cond = cls.norm.cond(w.hist, w.cal)
        cls.m = ConditionalDDPM(24, cls.cond.shape[1], T=10, hidden=16, n_layers=2,
                                backbone="attn", lookback=24)
        cls.m.fit(cls.norm.norm_target(w.target), cls.cond, steps=30, batch_size=32, verbose=False)

    def test_attention_weights_returned(self):
        x, a = self.m.sample(self.cond[:3], n_samples=4, return_attention=True)
        self.assertEqual(x.shape, (3, 4, 24))
        self.assertEqual(a.shape, (3, 4, 24))  # (N, heads, L)
        np.testing.assert_allclose(a.sum(-1), 1.0, rtol=1e-5)
        np.testing.assert_allclose(self.m.sample(self.cond[:3], 4), x)  # same samples without recording
        gen = ScenarioGenerator(self.m, self.norm)
        self.assertEqual(gen.attention_from_arrays(self.w.hist[:2], self.w.cal[:2], n=3).shape, (2, 4, 24))

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.npz")
            self.m.save(path)
            m2 = ConditionalDDPM.load(path)
        self.assertEqual(m2.cfg["backbone"], "attn")
        np.testing.assert_allclose(m2.sample(self.cond[:2], 3, seed=5), self.m.sample(self.cond[:2], 3, seed=5))

    def test_validation(self):
        with self.assertRaises(ValueError):
            ConditionalDDPM(24, 50, backbone="attn", lookback=24)  # 50-4 not divisible by 24
        with self.assertRaises(ValueError):
            ConditionalDDPM(24, 10, backbone="cnn")
        with self.assertRaises(ValueError):
            ConditionalDDPM(4, 3, T=5, hidden=8).sample(np.zeros((1, 3)), return_attention=True)


class TestScenarioGenerator(unittest.TestCase):
    def test_physical_units_nonnegative(self):
        _, w = small_windows()
        n = Normalizer().fit(w)
        m = ConditionalDDPM(24, 24 * 8 + 4, T=10, hidden=16, n_layers=1)
        g = ScenarioGenerator(m, n)
        s = g.scenarios(w.subset(slice(0, 5)), n=3, seed=0)
        self.assertEqual(s.shape, (5, 3, 24))
        self.assertTrue((s >= 0).all() and np.isfinite(s).all())


if __name__ == "__main__":
    unittest.main()
