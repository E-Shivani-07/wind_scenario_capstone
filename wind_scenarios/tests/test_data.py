import unittest

import numpy as np
import pandas as pd

from tests.helpers import small_windows
from windgen.data import (CAL_DIM, FEATURES, GROUPS, Normalizer, build_features, calendar_features,
                          direction_degrees, load_raw, make_synthetic_raw, make_windows,
                          relative_humidity)


class TestFeatures(unittest.TestCase):
    def test_relative_humidity_bounds_and_saturation(self):
        t = np.array([290.0, 300.0, 310.0])
        self.assertTrue(np.allclose(relative_humidity(t, t), 100.0))
        rh = relative_humidity(t, t - 10)
        self.assertTrue(((rh > 0) & (rh < 100)).all())
        self.assertTrue((relative_humidity(t, t + 5) <= 100.0).all())

    def test_direction_degrees(self):
        # wind blowing towards the east (u>0) comes from the west = 270
        self.assertAlmostEqual(float(direction_degrees(1.0, 0.0)), 270.0)
        self.assertAlmostEqual(float(direction_degrees(0.0, -1.0)), 0.0)  # from the north
        self.assertAlmostEqual(float(direction_degrees(-1.0, 0.0)), 90.0)

    def test_build_features(self):
        raw = make_synthetic_raw(500)
        f = build_features(raw)
        self.assertEqual(list(f.columns), FEATURES)
        self.assertEqual(len(f), 500)
        self.assertTrue(np.isfinite(f.to_numpy()).all())
        self.assertTrue(np.allclose(f.wind_speed, np.hypot(raw.u10, raw.v10)))
        self.assertTrue(np.allclose(f.wind_dir_sin ** 2 + f.wind_dir_cos ** 2, 1.0))
        self.assertTrue((f.precip_log1p >= 0).all() and (f.ssrd_mj >= 0).all())

    def test_build_features_missing_column(self):
        with self.assertRaises(ValueError):
            build_features(make_synthetic_raw(50).drop(columns=["sp"]))

    def test_groups_cover_all_features_once(self):
        cols = sorted(c for v in GROUPS.values() for c in v)
        self.assertEqual(cols, list(range(len(FEATURES))))

    def test_calendar(self):
        c = calendar_features(pd.date_range("2020-01-01", periods=48, freq="h"))
        self.assertEqual(c.shape, (48, CAL_DIM))
        self.assertTrue(np.allclose(c[:, 0] ** 2 + c[:, 1] ** 2, 1))
        self.assertTrue(np.allclose(c[:24, :2], c[24:, :2]))


class TestWindows(unittest.TestCase):
    def test_shapes_and_alignment(self):
        feats, w = small_windows()
        N = len(w)
        self.assertEqual(w.hist.shape, (N, 24, len(FEATURES)))
        self.assertEqual(w.target.shape, (N, 24))
        self.assertEqual(w.cal.shape, (N, CAL_DIM))
        ws = feats.wind_speed.to_numpy()
        pos = feats.index.get_indexer(pd.DatetimeIndex(w.origin))
        for k in (0, N // 2, N - 1):
            t = pos[k]
            np.testing.assert_allclose(w.target[k], ws[t:t + 24], rtol=1e-5)
            np.testing.assert_allclose(w.hist[k, -1, 0], ws[t - 1], rtol=1e-5)
            np.testing.assert_allclose(w.hist[k, 0, 0], ws[t - 24], rtol=1e-5)

    def test_no_leakage_history_before_target(self):
        feats, w = small_windows()
        pos = feats.index.get_indexer(pd.DatetimeIndex(w.origin))
        # history ends exactly one hour before the origin, which is the first target hour
        np.testing.assert_allclose(w.hist[:, -1, 0], feats.wind_speed.to_numpy()[pos - 1], rtol=1e-5)

    def test_period_filter_and_stride(self):
        feats = build_features(make_synthetic_raw(24 * 100))
        w = make_windows(feats, 24, 24, start="2018-02-01", end="2018-03-01", stride=5)
        o = pd.DatetimeIndex(w.origin)
        self.assertTrue((o >= "2018-02-01").all() and (o < "2018-03-01").all())
        self.assertTrue((np.diff(o.values) >= np.timedelta64(5, 'h')).all())

    def test_gap_windows_removed(self):
        feats = build_features(make_synthetic_raw(24 * 30))
        gapped = feats.drop(feats.index[200:210])
        w = make_windows(gapped, 24, 24)
        pos = gapped.index.get_indexer(pd.DatetimeIndex(w.origin))
        for t in pos:
            span = gapped.index[t + 23] - gapped.index[t - 24]
            self.assertEqual(span, pd.Timedelta(hours=47))

    def test_max_windows_deterministic(self):
        feats = build_features(make_synthetic_raw(24 * 60))
        a = make_windows(feats, max_windows=50, seed=1)
        b = make_windows(feats, max_windows=50, seed=1)
        self.assertEqual(len(a), 50)
        np.testing.assert_array_equal(a.origin, b.origin)

    def test_errors(self):
        feats = build_features(make_synthetic_raw(30))
        with self.assertRaises(ValueError):
            make_windows(feats, 24, 24)
        feats = build_features(make_synthetic_raw(24 * 10))
        with self.assertRaises(ValueError):
            make_windows(feats, start="2030-01-01")
        with self.assertRaises(ValueError):
            make_windows(feats, stride=0)


class TestNormalizer(unittest.TestCase):
    def test_roundtrip_and_stats(self):
        _, w = small_windows()
        n = Normalizer().fit(w)
        c = n.cond(w.hist, w.cal)
        self.assertEqual(c.shape, (len(w), 24 * len(FEATURES) + CAL_DIM))
        z = c[:, :-CAL_DIM].reshape(len(w), 24, -1)
        self.assertTrue(np.allclose(z.reshape(-1, z.shape[-1]).mean(0), 0, atol=1e-6))
        self.assertTrue(np.allclose(z.reshape(-1, z.shape[-1]).std(0), 1, atol=1e-6))
        np.testing.assert_allclose(n.denorm_target(n.norm_target(w.target)), w.target, rtol=1e-5)
        self.assertTrue(np.isfinite(c).all())

    def test_state_roundtrip(self):
        _, w = small_windows()
        n = Normalizer().fit(w)
        m = Normalizer.from_state(n.state())
        np.testing.assert_allclose(n.cond(w.hist, w.cal), m.cond(w.hist, w.cal))

    def test_shape_validation(self):
        _, w = small_windows()
        n = Normalizer().fit(w)
        with self.assertRaises(ValueError):
            n.cond(w.hist[:, :, 0], w.cal)
        with self.assertRaises(ValueError):
            n.cond(w.hist, w.cal[:-1])


class TestLoadRaw(unittest.TestCase):
    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_raw("/nonexistent_dir_xyz")

    def test_roundtrip_csv(self):
        import tempfile
        from pathlib import Path
        raw = make_synthetic_raw(100)
        with tempfile.TemporaryDirectory() as d:
            r = raw.reset_index()
            for fname, cols in {"wind.csv": ["u10", "v10"], "temperature.csv": ["t2m", "d2m"],
                                "pressure.csv": ["sp", "tp"], "radiation.csv": ["ssrd"]}.items():
                r[["valid_time", *cols]].assign(latitude=14.6, longitude=79.6).to_csv(Path(d) / fname, index=False)
            got = load_raw(d)
        self.assertEqual(len(got), 100)
        np.testing.assert_allclose(got.u10.to_numpy(), raw.u10.to_numpy(), rtol=1e-9)


if __name__ == "__main__":
    unittest.main()
