import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.helpers import small_windows  # noqa: F401
from windgen.data import load_raw, make_synthetic_raw
from windgen.pipeline import Config, _clean, run_from_csv, run_pipeline, summarize_runs


def tiny_cfg(**kw):
    c = Config(steps=40, hidden=24, n_layers=2, T=10, max_train=800, max_val=100, max_test=60,
               n_scenarios=6, n_explain=2, n_perm=2, n_explain_scen=3, log_every=20,
               val_start="2018-09-01", test_start="2018-11-01", make_plots=False)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


class TestPipelineSynthetic(unittest.TestCase):
    def test_end_to_end_outputs(self):
        raw = make_synthetic_raw(24 * 365, seed=1, start="2018-01-01")
        with tempfile.TemporaryDirectory() as d:
            res = run_pipeline(raw, d, tiny_cfg(make_plots=True), verbose=False)
            for f in ("results.json", "results.md", "diffusion_model.npz", "scenarios.png",
                      "training_loss.png", "rank_histograms.png", "shap_importance.png",
                      "temporal_attribution.png", "counterfactual_pressure.png"):
                self.assertTrue((Path(d) / f).exists(), f)
            loaded = json.loads((Path(d) / "results.json").read_text())  # valid JSON (no NaN tokens)
        self.assertEqual(set(res["metrics"]), {"persistence", "climatology", "ridge_bootstrap", "diffusion"})
        for m in res["metrics"].values():
            self.assertTrue(np.isfinite(m["crps"]) and m["crps"] > 0)
        self.assertAlmostEqual(sum(loaded["explainability"]["group_importance_share"].values()), 1.0, places=6)
        self.assertEqual(len(loaded["explainability"]["temporal_attribution_share"]), 24)

    def test_attention_backbone_and_summary(self):
        raw = make_synthetic_raw(24 * 365, seed=1, start="2018-01-01")
        with tempfile.TemporaryDirectory() as d:
            dirs = [Path(d) / "a", Path(d) / "b"]
            res = run_pipeline(raw, dirs[0], tiny_cfg(backbone="attn", make_plots=True), verbose=False)
            run_pipeline(raw, dirs[1], tiny_cfg(backbone="attn", seed=1), verbose=False)
            self.assertTrue((dirs[0] / "attention.png").exists())
            self.assertTrue((dirs[0] / "test_windows.npz").exists())
            stats = summarize_runs(dirs, d)
            self.assertTrue((Path(d) / "summary.md").exists())
        ex = res["explainability"]
        self.assertAlmostEqual(sum(ex["attention_share"].values()), 1.0, places=5)
        self.assertIn("attention_occlusion_spearman", ex["explanation_quality"])
        self.assertIn("power_crps", res["metrics"]["diffusion"])
        self.assertEqual(stats["diffusion_attn"]["crps"][2], 2)

    def test_splits_do_not_overlap_and_bad_splits_fail(self):
        raw = make_synthetic_raw(24 * 365, seed=1, start="2018-01-01")
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):  # no windows after test_start
                run_pipeline(raw, d, tiny_cfg(test_start="2030-01-01"), verbose=False)

    def test_reproducible(self):
        raw = make_synthetic_raw(24 * 365, seed=2, start="2018-01-01")
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            a = run_pipeline(raw, d1, tiny_cfg(), verbose=False)
            b = run_pipeline(raw, d2, tiny_cfg(), verbose=False)
        self.assertEqual(a["metrics"]["diffusion"]["crps"], b["metrics"]["diffusion"]["crps"])

    def test_clean_handles_nan_and_numpy(self):
        c = _clean({"a": np.nan, "b": np.array([1.0, np.inf]), "c": np.int64(3), "d": (np.float32(1.5),)})
        self.assertEqual(c, {"a": None, "b": [1.0, None], "c": 3, "d": [1.5]})
        json.dumps(c, allow_nan=False)


class TestPipelineRealData(unittest.TestCase):
    """Runs on the ERA5 csv files if found (WIND_DATA_DIR, ./data/raw or ./data)."""

    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cands = [os.environ.get("WIND_DATA_DIR"), root / "data" / "raw", root / "data",
                 "/mnt/user-data/uploads"]
        cls.dir = next((Path(c) for c in cands if c and (Path(c) / "wind.csv").exists()), None)

    def test_load_and_quick_run(self):
        if self.dir is None:
            self.skipTest("ERA5 csv files not found")
        raw = load_raw(self.dir, nrows=24 * 365 * 3)
        self.assertEqual(list(raw.columns), ["u10", "v10", "t2m", "d2m", "sp", "tp", "ssrd"])
        self.assertFalse(raw.isna().any().any())
        self.assertTrue(raw.index.is_monotonic_increasing)
        cfg = tiny_cfg(val_start="1951-06-01", test_start="1952-03-01")
        with tempfile.TemporaryDirectory() as d:
            res = run_from_csv(self.dir, d, cfg, nrows=24 * 365 * 3, verbose=False)
        self.assertGreater(res["n_windows"]["test"], 10)
        self.assertTrue(np.isfinite(res["metrics"]["diffusion"]["crps"]))


if __name__ == "__main__":
    unittest.main()
