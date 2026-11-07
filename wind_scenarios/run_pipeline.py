#!/usr/bin/env python3
"""Run the full experiment on the ERA5 csv files.

    python run_pipeline.py --data-dir data/raw --out-dir outputs
    python run_pipeline.py --synthetic --quick      # smoke test, no data needed
    python run_pipeline.py --data-dir data/raw --backbones mlp attn --seeds 0 1 2   # full study
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from windgen.data import make_synthetic_raw  # noqa: E402
from windgen.data import load_raw  # noqa: E402
from windgen.pipeline import Config, run_pipeline, summarize_runs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--steps", type=int, default=None, help="diffusion training steps")
    ap.add_argument("--scenarios", type=int, default=None)
    ap.add_argument("--synthetic", action="store_true", help="use synthetic data")
    ap.add_argument("--quick", action="store_true", help="tiny settings for a fast smoke test")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--backbones", nargs="+", default=["mlp"], choices=["mlp", "attn"])
    a = ap.parse_args()

    cfg = Config()
    if a.quick:
        cfg.steps, cfg.hidden, cfg.T = 200, 64, 20
        cfg.max_train, cfg.max_val, cfg.max_test = 2000, 200, 100
        cfg.n_explain, cfg.n_perm, cfg.n_scenarios, cfg.log_every = 3, 3, 8, 100
    if a.steps:
        cfg.steps = a.steps
    if a.scenarios:
        cfg.n_scenarios = a.scenarios

    if a.synthetic:
        raw = make_synthetic_raw(24 * 365 * 4, seed=0, start="2017-01-01")
        cfg.val_start, cfg.test_start = "2019-07-01", "2020-07-01"
    else:
        raw = load_raw(a.data_dir)

    single = len(a.seeds) == 1 and len(a.backbones) == 1
    dirs = []
    for bb in a.backbones:
        for seed in a.seeds:
            cfg.backbone, cfg.seed = bb, seed
            d = Path(a.out_dir) if single else Path(a.out_dir) / f"{bb}_seed{seed}"
            print(f"=== {bb}, seed {seed} -> {d}", flush=True)
            run_pipeline(raw, d, cfg)
            dirs.append(d)
    if not single:
        summarize_runs(dirs, a.out_dir)
        print((Path(a.out_dir) / "summary.md").read_text())


if __name__ == "__main__":
    main()
