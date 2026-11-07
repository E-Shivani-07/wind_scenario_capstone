import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from windgen.data import (Normalizer, build_features, make_synthetic_raw,  # noqa: E402
                          make_windows)


def small_windows(n_hours=24 * 120, seed=0, stride=3):
    feats = build_features(make_synthetic_raw(n_hours, seed=seed))
    return feats, make_windows(feats, 24, 24, stride=stride)
