"""Data loading, feature engineering and windowing for wind scenario generation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Model input features (per hour). precipitation and radiation are transformed
# to tame their heavy tails: log1p(mm/h) and MJ/m2/h.
FEATURES = [
    "wind_speed",     # m/s, sqrt(u10^2 + v10^2)
    "wind_dir_sin",   # sin(arctan2(v10, u10))
    "wind_dir_cos",   # cos(arctan2(v10, u10))
    "t2m",            # K
    "rh",             # %, from t2m and d2m (Magnus formula)
    "sp",             # Pa
    "precip_log1p",   # log(1 + mm/h)
    "ssrd_mj",        # MJ/m2 in the hour
]
GROUPS = {
    "wind_speed": [0],
    "wind_direction": [1, 2],
    "temperature": [3],
    "humidity": [4],
    "pressure": [5],
    "precipitation": [6],
    "radiation": [7],
}
CAL_DIM = 4  # sin/cos of hour-of-day and day-of-year at the forecast origin

RAW_FILES = {
    "wind": ("wind.csv", ["u10", "v10"]),
    "temperature": ("temperature.csv", ["t2m", "d2m"]),
    "pressure": ("pressure.csv", ["sp", "tp"]),
    "radiation": ("radiation.csv", ["ssrd"]),
}
RAW_COLUMNS = ["u10", "v10", "t2m", "d2m", "sp", "tp", "ssrd"]


def load_raw(data_dir, nrows=None) -> pd.DataFrame:
    """Load and inner-join the four ERA5 csv files on ``valid_time``."""
    parts = []
    for _, (fname, cols) in RAW_FILES.items():
        path = Path(data_dir) / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing data file: {path}")
        df = pd.read_csv(path, usecols=["valid_time", *cols], nrows=nrows,
                         parse_dates=["valid_time"])
        df = df.drop_duplicates("valid_time").set_index("valid_time").sort_index()
        parts.append(df)
    raw = parts[0].join(parts[1:], how="inner")
    raw = raw.dropna()
    if raw.empty:
        raise ValueError("Joined dataset is empty; check the time columns of the csv files.")
    return raw[RAW_COLUMNS]


def relative_humidity(t2m_k, d2m_k):
    """Relative humidity in % from air and dew-point temperature in Kelvin."""
    t = np.asarray(t2m_k, dtype=float) - 273.15
    d = np.asarray(d2m_k, dtype=float) - 273.15
    a, b = 17.62, 243.12
    es = np.exp(a * t / (b + t))
    ed = np.exp(a * d / (b + d))
    return np.clip(100.0 * ed / es, 0.0, 100.0)


def direction_degrees(u, v):
    """Meteorological wind direction (degrees the wind blows *from*, 0=N)."""
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Engineer the model features from raw ERA5 columns."""
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"raw data is missing columns: {missing}")
    raw = raw.dropna()
    u = raw["u10"].to_numpy(float)
    v = raw["v10"].to_numpy(float)
    theta = np.arctan2(v, u)
    feats = pd.DataFrame(
        {
            "wind_speed": np.hypot(u, v),
            "wind_dir_sin": np.sin(theta),
            "wind_dir_cos": np.cos(theta),
            "t2m": raw["t2m"].to_numpy(float),
            "rh": relative_humidity(raw["t2m"], raw["d2m"]),
            "sp": raw["sp"].to_numpy(float),
            "precip_log1p": np.log1p(np.clip(raw["tp"].to_numpy(float) * 1000.0, 0, None)),
            "ssrd_mj": np.clip(raw["ssrd"].to_numpy(float), 0, None) / 1e6,
        },
        index=raw.index,
    )
    return feats[FEATURES]


def calendar_features(times) -> np.ndarray:
    """(N, 4) array: sin/cos of hour-of-day and of day-of-year."""
    idx = pd.DatetimeIndex(times)
    hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    doy = idx.day_of_year.to_numpy().astype(float)
    return np.stack(
        [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
         np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)],
        axis=1,
    )


@dataclass
class Windows:
    hist: np.ndarray    # (N, L, F) past features (L hours before the origin)
    cal: np.ndarray     # (N, 4) calendar features of the origin hour
    target: np.ndarray  # (N, H) wind speed (m/s) for the H hours from the origin
    origin: np.ndarray  # (N,) datetime64 forecast origin

    def __len__(self):
        return len(self.target)

    def subset(self, ix) -> "Windows":
        return Windows(self.hist[ix], self.cal[ix], self.target[ix], self.origin[ix])


def make_windows(feats: pd.DataFrame, lookback=24, horizon=24, start=None, end=None,
                 stride=1, max_windows=None, seed=0) -> Windows:
    """Cut (history, target) windows whose origin lies in [start, end).

    Windows spanning a gap in the hourly time axis are discarded.
    """
    if lookback < 1 or horizon < 1 or stride < 1:
        raise ValueError("lookback, horizon and stride must be >= 1")
    arr = feats[FEATURES].to_numpy(dtype=np.float32)
    times = feats.index.values
    n = len(arr)
    if n < lookback + horizon:
        raise ValueError("Not enough data for a single window.")
    idx = np.arange(lookback, n - horizon + 1)
    mask = np.ones(len(idx), dtype=bool)
    if start is not None:
        mask &= times[idx] >= np.datetime64(start)
    if end is not None:
        mask &= times[idx] < np.datetime64(end)
    span = times[idx + horizon - 1] - times[idx - lookback]
    mask &= span == np.timedelta64(lookback + horizon - 1, "h")
    idx = idx[mask][::stride]
    if max_windows is not None and len(idx) > max_windows:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=max_windows, replace=False))
    if len(idx) == 0:
        raise ValueError("No valid windows in the requested period.")
    hist = arr[idx[:, None] + np.arange(-lookback, 0)[None, :]]
    target = arr[:, 0][idx[:, None] + np.arange(horizon)[None, :]]
    origin = times[idx]
    return Windows(hist=hist, cal=calendar_features(origin).astype(np.float32),
                   target=target.astype(np.float32), origin=origin)


class Normalizer:
    """Standardises history features and the wind-speed target (fit on train only)."""

    def __init__(self):
        self.feat_mean = None
        self.feat_std = None
        self.t_mean = 0.0
        self.t_std = 1.0

    def fit(self, w: Windows) -> "Normalizer":
        flat = w.hist.reshape(-1, w.hist.shape[-1]).astype(np.float64)
        self.feat_mean = flat.mean(0)
        self.feat_std = np.maximum(flat.std(0), 1e-6)
        self.t_mean = float(w.target.mean())
        self.t_std = float(max(w.target.std(), 1e-6))
        return self

    def cond(self, hist, cal) -> np.ndarray:
        """Conditioning vector: standardised flattened history + calendar."""
        hist = np.asarray(hist, dtype=np.float64)
        cal = np.asarray(cal, dtype=np.float64)
        if hist.ndim != 3 or cal.ndim != 2 or len(hist) != len(cal):
            raise ValueError("hist must be (N,L,F) and cal (N,4) with equal N")
        z = (hist - self.feat_mean) / self.feat_std
        return np.concatenate([z.reshape(len(z), -1), cal], axis=1)

    def norm_target(self, y):
        return (np.asarray(y, dtype=np.float64) - self.t_mean) / self.t_std

    def denorm_target(self, z):
        return np.asarray(z, dtype=np.float64) * self.t_std + self.t_mean

    def state(self) -> dict:
        return {"feat_mean": self.feat_mean, "feat_std": self.feat_std,
                "t_mean": np.array(self.t_mean), "t_std": np.array(self.t_std)}

    @classmethod
    def from_state(cls, s) -> "Normalizer":
        n = cls()
        n.feat_mean = np.asarray(s["feat_mean"])
        n.feat_std = np.asarray(s["feat_std"])
        n.t_mean = float(s["t_mean"])
        n.t_std = float(s["t_std"])
        return n


def make_synthetic_raw(n_hours=24 * 365 * 3, seed=0, start="2018-01-01") -> pd.DataFrame:
    """Synthetic ERA5-like data with diurnal cycles and pressure-wind coupling (tests/demo)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_hours, freq="h")
    hour = idx.hour.to_numpy()
    doy = idx.day_of_year.to_numpy()

    def ar1(phi, sigma):
        e = rng.standard_normal(n_hours) * sigma
        x = np.zeros(n_hours)
        for i in range(1, n_hours):
            x[i] = phi * x[i - 1] + e[i]
        return x

    p_anom = ar1(0.995, 30.0)
    sp = 100200.0 + p_anom
    base = 2.8 + 0.9 * np.sin(2 * np.pi * (hour - 9) / 24) + 0.6 * np.sin(2 * np.pi * doy / 365.25)
    speed = np.clip(base - 0.004 * p_anom + ar1(0.9, 0.25), 0.05, None)
    theta = 1.0 + 0.5 * ar1(0.98, 0.1)
    u, v = speed * np.cos(theta), speed * np.sin(theta)
    t2m = 301 + 4 * np.sin(2 * np.pi * (hour - 9) / 24) + 2 * np.sin(2 * np.pi * doy / 365.25) + ar1(0.97, 0.15)
    d2m = np.minimum(t2m - 7.5 + ar1(0.97, 0.1), t2m - 0.1)
    rain = np.where(rng.random(n_hours) < 0.04, rng.exponential(0.002, n_hours), 0.0)
    sun = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None) * 2.5e6 * (0.7 + 0.3 * rng.random(n_hours))
    return pd.DataFrame({"u10": u, "v10": v, "t2m": t2m, "d2m": d2m, "sp": sp,
                         "tp": rain, "ssrd": sun}, index=idx.rename("valid_time"))
