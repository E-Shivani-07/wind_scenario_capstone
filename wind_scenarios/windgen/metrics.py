"""Scenario evaluation: accuracy, probabilistic skill, similarity, diversity, calibration."""
from __future__ import annotations

import numpy as np
from scipy import stats


def _check(obs, ens):
    obs = np.asarray(obs, dtype=np.float64)
    ens = np.asarray(ens, dtype=np.float64)
    if obs.ndim != 2 or ens.ndim != 3 or ens.shape[0] != obs.shape[0] or ens.shape[2] != obs.shape[1]:
        raise ValueError("obs must be (N,H) and ens (N,M,H)")
    if not (np.isfinite(obs).all() and np.isfinite(ens).all()):
        raise ValueError("non-finite values in obs/ens")
    return obs, ens


def crps_ensemble(obs, ens) -> float:
    """Mean CRPS over all (window, hour); exact ensemble estimator (equals MAE for M=1)."""
    obs, ens = _check(obs, ens)
    M = ens.shape[1]
    term1 = np.abs(ens - obs[:, None, :]).mean(axis=1)
    srt = np.sort(ens, axis=1)
    w = (2 * np.arange(1, M + 1) - M - 1)[None, :, None]
    mean_abs_diff = 2.0 / M ** 2 * (w * srt).sum(axis=1)
    return float((term1 - 0.5 * mean_abs_diff).mean())


def _pair_norm_mean(ens, chunk=64):
    """E||X - X'|| per window over all ordered pairs (i, j incl. i=j)."""
    N, M, _ = ens.shape
    out = np.empty(N)
    for s in range(0, N, chunk):
        e = ens[s:s + chunk]
        d = np.sqrt(((e[:, :, None, :] - e[:, None, :, :]) ** 2).sum(-1))
        out[s:s + chunk] = d.mean(axis=(1, 2))
    return out


def energy_score(obs, ens) -> float:
    """Multivariate (whole-trajectory) energy score; lower is better."""
    obs, ens = _check(obs, ens)
    t1 = np.sqrt(((ens - obs[:, None, :]) ** 2).sum(-1)).mean(axis=1)
    return float((t1 - 0.5 * _pair_norm_mean(ens)).mean())


def point_errors(obs, ens) -> dict:
    obs, ens = _check(obs, ens)
    err = ens.mean(axis=1) - obs
    return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean()))}


def pit_ranks(obs, ens, seed=0):
    """Randomised rank of the observation within the ensemble (0..M)."""
    obs, ens = _check(obs, ens)
    rng = np.random.default_rng(seed)
    less = (ens < obs[:, None, :]).sum(axis=1)
    equal = (ens == obs[:, None, :]).sum(axis=1)
    return (less + np.floor(rng.random(less.shape) * (equal + 1))).astype(int).ravel()


def rank_histogram(obs, ens, seed=0):
    M = np.asarray(ens).shape[1]
    r = pit_ranks(obs, ens, seed)
    return np.bincount(r, minlength=M + 1) / len(r)


def coverage(obs, ens, level=0.9, seed=0) -> float:
    """Fraction of observations inside the central ``level`` predictive interval.

    Uses randomised PIT values u=(rank+U)/(M+1), which removes the small-ensemble bias of
    empirical-quantile intervals: a calibrated ensemble gives ``level`` for any M.
    """
    obs, ens = _check(obs, ens)
    if not 0 < level < 1:
        raise ValueError("level must be in (0,1)")
    M = ens.shape[1]
    rng = np.random.default_rng(seed + 1)
    r = pit_ranks(obs, ens, seed)
    u = (r + rng.random(r.shape)) / (M + 1)
    return float((np.abs(u - 0.5) <= level / 2).mean())


def calibration_error(obs, ens, levels=(0.5, 0.8, 0.9, 0.95)) -> float:
    return float(np.mean([abs(coverage(obs, ens, l) - l) for l in levels]))


def spread_skill(obs, ens) -> float:
    """mean ensemble std / RMSE of the ensemble mean; ~1 (times a small-M factor) if calibrated."""
    obs, ens = _check(obs, ens)
    if ens.shape[1] < 2:
        return float("nan")
    spread = ens.std(axis=1, ddof=1).mean()
    rmse = np.sqrt(((ens.mean(axis=1) - obs) ** 2).mean())
    return float(spread / max(rmse, 1e-12))


def diversity(ens) -> float:
    """Mean RMS difference (m/s per hour) between distinct scenarios of the same window."""
    ens = np.asarray(ens, dtype=np.float64)
    N, M, H = ens.shape
    if M < 2:
        return 0.0
    mean_all = _pair_norm_mean(ens)  # includes zero diagonal
    mean_distinct = mean_all * M / (M - 1)
    return float((mean_distinct / np.sqrt(H)).mean())


def ks_wasserstein(obs, ens) -> dict:
    obs, ens = _check(obs, ens)
    ks = stats.ks_2samp(obs.ravel(), ens.ravel())
    return {"ks": float(ks.statistic), "wasserstein": float(stats.wasserstein_distance(obs.ravel(), ens.ravel()))}


def acf(traj, max_lag=6) -> np.ndarray:
    """Pooled lag-k autocorrelation of (n, H) trajectories."""
    traj = np.asarray(traj, dtype=np.float64)
    out = np.empty(max_lag)
    for k in range(1, max_lag + 1):
        a, b = traj[:, :-k].ravel(), traj[:, k:].ravel()
        out[k - 1] = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else 0.0
    return out


def acf_distance(obs, ens, max_lag=6) -> float:
    obs, ens = _check(obs, ens)
    gen = ens.reshape(-1, ens.shape[2])
    return float(np.abs(acf(obs, max_lag) - acf(gen, max_lag)).mean())


def ramp_wasserstein(obs, ens) -> float:
    """Distance between distributions of hourly changes (ramps)."""
    obs, ens = _check(obs, ens)
    return float(stats.wasserstein_distance(np.diff(obs, axis=1).ravel(), np.diff(ens, axis=2).ravel()))


def evaluate_scenarios(obs, ens) -> dict:
    """All metrics for one generator. NaN-valued entries are returned as None."""
    obs, ens = _check(obs, ens)
    res = {"crps": crps_ensemble(obs, ens), "energy_score": energy_score(obs, ens)}
    res.update(point_errors(obs, ens))
    res.update(ks_wasserstein(obs, ens))
    res["acf_distance"] = acf_distance(obs, ens)
    res["ramp_wasserstein"] = ramp_wasserstein(obs, ens)
    res["diversity"] = diversity(ens)
    res["spread_skill"] = spread_skill(obs, ens)
    if ens.shape[1] >= 2:
        res["calibration_error"] = calibration_error(obs, ens)
        res["coverage_90"] = coverage(obs, ens, 0.9)
    else:
        res["calibration_error"] = None
        res["coverage_90"] = None
    return {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in res.items()}
