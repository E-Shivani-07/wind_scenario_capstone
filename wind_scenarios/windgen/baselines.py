"""Baseline scenario generators with the same interface as the diffusion generator."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Normalizer, Windows


class PersistenceBaseline:
    """Deterministic: tomorrow's trajectory = the previous 24 h of wind speed."""

    name = "persistence"

    def fit(self, train: Windows):
        return self

    def scenarios(self, w: Windows, n=1, seed=0):
        H = w.target.shape[1]
        L = w.hist.shape[1]
        if L < H:
            raise ValueError("persistence needs lookback >= horizon")
        last = w.hist[:, L - H:, 0].astype(np.float64)
        return last[:, None, :]  # a single scenario


class ClimatologyBaseline:
    """Bootstrap whole trajectories from training windows with the same month and hour of day."""

    name = "climatology"

    def __init__(self, min_group=20):
        self.min_group = min_group
        self.groups = {}
        self.by_hour = {}

    def fit(self, train: Windows):
        idx = pd.DatetimeIndex(train.origin)
        month, hour = idx.month.to_numpy(), idx.hour.to_numpy()
        tgt = train.target.astype(np.float64)
        self.groups = {}
        self.by_hour = {}
        for h in np.unique(hour):
            self.by_hour[int(h)] = tgt[hour == h]
            for m in np.unique(month):
                sel = (hour == h) & (month == m)
                if sel.sum() >= self.min_group:
                    self.groups[(int(m), int(h))] = tgt[sel]
        return self

    def scenarios(self, w: Windows, n=20, seed=0):
        if not self.by_hour:
            raise RuntimeError("fit() first")
        idx = pd.DatetimeIndex(w.origin)
        rng = np.random.default_rng(seed)
        out = np.empty((len(w), n, w.target.shape[1]))
        for i, (m, h) in enumerate(zip(idx.month, idx.hour)):
            pool = self.groups.get((int(m), int(h)))
            if pool is None:
                pool = self.by_hour.get(int(h))
            if pool is None:
                pool = np.concatenate(list(self.by_hour.values()))
            out[i] = pool[rng.integers(0, len(pool), size=n)]
        return out


class RidgeResidualBaseline:
    """Linear (ridge) conditional mean + bootstrap of training residual trajectories."""

    name = "ridge_bootstrap"

    def __init__(self, norm: Normalizer, alpha=10.0):
        self.norm = norm
        self.alpha = alpha
        self.W = None
        self.resid = None

    def _X(self, w: Windows):
        c = self.norm.cond(w.hist, w.cal)
        return np.concatenate([c, np.ones((len(c), 1))], axis=1)

    def fit(self, train: Windows):
        X = self._X(train)
        Y = self.norm.norm_target(train.target)
        A = X.T @ X + self.alpha * np.eye(X.shape[1])
        A[-1, -1] -= self.alpha  # do not penalise the intercept
        self.W = np.linalg.solve(A, X.T @ Y)
        self.resid = Y - X @ self.W
        return self

    def scenarios(self, w: Windows, n=20, seed=0):
        if self.W is None:
            raise RuntimeError("fit() first")
        rng = np.random.default_rng(seed)
        mean = self._X(w) @ self.W
        r = self.resid[rng.integers(0, len(self.resid), size=(len(w), n))]
        return np.clip(self.norm.denorm_target(mean[:, None, :] + r), 0.0, None)
