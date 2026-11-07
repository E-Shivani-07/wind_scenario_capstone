"""Explainability for generated scenarios.

* Shapley values over feature *groups* (permutation sampling, interventional baseline).
  The explained quantity is the mean generated wind speed over scenarios and horizon;
  common random numbers make the value function deterministic.
* Temporal (lag) attribution by occlusion - the attention-style analysis of this
  framework (the denoiser is an MLP, so there are no literal attention weights).
* Counterfactual explanations: shift one feature group and compare scenario ensembles,
  or search for the smallest shift that achieves a target change.
* Attention analysis (attn backbone): the denoiser's cross-attention weights over history hours.
* Explainability metrics: deletion faithfulness of SHAP, seed stability, attention/occlusion agreement.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .data import FEATURES, GROUPS, Windows

GROUP_NAMES = list(GROUPS) + ["calendar"]

# units in which counterfactual deltas are expressed
DELTA_UNITS = {
    "wind_speed": "relative change (0.1 = +10 %)",
    "wind_direction": "rotation in degrees",
    "temperature": "K",
    "humidity": "% points",
    "pressure": "Pa (100 Pa = 1 hPa)",
    "precipitation": "added to log(1+mm/h)",
    "radiation": "relative change (0.1 = +10 %)",
}


def shift_group(hist, group, delta):
    """Return a copy of hist (..., L, F) with ``group`` shifted by ``delta`` (see DELTA_UNITS)."""
    if group not in GROUPS:
        raise ValueError(f"unknown group '{group}'; choose from {list(GROUPS)}")
    h = np.array(hist, dtype=np.float64, copy=True)
    cols = GROUPS[group]
    if group in ("wind_speed", "radiation"):
        h[..., cols] = h[..., cols] * (1.0 + delta)
        h[..., cols] = np.maximum(h[..., cols], 0.0)
    elif group == "wind_direction":
        s, c = h[..., 1], h[..., 2]
        th = np.arctan2(s, c) + np.radians(delta)
        h[..., 1], h[..., 2] = np.sin(th), np.cos(th)
    elif group == "humidity":
        h[..., cols] = np.clip(h[..., cols] + delta, 0.0, 100.0)
    elif group == "precipitation":
        h[..., cols] = np.maximum(h[..., cols] + delta, 0.0)
    else:  # temperature, pressure
        h[..., cols] = h[..., cols] + delta
    return h


class ScenarioExplainer:
    def __init__(self, generator, background: Windows, n_scenarios=8, seed=0, n_background=64):
        if len(background) < 1:
            raise ValueError("background must contain at least one window")
        self.gen = generator
        self.n_scen = int(n_scenarios)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        k = min(n_background, len(background))
        ix = np.sort(rng.choice(len(background), size=k, replace=False))
        self.bg_hist = background.hist[ix].astype(np.float64)
        self.bg_cal = background.cal[ix].astype(np.float64)
        self.bg_mean_hist = self.bg_hist.mean(axis=0)  # (L, F)

    # value function: mean generated wind speed (m/s), common random numbers
    def _f(self, hist, cal):
        s = self.gen.scenarios_from_arrays(hist, cal, n=self.n_scen, seed=self.seed, common_noise=True)
        return s.mean(axis=(1, 2))

    def predict_mean(self, hist, cal):
        return self._f(np.asarray(hist, dtype=np.float64), np.asarray(cal, dtype=np.float64))

    # ------------------------------------------------------------------ SHAP
    def shapley(self, hist, cal, n_perm=16, seed=0):
        """Group Shapley values for one window. hist (L,F), cal (4,)."""
        hist = np.asarray(hist, dtype=np.float64)
        cal = np.asarray(cal, dtype=np.float64)
        if hist.ndim != 2 or cal.ndim != 1:
            raise ValueError("hist must be (L,F) and cal (4,)")
        rng = np.random.default_rng(seed)
        G = len(GROUP_NAMES)
        phi = np.zeros(G)
        base_vals, full_vals = [], []
        for _ in range(n_perm):
            b = rng.integers(len(self.bg_hist))
            order = rng.permutation(G)
            cur_h, cur_c = self.bg_hist[b].copy(), self.bg_cal[b].copy()
            hs = np.empty((G + 1,) + hist.shape)
            cs = np.empty((G + 1,) + cal.shape)
            hs[0], cs[0] = cur_h, cur_c
            for k, g in enumerate(order):
                name = GROUP_NAMES[g]
                if name == "calendar":
                    cur_c = cal.copy()
                else:
                    cur_h[:, GROUPS[name]] = hist[:, GROUPS[name]]
                hs[k + 1], cs[k + 1] = cur_h, cur_c
            v = self._f(hs, cs)
            for k, g in enumerate(order):
                phi[g] += v[k + 1] - v[k]
            base_vals.append(v[0])
            full_vals.append(v[-1])
        phi /= n_perm
        return {"names": GROUP_NAMES, "phi": phi, "base_value": float(np.mean(base_vals)),
                "prediction": float(np.mean(full_vals))}

    def global_importance(self, windows: Windows, n_instances=20, n_perm=12, seed=0):
        """Mean |Shapley| per group over evenly spaced windows."""
        ix = np.unique(np.linspace(0, len(windows) - 1, min(n_instances, len(windows))).astype(int))
        phis = np.array([self.shapley(windows.hist[i], windows.cal[i], n_perm=n_perm, seed=seed + int(i))["phi"]
                         for i in ix])
        mean_abs = np.abs(phis).mean(axis=0)
        total = max(mean_abs.sum(), 1e-12)
        return {"names": GROUP_NAMES, "mean_abs_phi": mean_abs, "share": mean_abs / total,
                "mean_phi": phis.mean(axis=0), "n_instances": len(ix), "phi_all": phis, "index": ix}

    # ------------------------------------------------------------------ temporal
    def temporal_attribution(self, windows: Windows, n_instances=20):
        """Occlude every feature of one lag hour at a time -> |change in mean prediction| per lag."""
        ix = np.unique(np.linspace(0, len(windows) - 1, min(n_instances, len(windows))).astype(int))
        L = windows.hist.shape[1]
        acc = np.zeros(L)
        for i in ix:
            h = windows.hist[i].astype(np.float64)
            c = windows.cal[i].astype(np.float64)
            hs = np.repeat(h[None], L + 1, axis=0)
            for lag in range(L):
                hs[lag + 1, lag, :] = self.bg_mean_hist[lag]
            v = self._f(hs, np.repeat(c[None], L + 1, axis=0))
            acc += np.abs(v[1:] - v[0])
        acc /= len(ix)
        # index 0 = oldest hour in history, L-1 = the hour just before the forecast origin
        return {"lag_hours_before_origin": np.arange(L, 0, -1), "importance": acc,
                "share": acc / max(acc.sum(), 1e-12)}

    # ------------------------------------------------------------------ attention
    def attention_by_lag(self, windows: Windows, n_instances=20):
        """Mean cross-attention weight per history hour (needs the attn backbone)."""
        ix = np.unique(np.linspace(0, len(windows) - 1, min(n_instances, len(windows))).astype(int))
        a = self.gen.attention_from_arrays(windows.hist[ix], windows.cal[ix], n=self.n_scen, seed=self.seed)
        per_head = a.mean(axis=0)  # (heads, L), each row sums to 1
        return {"lag_hours_before_origin": np.arange(a.shape[2], 0, -1), "share": per_head.mean(axis=0),
                "per_head": per_head}

    # ------------------------------------------------------------------ explanation quality
    def deletion_faithfulness(self, windows: Windows, importance):
        """Replace each group by its background mean; do the groups with larger |SHAP| move the output more?

        Returns the mean Spearman rank correlation between |phi| and the deletion effect
        (1 = perfectly faithful ranking) and the effect of deleting the top-|SHAP| group relative
        to the average group (> 1 = the top group really matters most).
        """
        G = len(GROUP_NAMES)
        rhos, ratios = [], []
        for phi, i in zip(importance["phi_all"], importance["index"]):
            hs = np.repeat(windows.hist[i][None].astype(np.float64), G + 1, axis=0)
            cs = np.repeat(windows.cal[i][None].astype(np.float64), G + 1, axis=0)
            for g, name in enumerate(GROUP_NAMES):
                if name == "calendar":
                    cs[g + 1] = self.bg_cal.mean(axis=0)
                else:
                    hs[g + 1][:, GROUPS[name]] = self.bg_mean_hist[:, GROUPS[name]]
            v = self._f(hs, cs)
            effect = np.abs(v[1:] - v[0])
            rhos.append(stats.spearmanr(np.abs(phi), effect).statistic)
            ratios.append(effect[np.argmax(np.abs(phi))] / max(effect.mean(), 1e-12))
        return {"spearman": float(np.nanmean(rhos)), "top1_deletion_ratio": float(np.mean(ratios))}

    # ------------------------------------------------------------------ counterfactual
    def counterfactual(self, hist, cal, group, delta, n=None):
        """Compare scenarios for the factual window and for ``group`` shifted by ``delta``."""
        n = n or self.n_scen
        hist = np.asarray(hist, dtype=np.float64)[None]
        cal = np.asarray(cal, dtype=np.float64)[None]
        cf = shift_group(hist, group, delta)
        base = self.gen.scenarios_from_arrays(hist, cal, n=n, seed=self.seed, common_noise=True)[0]
        alt = self.gen.scenarios_from_arrays(cf, cal, n=n, seed=self.seed, common_noise=True)[0]
        return {"group": group, "delta": float(delta), "unit": DELTA_UNITS[group],
                "base": base, "counterfactual": alt,
                "mean_shift": float(alt.mean() - base.mean()),
                "wasserstein": float(stats.wasserstein_distance(base.ravel(), alt.ravel()))}

    def find_counterfactual(self, hist, cal, group, target_shift, deltas=None):
        """Smallest |delta| (from a grid) whose mean-wind change reaches ``target_shift`` (m/s).

        Returns the counterfactual dict, or None if no delta on the grid is sufficient.
        """
        if target_shift == 0:
            raise ValueError("target_shift must be non-zero")
        if deltas is None:
            deltas = {"pressure": np.arange(-1500, 1501, 100), "temperature": np.arange(-5, 5.1, 0.5),
                      "humidity": np.arange(-30, 30.1, 5), "wind_speed": np.arange(-0.5, 0.51, 0.05),
                      "wind_direction": np.arange(-90, 91, 15), "precipitation": np.arange(0, 3.1, 0.25),
                      "radiation": np.arange(-0.5, 0.51, 0.05)}[group]
        deltas = sorted([d for d in deltas if d != 0], key=abs)
        for d in deltas:
            res = self.counterfactual(hist, cal, group, d)
            if np.sign(res["mean_shift"]) == np.sign(target_shift) and abs(res["mean_shift"]) >= abs(target_shift):
                return res
        return None


def rank_agreement(a, b) -> float:
    """Spearman rank correlation of two importance vectors (stability / agreement metric)."""
    return float(stats.spearmanr(np.asarray(a), np.asarray(b)).statistic)
