"""Conditional DDPM for wind-speed trajectories (pure NumPy, manual backprop)."""
from __future__ import annotations

import json

import numpy as np

from .nn import MLP, Adam, AttnDenoiser


def cosine_betas(T: int, s: float = 0.008) -> np.ndarray:
    x = np.linspace(0, T, T + 1)
    f = np.cos(((x / T) + s) / (1 + s) * np.pi / 2) ** 2
    ab = f / f[0]
    betas = 1.0 - ab[1:] / ab[:-1]
    return np.clip(betas, 1e-5, 0.999)


def time_embedding(t, dim: int) -> np.ndarray:
    if dim % 2:
        raise ValueError("time embedding dim must be even")
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    half = dim // 2
    freqs = np.exp(-np.log(1000.0) * np.arange(half) / half)
    ang = t[:, None] * freqs[None, :]
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)


class ConditionalDDPM:
    """epsilon-prediction DDPM: p(trajectory | conditioning vector)."""

    def __init__(self, horizon, cond_dim, T=100, hidden=256, n_layers=3,
                 time_dim=16, seed=0, dtype=np.float32, backbone="mlp", lookback=24, cal_dim=4):
        if horizon < 1 or cond_dim < 1 or T < 2 or n_layers < 1:
            raise ValueError("invalid model dimensions")
        if backbone not in ("mlp", "attn"):
            raise ValueError("backbone must be 'mlp' or 'attn'")
        self.cfg = dict(horizon=int(horizon), cond_dim=int(cond_dim), T=int(T),
                        hidden=int(hidden), n_layers=int(n_layers),
                        time_dim=int(time_dim), seed=int(seed),
                        dtype=np.dtype(dtype).name, backbone=backbone,
                        lookback=int(lookback), cal_dim=int(cal_dim))
        self.H, self.D, self.T, self.time_dim = horizon, cond_dim, T, time_dim
        self.dtype = np.dtype(dtype)
        rng = np.random.default_rng(seed)
        if backbone == "attn":
            F, rem = divmod(cond_dim - cal_dim, lookback)
            if rem or F < 1:
                raise ValueError("cond_dim must equal lookback * n_features + cal_dim")
            self.net = AttnDenoiser(horizon, lookback, F, cal_dim, time_dim, hidden, n_layers,
                                    rng, dtype=self.dtype)
        else:
            sizes = [horizon + cond_dim + time_dim] + [hidden] * n_layers + [horizon]
            self.net = MLP(sizes, rng, dtype=self.dtype)
        self.ema = [p.copy() for p in self.net.params()]

        self.betas = cosine_betas(T)
        self.alphas = 1.0 - self.betas
        self.ab = np.cumprod(self.alphas)
        self.ab_prev = np.concatenate([[1.0], self.ab[:-1]])
        self.post_var = self.betas * (1 - self.ab_prev) / (1 - self.ab)
        self.coef_x0 = self.betas * np.sqrt(self.ab_prev) / (1 - self.ab)
        self.coef_xt = (1 - self.ab_prev) * np.sqrt(self.alphas) / (1 - self.ab)
        self._temb = time_embedding(np.arange(T), time_dim).astype(self.dtype)

    # ------------------------------------------------------------------ training
    def _inp(self, xt, cond, t):
        return np.concatenate([xt, cond, self._temb[t]], axis=1).astype(self.dtype, copy=False)

    def _check(self, x0, cond):
        x0 = np.asarray(x0)
        cond = np.asarray(cond)
        if x0.ndim != 2 or x0.shape[1] != self.H:
            raise ValueError(f"x0 must be (N,{self.H})")
        if cond.ndim != 2 or cond.shape[1] != self.D or len(cond) != len(x0):
            raise ValueError(f"cond must be (N,{self.D}) with N matching x0")
        if not (np.isfinite(x0).all() and np.isfinite(cond).all()):
            raise ValueError("non-finite values in training data")
        return x0.astype(self.dtype), cond.astype(self.dtype)

    def _loss_on(self, x0, cond, rng):
        t = rng.integers(0, self.T, size=len(x0))
        eps = rng.standard_normal(x0.shape).astype(self.dtype)
        a = np.sqrt(self.ab[t])[:, None].astype(self.dtype)
        s = np.sqrt(1 - self.ab[t])[:, None].astype(self.dtype)
        xt = a * x0 + s * eps
        pred = self.net.forward(self._inp(xt, cond, t), train=False)
        return float(np.mean((pred - eps) ** 2))

    def val_loss(self, x0, cond, seed=123, repeats=4):
        x0, cond = self._check(x0, cond)
        rng = np.random.default_rng(seed)
        return float(np.mean([self._loss_on(x0, cond, rng) for _ in range(repeats)]))

    def fit(self, x0, cond, steps=2000, batch_size=256, lr=1e-3, clip=1.0,
            seed=0, log_every=200, val=None, warmup=100, ema_decay=0.999, verbose=True):
        """Train with Adam, cosine LR decay and EMA of the weights."""
        x0, cond = self._check(x0, cond)
        if steps < 1 or batch_size < 1:
            raise ValueError("steps and batch_size must be >= 1")
        rng = np.random.default_rng(seed)
        opt = Adam(self.net.params(), lr=lr)
        hist = {"step": [], "train_loss": [], "val_loss": []}
        run, cnt = 0.0, 0
        n = len(x0)
        for step in range(1, steps + 1):
            ix = rng.integers(0, n, size=min(batch_size, n))
            xb, cb = x0[ix], cond[ix]
            t = rng.integers(0, self.T, size=len(ix))
            eps = rng.standard_normal(xb.shape).astype(self.dtype)
            a = np.sqrt(self.ab[t])[:, None].astype(self.dtype)
            s = np.sqrt(1 - self.ab[t])[:, None].astype(self.dtype)
            xt = a * xb + s * eps
            pred = self.net.forward(self._inp(xt, cb, t), train=True)
            diff = pred - eps
            loss = float(np.mean(diff ** 2))
            if not np.isfinite(loss):
                raise FloatingPointError(f"loss became non-finite at step {step}")
            self.net.backward((2.0 / diff.size) * diff)
            frac = (step - 1) / max(1, steps - 1)
            cur_lr = lr * 0.5 * (1 + np.cos(np.pi * frac)) * min(1.0, step / max(1, warmup))
            opt.step(self.net.grads(), lr=max(cur_lr, 1e-6), clip=clip)
            d = min(ema_decay, (1 + step) / (10 + step))
            for e, p in zip(self.ema, self.net.params()):
                e *= d
                e += (1 - d) * p
            run += loss
            cnt += 1
            if step % log_every == 0 or step == steps:
                vl = self.val_loss(*val) if val is not None else float("nan")
                hist["step"].append(step)
                hist["train_loss"].append(run / cnt)
                hist["val_loss"].append(vl)
                if verbose:
                    print(f"  step {step:5d}/{steps}  train {run / cnt:.4f}  val {vl:.4f}", flush=True)
                run, cnt = 0.0, 0
        return hist

    # ------------------------------------------------------------------ sampling
    def sample(self, cond, n_samples=1, seed=0, common_noise=False, use_ema=True,
               clip_x0=5.0, chunk_rows=8192, return_attention=False):
        """Draw ``n_samples`` trajectories per condition -> (N, n_samples, H), normalised units.

        With ``return_attention`` (attn backbone only) also returns (N, heads, L) attention
        weights averaged over scenarios and denoising steps.
        """
        if return_attention and self.cfg["backbone"] != "attn":
            raise ValueError("attention weights need backbone='attn'")
        cond = np.asarray(cond)
        if cond.ndim != 2 or cond.shape[1] != self.D:
            raise ValueError(f"cond must be (N,{self.D})")
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if not np.isfinite(cond).all():
            raise ValueError("non-finite conditioning values")
        cond = cond.astype(self.dtype)
        N = len(cond)
        out = np.empty((N, n_samples, self.H), dtype=np.float64)
        attn = np.empty((N, self.net.h, self.net.L)) if return_attention else None
        per = max(1, chunk_rows // n_samples)
        backup = None
        if use_ema:
            backup = [p.copy() for p in self.net.params()]
            for p, e in zip(self.net.params(), self.ema):
                p[...] = e
        try:
            for s in range(0, N, per):
                rng = np.random.default_rng(seed if common_noise else [seed, s])
                out[s:s + per], a = self._sample_chunk(cond[s:s + per], n_samples, rng,
                                                       common_noise, clip_x0, return_attention)
                if return_attention:
                    attn[s:s + per] = a
        finally:
            if backup is not None:
                for p, b in zip(self.net.params(), backup):
                    p[...] = b
        return (out, attn) if return_attention else out

    def _sample_chunk(self, cond, n, rng, common, clip_x0, record=False):
        B = len(cond)
        acc = 0.0
        c = np.repeat(cond, n, axis=0)

        def noise():
            if common:
                return np.tile(rng.standard_normal((n, self.H)), (B, 1)).astype(self.dtype)
            return rng.standard_normal((B * n, self.H)).astype(self.dtype)

        x = noise()
        for t in range(self.T - 1, -1, -1):
            tt = np.full(B * n, t)
            eps = self.net.forward(self._inp(x, c, tt), train=False)
            if record:
                acc = acc + self.net.last_attn
            x0 = (x - np.sqrt(1 - self.ab[t]) * eps) / np.sqrt(self.ab[t])
            x0 = np.clip(x0, -clip_x0, clip_x0)
            mean = self.coef_x0[t] * x0 + self.coef_xt[t] * x
            x = mean + np.sqrt(self.post_var[t]) * noise() if t > 0 else mean
            x = x.astype(self.dtype, copy=False)
        x = x.reshape(B, n, self.H).astype(np.float64)
        if not record:
            return x, None
        return x, (acc / self.T).reshape(B, n, *acc.shape[1:]).mean(axis=1)

    # ------------------------------------------------------------------ persistence
    def save(self, path, extra=None):
        arrays = {f"p{i}": p for i, p in enumerate(self.net.params())}
        arrays.update({f"e{i}": e for i, e in enumerate(self.ema)})
        for k, v in (extra or {}).items():
            arrays[f"x_{k}"] = np.asarray(v)
        np.savez(path, cfg=json.dumps(self.cfg), **arrays)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        cfg = json.loads(str(z["cfg"]))
        cfg["dtype"] = np.dtype(cfg["dtype"])
        m = cls(**cfg)
        for i, p in enumerate(m.net.params()):
            p[...] = z[f"p{i}"]
        for i, e in enumerate(m.ema):
            e[...] = z[f"e{i}"]
        return m


class ScenarioGenerator:
    """Physical-units wrapper: history/calendar arrays -> wind-speed scenarios (m/s)."""

    def __init__(self, model: ConditionalDDPM, norm):
        self.model = model
        self.norm = norm

    def scenarios_from_arrays(self, hist, cal, n=20, seed=0, common_noise=False):
        cond = self.norm.cond(hist, cal)
        z = self.model.sample(cond, n, seed=seed, common_noise=common_noise)
        return np.clip(self.norm.denorm_target(z), 0.0, None)

    def scenarios(self, windows, n=20, seed=0, common_noise=False):
        return self.scenarios_from_arrays(windows.hist, windows.cal, n, seed, common_noise)

    def attention_from_arrays(self, hist, cal, n=8, seed=0):
        """(N, heads, L) attention over history hours (attn backbone), averaged over scenarios and steps."""
        cond = self.norm.cond(hist, cal)
        return self.model.sample(cond, n, seed=seed, common_noise=True, return_attention=True)[1]
