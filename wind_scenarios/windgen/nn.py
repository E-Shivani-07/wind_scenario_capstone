"""Minimal NumPy neural-network toolkit (MLP with SiLU, manual backprop, Adam)."""
from __future__ import annotations

import numpy as np
from scipy.special import expit


class Dense:
    def __init__(self, n_in, n_out, rng, dtype=np.float64, scale=1.0):
        std = np.sqrt(2.0 / n_in) * scale
        self.W = (rng.standard_normal((n_in, n_out)) * std).astype(dtype)
        self.b = np.zeros(n_out, dtype=dtype)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x = None

    def forward(self, x, train=True):
        if train:
            self._x = x
        return x @ self.W + self.b

    def backward(self, g):
        self.dW[...] = self._x.T @ g
        self.db[...] = g.sum(axis=0)
        return g @ self.W.T


class MLP:
    """Fully connected network with SiLU activations on hidden layers."""

    def __init__(self, sizes, rng, dtype=np.float64, out_scale=0.1):
        if len(sizes) < 2:
            raise ValueError("sizes needs at least input and output dims")
        self.sizes = list(sizes)
        self.dtype = np.dtype(dtype)
        self.layers = []
        for i in range(len(sizes) - 1):
            last = i == len(sizes) - 2
            self.layers.append(Dense(sizes[i], sizes[i + 1], rng, dtype,
                                     scale=out_scale if last else 1.0))
        self._pre = []

    def forward(self, x, train=True):
        h = np.asarray(x, dtype=self.dtype)
        self._pre = []
        for i, layer in enumerate(self.layers):
            a = layer.forward(h, train)
            if i < len(self.layers) - 1:
                if train:
                    self._pre.append(a)
                h = a * expit(a)
            else:
                h = a
        return h

    def backward(self, gout):
        g = np.asarray(gout, dtype=self.dtype)
        for i in reversed(range(len(self.layers))):
            g = self.layers[i].backward(g)
            if i > 0:
                a = self._pre[i - 1]
                s = expit(a)
                g = g * (s * (1.0 + a * (1.0 - s)))
        return g

    def params(self):
        out = []
        for l in self.layers:
            out += [l.W, l.b]
        return out

    def grads(self):
        out = []
        for l in self.layers:
            out += [l.dW, l.db]
        return out


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads, lr=None, clip=None):
        lr = self.lr if lr is None else lr
        if clip is not None:
            norm = np.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads))
            if norm > clip:
                grads = [g * (clip / (norm + 1e-12)) for g in grads]
        self.t += 1
        c1 = 1 - self.b1 ** self.t
        c2 = 1 - self.b2 ** self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p -= (lr * (m / c1) / (np.sqrt(v / c2) + self.eps)).astype(p.dtype)


class AttnDenoiser:
    """Cross-attention denoiser: the noisy trajectory, calendar and step embedding form the query;
    the L history hours are the keys/values. History reaches the MLP *only* through attention,
    so the attention weights show which past hours the model actually reads.

    Input layout matches the MLP denoiser: [x_t (H) | history (L*F) | calendar (C) | t-emb].
    """

    def __init__(self, H, L, F, cal_dim, time_dim, hidden, n_layers, rng, dtype=np.float64,
                 heads=4, d_head=16, pos_dim=8):
        self.H, self.L, self.F, self.C, self.time_dim = H, L, F, cal_dim, time_dim
        self.h, self.d = heads, d_head
        self.dtype = np.dtype(dtype)
        q_dim, k_dim, hd = H + cal_dim + time_dim, F + pos_dim, heads * d_head
        self.Wq = (rng.standard_normal((q_dim, hd)) / np.sqrt(q_dim)).astype(self.dtype)
        self.Wk = (rng.standard_normal((k_dim, hd)) / np.sqrt(k_dim)).astype(self.dtype)
        self.Wv = (rng.standard_normal((k_dim, hd)) / np.sqrt(k_dim)).astype(self.dtype)
        self.dWq, self.dWk, self.dWv = (np.zeros_like(w) for w in (self.Wq, self.Wk, self.Wv))
        lag = np.arange(L)[:, None]
        freqs = np.exp(-np.log(100.0) * np.arange(pos_dim // 2) / max(1, pos_dim // 2))[None]
        self.pos = np.concatenate([np.sin(lag * freqs), np.cos(lag * freqs)], 1).astype(self.dtype)
        self.mlp = MLP([q_dim + hd] + [hidden] * n_layers + [H], rng, dtype=self.dtype)
        self.last_attn = None  # (B, heads, L) from the latest forward pass
        self._cache = None

    def _split(self, x):
        H, LF = self.H, self.L * self.F
        hist = x[:, H:H + LF].reshape(len(x), self.L, self.F)
        q_in = np.concatenate([x[:, :H], x[:, H + LF:]], axis=1)
        tok = np.concatenate([hist, np.broadcast_to(self.pos, (len(x),) + self.pos.shape)], axis=2)
        return q_in, tok

    def forward(self, x, train=True):
        x = np.asarray(x, dtype=self.dtype)
        B, h, d = len(x), self.h, self.d
        q_in, tok = self._split(x)
        q = (q_in @ self.Wq).reshape(B, h, d)
        K = (tok @ self.Wk).reshape(B, self.L, h, d)
        V = (tok @ self.Wv).reshape(B, self.L, h, d)
        s = np.einsum("bhd,blhd->bhl", q, K) / np.sqrt(d)
        a = np.exp(s - s.max(-1, keepdims=True))
        a /= a.sum(-1, keepdims=True)
        ctx = np.einsum("bhl,blhd->bhd", a, V).reshape(B, h * d)
        self.last_attn = a
        if train:
            self._cache = (q_in, tok, q, K, V, a)
        return self.mlp.forward(np.concatenate([q_in, ctx], axis=1), train)

    def backward(self, gout):
        q_in, tok, q, K, V, a = self._cache
        B, h, d = len(q_in), self.h, self.d
        g = self.mlp.backward(gout)
        g_qin, g_ctx = g[:, :q_in.shape[1]], g[:, q_in.shape[1]:].reshape(B, h, d)
        g_a = np.einsum("bhd,blhd->bhl", g_ctx, V)
        g_V = np.einsum("bhl,bhd->blhd", a, g_ctx)
        g_s = a * (g_a - (g_a * a).sum(-1, keepdims=True)) / np.sqrt(d)
        g_q = np.einsum("bhl,blhd->bhd", g_s, K).reshape(B, h * d)
        g_K = np.einsum("bhl,bhd->blhd", g_s, q)
        flat = tok.reshape(-1, tok.shape[-1])
        self.dWq[...] = q_in.T @ g_q
        self.dWk[...] = flat.T @ g_K.reshape(-1, h * d)
        self.dWv[...] = flat.T @ g_V.reshape(-1, h * d)
        return g_qin + g_q @ self.Wq.T  # gradient w.r.t. the query inputs

    def params(self):
        return [self.Wq, self.Wk, self.Wv] + self.mlp.params()

    def grads(self):
        return [self.dWq, self.dWk, self.dWv] + self.mlp.grads()
