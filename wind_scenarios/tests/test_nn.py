import unittest

import numpy as np

from tests.helpers import small_windows  # noqa: F401  (sets sys.path)
from windgen.nn import MLP, Adam, AttnDenoiser


class TestMLP(unittest.TestCase):
    def test_gradient_check(self):
        rng = np.random.default_rng(0)
        net = MLP([5, 7, 6, 3], rng, dtype=np.float64, out_scale=1.0)
        x = rng.standard_normal((4, 5))
        y = rng.standard_normal((4, 3))

        def loss():
            return 0.5 * np.sum((net.forward(x, train=False) - y) ** 2)

        out = net.forward(x, train=True)
        net.backward(out - y)
        analytic = [g.copy() for g in net.grads()]
        eps = 1e-6
        for p, g in zip(net.params(), analytic):
            flat_ix = rng.choice(p.size, size=min(6, p.size), replace=False)
            for fi in flat_ix:
                ix = np.unravel_index(fi, p.shape)
                old = p[ix]
                p[ix] = old + eps
                lp = loss()
                p[ix] = old - eps
                lm = loss()
                p[ix] = old
                num = (lp - lm) / (2 * eps)
                self.assertAlmostEqual(num, g[ix], delta=1e-5 * max(1.0, abs(num)))

    def test_input_gradient_shape(self):
        rng = np.random.default_rng(1)
        net = MLP([4, 8, 2], rng)
        x = rng.standard_normal((3, 4))
        net.forward(x)
        self.assertEqual(net.backward(np.ones((3, 2))).shape, (3, 4))

    def test_adam_fits_regression(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((256, 3))
        y = np.sin(x[:, :1]) + 0.5 * x[:, 1:2] * x[:, 2:3]
        net = MLP([3, 32, 32, 1], rng, out_scale=1.0)
        opt = Adam(net.params(), lr=3e-3)
        first = None
        for i in range(400):
            out = net.forward(x)
            d = out - y
            loss = float(np.mean(d ** 2))
            first = loss if first is None else first
            net.backward(2 * d / d.size)
            opt.step(net.grads(), clip=5.0)
        self.assertLess(loss, 0.25 * first)

    def test_float32_dtype_preserved(self):
        net = MLP([3, 4, 2], np.random.default_rng(0), dtype=np.float32)
        out = net.forward(np.ones((2, 3)))
        self.assertEqual(out.dtype, np.float32)
        net.backward(np.ones((2, 2)))
        opt = Adam(net.params())
        opt.step(net.grads())
        self.assertTrue(all(p.dtype == np.float32 for p in net.params()))

    def test_bad_sizes(self):
        with self.assertRaises(ValueError):
            MLP([3], np.random.default_rng(0))


class TestAttnDenoiser(unittest.TestCase):
    def make(self):
        rng = np.random.default_rng(3)
        H, L, F, C, Tm = 3, 5, 2, 4, 4
        net = AttnDenoiser(H, L, F, C, Tm, hidden=6, n_layers=2, rng=rng, heads=2, d_head=3, pos_dim=4)
        net.mlp.layers[-1].W *= 10  # out_scale=0.1 would make the check too lenient
        x = rng.standard_normal((4, H + L * F + C + Tm))
        return net, x, rng.standard_normal((4, H)), rng

    def test_gradient_check(self):
        net, x, y, rng = self.make()

        def loss():
            return 0.5 * np.sum((net.forward(x, train=False) - y) ** 2)

        out = net.forward(x, train=True)
        net.backward(out - y)
        analytic = [g.copy() for g in net.grads()]
        eps = 1e-6
        for p, g in zip(net.params(), analytic):
            for fi in rng.choice(p.size, size=min(6, p.size), replace=False):
                ix = np.unravel_index(fi, p.shape)
                old = p[ix]
                p[ix] = old + eps
                lp = loss()
                p[ix] = old - eps
                lm = loss()
                p[ix] = old
                num = (lp - lm) / (2 * eps)
                self.assertAlmostEqual(num, g[ix], delta=1e-5 * max(1.0, abs(num)))

    def test_attention_is_a_distribution_and_history_only_enters_through_it(self):
        net, x, _, _ = self.make()
        net.forward(x, train=False)
        self.assertEqual(net.last_attn.shape, (4, 2, 5))
        np.testing.assert_allclose(net.last_attn.sum(-1), 1.0)
        # zero the key/value rows of the F=2 history features -> output must ignore history
        net.Wk[:2] = 0.0
        net.Wv[:2] = 0.0
        a = net.forward(x, train=False)
        x2 = x.copy()
        x2[:, 3:13] += 5.0  # shift the history block
        np.testing.assert_allclose(a, net.forward(x2, train=False))


if __name__ == "__main__":
    unittest.main()
