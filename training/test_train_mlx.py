"""Check train_mlx.py's chunked gated delta rule against mlx-lm's reference loop.

  cd training && python -m unittest test_train_mlx

Needs mlx and mlx-lm (Apple silicon); skipped elsewhere. Runs on the CPU.
"""
import unittest

try:
    import mlx.core as mx
    from mlx_lm.models.gated_delta import gated_delta_ops
except ImportError:  # not on Apple silicon, or the MLX environment is not active
    mx = None

if mx is not None:
    import numpy as np

    import train_mlx as T


@unittest.skipIf(mx is None, "mlx-lm is not installed")
class ChunkedGatedDeltaTest(unittest.TestCase):
    def setUp(self):
        mx.set_default_device(mx.cpu)
        mx.random.seed(1)
        B, S, H, D = 2, 150, 4, 32  # 150 tokens: three chunks, the last one padded
        q, k = mx.random.normal((B, S, H, D)), mx.random.normal((B, S, H, D))
        self.q = q / mx.linalg.norm(q, axis=-1, keepdims=True) * D**-0.5
        self.k = k / mx.linalg.norm(k, axis=-1, keepdims=True)
        self.v = mx.random.normal((B, S, H, D))
        self.log_g = -mx.random.uniform(0.0, 0.5, (B, S, H))
        self.beta = mx.random.uniform(0.0, 1.0, (B, S, H))
        self.w = mx.random.normal((B, S, H, D))

    def test_forward_matches_reference(self):
        y_ref, s_ref = gated_delta_ops(self.q, self.k, self.v, mx.exp(self.log_g), self.beta)
        y, s = T.chunked_gated_delta(self.q, self.k, self.v, self.log_g, self.beta)
        self.assertLess(mx.abs(y - y_ref).max().item(), 1e-5)
        self.assertLess(mx.abs(s - s_ref).max().item(), 1e-5)

    def test_gradients_match_reference(self):
        def ref(q, k, v, lg, b):
            y, s = gated_delta_ops(q, k, v, mx.exp(lg), b)
            return (y * self.w).sum() + s.sum()

        def chunked(q, k, v, lg, b):
            y, s = T.chunked_gated_delta(q, k, v, lg, b)
            return (y * self.w).sum() + s.sum()

        args = (self.q, self.k, self.v, self.log_g, self.beta)
        g_ref = mx.grad(ref, argnums=(0, 1, 2, 3, 4))(*args)
        g = mx.grad(chunked, argnums=(0, 1, 2, 3, 4))(*args)
        for name, a, b in zip(["q", "k", "v", "log_g", "beta"], g_ref, g):
            self.assertLess(mx.abs(a - b).max().item(), 1e-3 * max(1.0, mx.abs(a).max().item()), name)

    def test_repeated_keys_stay_finite(self):
        """Identical keys (padding, repeated tokens) make the in-chunk system nearly all ones."""
        k = mx.broadcast_to(self.k[:, :1], self.k.shape)
        ones, zeros = mx.ones_like(self.beta), mx.zeros_like(self.log_g)
        y_ref, _ = gated_delta_ops(self.q, k, self.v, mx.exp(zeros), ones)
        y, _ = T.chunked_gated_delta(self.q, k, self.v, zeros, ones)
        self.assertLess(mx.abs(y - y_ref).max().item(), 1e-4)

    def test_unit_lower_inverse_and_gradient(self):
        rng = np.random.default_rng(0)
        C = 64
        L = np.tril(rng.normal(size=(2, C, C)), -1) * 0.1
        W = rng.normal(size=(2, C, C))
        Lm, Wm = mx.array(L.astype(np.float32)), mx.array(W.astype(np.float32))
        inv = np.linalg.inv(np.eye(C) + L)
        self.assertLess(np.abs(np.array(T.unit_lower_inverse(Lm)) - inv).max(), 1e-4)
        g = np.array(mx.grad(lambda x: (T.unit_lower_inverse(x) * Wm).sum())(Lm))
        exact = -np.einsum("bki,bkl,bjl->bij", inv, W, inv)
        lower = np.tril(np.ones((C, C)), -1) > 0
        self.assertLess(np.abs((g - exact)[:, lower]).max(), 1e-4)


if __name__ == "__main__":
    unittest.main()
