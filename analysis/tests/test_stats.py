"""Unit tests for wdiff.stats (pure math, no I/O)."""
import math
import unittest

import numpy as np
import torch

from wdiff import stats


class TestFro(unittest.TestCase):
    def test_fro_sq_matches_numpy_odd_sizes(self):
        g = torch.Generator().manual_seed(1)
        for shape in [(7,), (1023,), (513, 257), (3, 1)]:
            x = torch.randn(*shape, generator=g)
            want = float(np.linalg.norm(x.numpy().astype(np.float64)) ** 2)
            # relative tolerance: chunked f64 accumulation vs numpy pairwise
            # summation differ only in summation order (~1e-13 relative)
            self.assertAlmostEqual(stats.fro_sq(x), want, delta=1e-10 * want)

    def test_chunking(self):
        old = stats._CHUNK
        try:
            stats._CHUNK = 64
            x = torch.randn(1000, generator=torch.Generator().manual_seed(2))
            want = float(np.dot(x.numpy().astype(np.float64), x.numpy().astype(np.float64)))
            self.assertAlmostEqual(stats.fro_sq(x), want, places=8)
        finally:
            stats._CHUNK = old

    def test_cos_flat(self):
        x = torch.randn(64, 32, generator=torch.Generator().manual_seed(3))
        self.assertAlmostEqual(stats.cos_flat(x, x), 1.0, places=12)
        self.assertAlmostEqual(stats.cos_flat(x, -x), -1.0, places=12)
        self.assertTrue(math.isnan(stats.cos_flat(x, torch.zeros_like(x))))


class TestRanks(unittest.TestCase):
    def test_energy_rank_exact(self):
        sig = np.array([3.0, 2.0, 1.0])
        fro = 14.0
        self.assertEqual(stats.energy_rank(sig, fro, 0.5, exact=True), 1)   # 9 >= 7
        self.assertEqual(stats.energy_rank(sig, fro, 0.9, exact=True), 2)   # 13 >= 12.6
        self.assertEqual(stats.energy_rank(sig, fro, 0.99, exact=True), 3)
        self.assertEqual(stats.energy_rank(sig, 0.0, 0.9, exact=True), 0)

    def test_energy_rank_partial(self):
        sig = np.array([3.0, 2.0])          # truncated spectrum; true fro 14
        self.assertEqual(stats.energy_rank(sig, 14.0, 0.5, exact=False), 1)
        # top-2 capture 13/14 = 92.9% >= 90% -> exact value reachable within K
        self.assertEqual(stats.energy_rank(sig, 14.0, 0.9, exact=False), 2)
        # ... but 99% is not reachable -> ">K"
        self.assertEqual(stats.energy_rank(sig, 14.0, 0.99, exact=False), ">2")

    def test_energy_rank_boundary_fp(self):
        # cum exactly == target must count (>=, with fp tolerance)
        sig = np.array([1.0, 1.0])
        self.assertEqual(stats.energy_rank(sig, 2.0, 0.5, exact=False), 1)
        self.assertEqual(stats.energy_rank(sig, 2.0, 1.0, exact=False), 2)

    def test_stable_rank(self):
        self.assertEqual(stats.stable_rank(0.0, 0.0), 0.0)
        self.assertAlmostEqual(stats.stable_rank(12.0, 1.0), 12.0)

    def test_entropy_erank(self):
        self.assertAlmostEqual(stats.entropy_erank(np.ones(7)), 7.0, places=10)
        self.assertAlmostEqual(stats.entropy_erank(np.array([5.0, 0.0])), 1.0, places=10)
        self.assertEqual(stats.entropy_erank(np.zeros(3)), 0.0)


class TestSpectrum(unittest.TestCase):
    def _rank_r(self, m, n, r, seed=0):
        g = torch.Generator().manual_seed(seed)
        U, _ = torch.linalg.qr(torch.randn(m, r, generator=g))
        V, _ = torch.linalg.qr(torch.randn(n, r, generator=g))
        return U @ V.T  # singular values all 1

    def test_full_path_recovers_rank(self):
        dW = self._rank_r(96, 64, 12)
        fro = stats.fro_sq(dW)
        s, spec = stats.two_d_stats(dW, fro, full_threshold=8192, topk=64)
        self.assertEqual(s["spectrum"], "full")
        self.assertEqual(s["energy_rank_0.99"], 12)
        self.assertEqual(s["energy_rank_0.9"], 11)   # ceil(0.9*12)
        self.assertEqual(s["energy_rank_0.5"], 6)
        self.assertAlmostEqual(s["stable_rank"], 12.0, places=3)
        self.assertAlmostEqual(s["erank_entropy"], 12.0, places=2)

    def test_randomized_path_recovers_rank(self):
        dW = self._rank_r(96, 64, 12)
        fro = stats.fro_sq(dW)
        s, spec = stats.two_d_stats(dW, fro, full_threshold=8, topk=32)
        self.assertEqual(s["spectrum"], "randomized")
        self.assertIsNone(s["erank_entropy"])
        self.assertEqual(s["energy_rank_0.99"], 12)
        self.assertAlmostEqual(s["stable_rank"], 12.0, places=2)

    def test_randomized_unreachable_energy(self):
        # full-rank noise, tiny topk -> ">K"
        g = torch.Generator().manual_seed(4)
        dW = torch.randn(256, 200, generator=g)
        fro = stats.fro_sq(dW)
        s, _ = stats.two_d_stats(dW, fro, full_threshold=8, topk=8, oversample=8)
        self.assertEqual(s["energy_rank_0.99"], ">8")

    def test_zero_matrix(self):
        dW = torch.zeros(32, 16)
        s, spec = stats.two_d_stats(dW, 0.0)
        self.assertEqual(s["stable_rank"], 0.0)
        self.assertEqual(s["energy_rank_0.9"], 0)
        self.assertEqual(s["sigma1"], 0.0)

    def test_vectors_orthonormal_and_reconstruct(self):
        dW = self._rank_r(96, 64, 12) * 3.0
        _, spec = stats.two_d_stats(dW, stats.fro_sq(dW), keep_vectors=12)
        approx = spec.u @ np.diag(spec.sigmas[:12]) @ spec.v.T
        self.assertLess(np.abs(approx - dW.numpy()).max(), 1e-4)

    def test_max_svd_dim_guard(self):
        dW = torch.randn(64, 32, generator=torch.Generator().manual_seed(5))
        spec = stats.compute_spectrum(dW, full_threshold=8192, max_svd_dim=48,
                                      topk=16, device="cpu")
        self.assertEqual(spec.method, "randomized")


class TestPrincipalAngles(unittest.TestCase):
    def test_identical_subspaces(self):
        g = torch.Generator().manual_seed(6)
        q, _ = torch.linalg.qr(torch.randn(64, 8, generator=g))
        c2 = stats.principal_angle_cos2(q.numpy(), q.numpy())
        np.testing.assert_allclose(c2, np.ones(8), atol=1e-10)

    def test_orthogonal_subspaces(self):
        q, _ = torch.linalg.qr(torch.randn(64, 16, generator=torch.Generator().manual_seed(7)))
        c2 = stats.principal_angle_cos2(q.numpy()[:, :8], q.numpy()[:, 8:])
        np.testing.assert_allclose(c2, np.zeros(8), atol=1e-10)

    def test_random_baseline(self):
        self.assertAlmostEqual(stats.random_subspace_mean_cos2(64, 16), 0.25)


if __name__ == "__main__":
    unittest.main()
