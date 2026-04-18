"""Sanity tests. Each compressor must produce IP estimates that correlate
with true IPs at r > 0.98 on 4+ bits. If this fails, the estimator math
is broken and the downstream sweep is meaningless.

    pytest tests/
"""
from __future__ import annotations

import numpy as np
import pytest

from src.compressors import FaissPQ, PolarQuant, QJL, TurboQuant, Uncompressed


D = 512
N_DB = 2000
N_Q = 100
SEED = 0


def _fixture():
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((N_DB, D)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Q = rng.standard_normal((N_Q, D)).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    true_ip = Q @ X.T
    return X, Q, true_ip


def _correlation(est: np.ndarray, truth: np.ndarray) -> float:
    e = est.ravel()
    t = truth.ravel()
    return float(np.corrcoef(e, t)[0, 1])


def test_uncompressed_exact():
    X, Q, true_ip = _fixture()
    c = Uncompressed(D).fit(X)
    est = c.ip_estimate(Q, c.encode(X))
    assert np.allclose(est, true_ip, atol=1e-5)


@pytest.mark.parametrize("bits_per_dim", [1, 2, 4, 8])
def test_qjl_correlation(bits_per_dim):
    X, Q, true_ip = _fixture()
    c = QJL(D, bits_per_dim=bits_per_dim, seed=SEED).fit(X)
    est = c.ip_estimate(Q, c.encode(X))
    r = _correlation(est, true_ip)
    print(f"QJL {bits_per_dim}b: r={r:.3f}")
    threshold = {1: 0.75, 2: 0.85, 4: 0.92, 8: 0.97}[bits_per_dim]
    assert r > threshold, f"QJL@{bits_per_dim}b correlation {r:.3f} below {threshold}"


@pytest.mark.parametrize("angle_bits", [2, 3, 4])
def test_polarquant_correlation(angle_bits):
    X, Q, true_ip = _fixture()
    c = PolarQuant(D, angle_bits=angle_bits, seed=SEED).fit(X)
    est = c.ip_estimate(Q, c.encode(X))
    r = _correlation(est, true_ip)
    print(f"PolarQuant {angle_bits}b: r={r:.3f}")
    threshold = {2: 0.85, 3: 0.93, 4: 0.96}[angle_bits]
    assert r > threshold


@pytest.mark.parametrize("angle_bits", [2, 3])
def test_turboquant_correlation(angle_bits):
    X, Q, true_ip = _fixture()
    c = TurboQuant(D, angle_bits=angle_bits, qjl_bits_per_dim=1.0, seed=SEED).fit(X)
    est = c.ip_estimate(Q, c.encode(X))
    r = _correlation(est, true_ip)
    print(f"TurboQuant {angle_bits}b+1: r={r:.3f}")
    assert r > 0.92


def test_faiss_pq_correlation():
    faiss = pytest.importorskip("faiss")
    X, Q, true_ip = _fixture()
    c = FaissPQ(D, bits_per_dim=4, seed=SEED).fit(X)
    est = c.ip_estimate(Q, c.encode(X))
    r = _correlation(est, true_ip)
    print(f"FAISS PQ 4b: r={r:.3f}")
    assert r > 0.92
