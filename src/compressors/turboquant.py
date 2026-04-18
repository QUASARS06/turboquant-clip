"""TurboQuant — PolarQuant main stage + QJL residual correction.

Reference: Zandieh, Daliri, Hadian, Mirrokni. "TurboQuant: Online Vector
Quantization with Near-optimal Distortion Rate." ICLR 2026.
(arXiv:2504.19874, Algorithm 2 + Figure 1)

Idea
----
PolarQuant's reconstruction `x_hat` is accurate, but the implied
inner-product estimator `<q, x_hat>` has a small bias. QJL on the residual
`r = x - x_hat` gives an unbiased correction at the cost of 1 extra bit per
dimension. So:

    <q, x> ≈ <q, x_hat> + sqrt(π/2) * (1/m) * Σ (S q)_i sign((S r)_i)

Budget bookkeeping:
    total_bits_per_dim ≈ angle_bits * (d-1)/d + qjl_bits_per_dim
Paper target: total ≈ 3 bits (angle_bits=2, qjl=1) or 4 bits (angle_bits=3, qjl=1).
"""
from __future__ import annotations

import numpy as np

from .base import Compressor
from .polarquant import PolarQuant
from .qjl import QJL


class TurboQuant(Compressor):
    name = "turboquant"

    def __init__(self, d: int, angle_bits: int = 2, qjl_bits_per_dim: float = 1.0, seed: int = 0):
        self.d = d
        self.angle_bits = angle_bits
        self.qjl_bits_per_dim = qjl_bits_per_dim
        self.seed = seed
        self.polar = PolarQuant(d, angle_bits=angle_bits, seed=seed)
        # Use a DIFFERENT seed for QJL rotation so the two stages don't correlate.
        self.qjl = QJL(d, bits_per_dim=qjl_bits_per_dim, seed=seed + 10_000)

    def fit(self, X: np.ndarray) -> "TurboQuant":
        self.polar.fit(X)
        return self

    def encode(self, X: np.ndarray) -> dict:
        X = X.astype(np.float32)
        polar_code = self.polar.encode(X)
        X_hat = self.polar.decode(polar_code)
        residual = X - X_hat
        qjl_code = self.qjl.encode(residual)
        return {"polar": polar_code, "qjl": qjl_code}

    def ip_estimate(self, Q: np.ndarray, code: dict) -> np.ndarray:
        Q = Q.astype(np.float32)
        main = self.polar.ip_estimate(Q, code["polar"])
        correction = self.qjl.ip_estimate(Q, code["qjl"])
        return main + correction

    def bytes_per_vector(self) -> float:
        return self.polar.bytes_per_vector() + self.qjl.bytes_per_vector()
