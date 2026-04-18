"""QJL — 1-bit Quantized Johnson-Lindenstrauss transform.

Reference: Zandieh, Daliri, Han. "QJL: 1-Bit Quantized JL Transform for
KV Cache Quantization with Zero Overhead." AAAI 2025. (arXiv:2406.03482)

Algorithm
---------
1. Sample a fixed random projection S ~ N(0, 1)^(m x d). Reused for
   every vector (shared with the query side).
2. Encode a unit vector x: code = sign(S x) ∈ {+1, -1}^m, packed to 1 bit/entry.
3. Asymmetric inner-product estimator:
      <q, x> ≈ sqrt(π/2) * (1/m) * Σ_i (S q)_i * sign((S x)_i)
   The query stays at full precision, so the estimator is unbiased and
   avoids the cosine-only limitation of symmetric sign-sketching.

Derivation (for S ~ N(0, I), unit-norm x):
   E[(Sq)_i · sign((Sx)_i)] = <q, x> · sqrt(2/π)
   so the mean over m rows times sqrt(π/2) is an unbiased estimate of <q, x>.

Parameters
----------
d : int
    Input dimension (512 for CLIP ViT-B/32).
bits_per_dim : float
    m / d. We instantiate m = round(bits_per_dim * d) rows. So bits_per_dim=1
    gives m=512 rows (1 bit per input dim).
seed : int
    Controls the random rotation. Run the full sweep with 5 different seeds
    and report 95% CIs (as promised in the proposal).
"""
from __future__ import annotations

import numpy as np

from .base import Compressor


class QJL(Compressor):
    name = "qjl"

    def __init__(self, d: int, bits_per_dim: float = 1.0, seed: int = 0):
        self.d = d
        self.bits_per_dim = bits_per_dim
        self.m = int(round(bits_per_dim * d))
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.S = rng.standard_normal((self.m, d)).astype(np.float32)

    def fit(self, X: np.ndarray) -> "QJL":
        return self

    def encode(self, X: np.ndarray) -> dict:
        """Returns a dict of {"norms": (N,) float32, "signs": packed bits uint8}.

        We normalize to unit length before sign-sketching because the QJL
        estimator naturally recovers <q, x>/||x||. Storing ||x|| (4 bytes, i.e.
        0.06 bits/dim for d=512) lets us handle non-unit vectors like the
        residuals produced by TurboQuant.
        """
        norms = np.linalg.norm(X, axis=1).astype(np.float32)
        safe = np.maximum(norms, 1e-12)[:, None]
        projected = (X / safe) @ self.S.T              # (N, m)
        signs = projected >= 0                          # (N, m) bool
        return {
            "norms": norms,
            "signs": np.packbits(signs, axis=1, bitorder="little"),
        }

    def _unpack(self, packed: np.ndarray) -> np.ndarray:
        bits = np.unpackbits(packed, axis=1, count=self.m, bitorder="little")
        return np.where(bits.astype(bool), 1.0, -1.0).astype(np.float32)

    def ip_estimate(self, Q: np.ndarray, code: dict) -> np.ndarray:
        Q_proj = (Q @ self.S.T).astype(np.float32)      # (nq, m)
        signs = self._unpack(code["signs"])             # (N, m)
        scale = np.sqrt(np.pi / 2.0) / self.m
        unit_ip = scale * (Q_proj @ signs.T)            # estimates <q, x/||x||>
        return unit_ip * code["norms"][None, :]         # -> <q, x>

    def bytes_per_vector(self) -> float:
        # signs: m bits; norm: 32 bits
        return self.m / 8.0 + 4.0
