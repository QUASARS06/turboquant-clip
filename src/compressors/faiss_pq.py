"""FAISS Product Quantization — the trained industry baseline.

Splits each d-dim vector into M subvectors and runs k-means with 2^nbits
centroids per subspace. We configure (M, nbits) so M*nbits/d ≈ target
bits/dim.

Constraint: d must be divisible by M. For d=512, valid M is any power-of-2
up to 512. We keep nbits=8 (FAISS default) and sweep M.

PQ does not have a native bits-per-dim of 3 at nbits=8 (would need M=192,
which does not divide 512). For 3-bit, we use PQ with nbits=4 and M=384...
that also doesn't divide. So we report PQ at {1, 2, 4, 8} bits/dim only
and note the limitation in the report.
"""
from __future__ import annotations

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None

from .base import Compressor


_BITWIDTH_TO_CONFIG = {
    1: (64, 8),    # 64 subvectors × 8 bits = 512 bits = 1 bit/dim
    2: (128, 8),   # 1024 bits = 2 bits/dim
    4: (256, 8),   # 2048 bits = 4 bits/dim
    8: (512, 8),   # 4096 bits = 8 bits/dim
}


class FaissPQ(Compressor):
    name = "faiss_pq"

    def __init__(self, d: int, bits_per_dim: int = 2, seed: int = 0):
        if faiss is None:
            raise ImportError("faiss-cpu not installed. `pip install faiss-cpu`.")
        if bits_per_dim not in _BITWIDTH_TO_CONFIG:
            raise ValueError(
                f"PQ only supports bits_per_dim in {list(_BITWIDTH_TO_CONFIG)}; "
                f"got {bits_per_dim}. 3-bit PQ is skipped — see faiss_pq.py docstring."
            )
        self.d = d
        self.bits_per_dim = bits_per_dim
        self.M, self.nbits = _BITWIDTH_TO_CONFIG[bits_per_dim]
        self.seed = seed
        self.index: "faiss.IndexPQ | None" = None
        self._codes: "np.ndarray | None" = None

    def fit(self, X: np.ndarray) -> "FaissPQ":
        faiss.omp_set_num_threads(1)
        self.index = faiss.IndexPQ(self.d, self.M, self.nbits, faiss.METRIC_INNER_PRODUCT)
        # Reproducible clustering:
        self.index.pq.cp.seed = int(self.seed)
        self.index.train(np.ascontiguousarray(X.astype(np.float32)))
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        assert self.index is not None, "call fit(X) first"
        codes = self.index.pq.compute_codes(np.ascontiguousarray(X.astype(np.float32)))
        self._codes = codes
        return codes

    def ip_estimate(self, Q: np.ndarray, code: np.ndarray) -> np.ndarray:
        """Use asymmetric distance: reconstruct codebooks, do Q · codes^T via LUT.

        For simplicity we decode to full vectors and do a dense dot-product.
        This is slower than FAISS's built-in asymmetric search but keeps the
        interface uniform and still fits comfortably in RAM for 30k vectors.
        """
        assert self.index is not None
        X_hat = self.index.pq.decode(code)
        return Q.astype(np.float32) @ X_hat.T

    def bytes_per_vector(self) -> float:
        return self.M * self.nbits / 8.0
