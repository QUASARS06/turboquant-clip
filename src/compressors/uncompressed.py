"""Float32 reference — the gold standard every compressed method is compared to."""
from __future__ import annotations

import numpy as np

from .base import Compressor


class Uncompressed(Compressor):
    name = "float32"

    def __init__(self, d: int):
        self.d = d

    def fit(self, X: np.ndarray) -> "Uncompressed":
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        return X.astype(np.float32, copy=False)

    def ip_estimate(self, Q: np.ndarray, code: np.ndarray) -> np.ndarray:
        return Q @ code.T

    def bytes_per_vector(self) -> float:
        return 4.0 * self.d
