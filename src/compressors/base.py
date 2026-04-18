"""Compressor interface shared by QJL, PolarQuant, TurboQuant, FAISS PQ, and Uncompressed.

All compressors in this project implement the same API so the evaluation
harness can treat them interchangeably.

Contract
--------
- Inputs are L2-normalized float32 vectors of shape (N, d).
- `fit(X)` may train codebooks / rotations. Cheap methods (QJL) just sample
  a random rotation; PQ does k-means; PolarQuant fits Lloyd-Max codebooks.
- `encode(X)` returns an opaque "code" object (numpy array, dict, or index).
- `ip_estimate(Q, code)` returns an (nq, N) matrix of estimated inner products
  where queries `Q` are at full float32 precision (asymmetric search).
- `bytes_per_vector()` reports the effective storage cost for reporting.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Compressor(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, X: np.ndarray) -> "Compressor":
        ...

    @abstractmethod
    def encode(self, X: np.ndarray) -> Any:
        ...

    @abstractmethod
    def ip_estimate(self, Q: np.ndarray, code: Any) -> np.ndarray:
        ...

    @abstractmethod
    def bytes_per_vector(self) -> float:
        ...

    def bits_per_dim(self, d: int) -> float:
        return 8.0 * self.bytes_per_vector() / d
