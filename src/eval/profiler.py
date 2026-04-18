"""Memory + latency profiler for compressors.

We report:
  * `bytes_per_vector`  : from the compressor itself (bookkeeping).
  * `index_bytes`       : actual bytes materialized by `encode` (via sys.getsizeof / .nbytes).
  * `compress_seconds`  : wall-clock time to encode the full database.
  * `query_latency_ms`  : mean time per query (averaged over `n_queries`).
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class ProfileResult:
    method: str
    bytes_per_vector: float
    index_bytes: int
    compress_seconds: float
    query_latency_ms: float

    def to_dict(self):
        return asdict(self)


def _sizeof(obj) -> int:
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, dict):
        return sum(_sizeof(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(_sizeof(v) for v in obj)
    return sys.getsizeof(obj)


def profile(compressor, X_db: np.ndarray, Q: np.ndarray, n_queries: int = 200) -> ProfileResult:
    compressor.fit(X_db)
    t0 = time.perf_counter()
    code = compressor.encode(X_db)
    t1 = time.perf_counter()

    nq = min(n_queries, Q.shape[0])
    q_sub = Q[:nq]
    t2 = time.perf_counter()
    _ = compressor.ip_estimate(q_sub, code)
    t3 = time.perf_counter()

    return ProfileResult(
        method=compressor.name,
        bytes_per_vector=compressor.bytes_per_vector(),
        index_bytes=_sizeof(code),
        compress_seconds=t1 - t0,
        query_latency_ms=1000.0 * (t3 - t2) / nq,
    )
