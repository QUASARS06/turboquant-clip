"""PolarQuant — recursive polar-coordinate quantization.

Reference: Han, Kacham, Karbasi, Mirrokni, Zandieh. "PolarQuant: Quantizing
KV Caches with Polar Transformation." AISTATS 2026. (arXiv:2502.02617)

Algorithm (from Algorithm 1 of the paper)
-----------------------------------------
Let d be a power of two (d=512 for CLIP ViT-B/32).

1. Precondition: x <- R x, where R is a shared fixed random orthogonal
   matrix. This is what makes coordinates behave "uniformly" — so a small
   codebook per level covers all vectors well.
2. Recursive polar transform:
      level 0: group into d/2 pairs (x_{2i}, x_{2i+1}) -> (r_i, θ_i)
               with r_i = sqrt(x_{2i}^2 + x_{2i+1}^2), θ_i = atan2(x_{2i+1}, x_{2i})
      level k: apply the same pair transform to the radii from level k-1
   After log2(d) levels you have ONE magnitude and d-1 angles total
   (256 + 128 + 64 + ... + 1 = 511 for d=512).
3. Magnitude is stored at float32 (it is ||x||, already ~1 for unit vectors).
4. Angles are quantized to `angle_bits` bits each with a Lloyd-Max codebook
   fit on the empirical angle distribution from training data. We fit one
   codebook per level, since the distributions differ slightly.

Inner-product estimator
-----------------------
Unlike QJL, PolarQuant does not have a closed-form inner-product estimator.
We reconstruct x_hat from the code and compute <q, x_hat> directly. This is
slightly biased but very accurate at 3+ bits/angle.

Parameters
----------
d : int
    Must be a power of two.
angle_bits : int
    Bits per angle. Effective bits-per-dim ≈ angle_bits * (d-1)/d + 32/d
    (the last term is the magnitude). For d=512 this is essentially angle_bits.
seed : int
    Rotation seed.
"""
from __future__ import annotations

import numpy as np

from .base import Compressor


def _random_orthogonal(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(A)
    return Q.astype(np.float32)


def _lloyd_max_1d(samples: np.ndarray, n_levels: int, n_iter: int = 25) -> np.ndarray:
    """Fit a 1-D Lloyd-Max codebook (k-means with n_levels centroids) on samples."""
    lo, hi = np.percentile(samples, [0.5, 99.5])
    centroids = np.linspace(lo, hi, n_levels)
    for _ in range(n_iter):
        # Assign
        dists = np.abs(samples[:, None] - centroids[None, :])
        assignments = dists.argmin(axis=1)
        # Update
        new_centroids = centroids.copy()
        for k in range(n_levels):
            mask = assignments == k
            if mask.any():
                new_centroids[k] = samples[mask].mean()
        if np.allclose(new_centroids, centroids, atol=1e-6):
            break
        centroids = new_centroids
    return np.sort(centroids).astype(np.float32)


def _quantize_to_codebook(values: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Return the index (uint16) of the nearest codebook entry for each value."""
    dists = np.abs(values[..., None] - codebook[None, :])
    return dists.argmin(axis=-1).astype(np.uint16)


def _polar_encode_level(x: np.ndarray):
    """Pair up consecutive entries of the last axis; return (radii, angles)."""
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    a = pairs[..., 0]
    b = pairs[..., 1]
    r = np.sqrt(a * a + b * b)
    theta = np.arctan2(b, a)  # in (-π, π]
    return r.astype(np.float32), theta.astype(np.float32)


def _polar_decode_level(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    a = r * np.cos(theta)
    b = r * np.sin(theta)
    stacked = np.stack([a, b], axis=-1)
    return stacked.reshape(*stacked.shape[:-2], -1)


class PolarQuant(Compressor):
    name = "polarquant"

    def __init__(self, d: int, angle_bits: int = 3, seed: int = 0):
        if d & (d - 1) != 0:
            raise ValueError(f"d must be a power of 2 (got {d}).")
        self.d = d
        self.angle_bits = angle_bits
        self.seed = seed
        self.n_levels_per_angle = 2 ** angle_bits
        self.n_transform_levels = int(np.log2(d))  # log2(512)=9
        self.R = _random_orthogonal(d, seed)
        self.codebooks: list[np.ndarray] = []

    def _transform(self, X: np.ndarray):
        """Apply recursive polar transform. Returns (magnitudes, angles_per_level).
        `angles_per_level[k]` has shape (N, d // 2**(k+1))."""
        rotated = X @ self.R.T
        angles_per_level = []
        current = rotated
        for _ in range(self.n_transform_levels):
            r, theta = _polar_encode_level(current)
            angles_per_level.append(theta)
            current = r
        magnitudes = current.squeeze(-1)  # (N,)
        return magnitudes, angles_per_level

    def _inverse_transform(self, magnitudes: np.ndarray, angles_per_level: list[np.ndarray]) -> np.ndarray:
        current = magnitudes.reshape(-1, 1)
        for theta in reversed(angles_per_level):
            a = current * np.cos(theta)
            b = current * np.sin(theta)
            stacked = np.stack([a, b], axis=-1)
            current = stacked.reshape(stacked.shape[0], -1)
        return (current @ self.R).astype(np.float32)

    def fit(self, X: np.ndarray) -> "PolarQuant":
        """Fit per-level Lloyd-Max codebooks on a sample of X (up to 20k vectors)."""
        sample = X if len(X) <= 20000 else X[np.random.default_rng(self.seed).choice(len(X), 20000, replace=False)]
        _, angles_per_level = self._transform(sample.astype(np.float32))
        self.codebooks = [
            _lloyd_max_1d(theta.ravel(), self.n_levels_per_angle)
            for theta in angles_per_level
        ]
        return self

    def encode(self, X: np.ndarray) -> dict:
        magnitudes, angles_per_level = self._transform(X.astype(np.float32))
        quantized = [
            _quantize_to_codebook(theta, cb)
            for theta, cb in zip(angles_per_level, self.codebooks)
        ]
        return {"magnitudes": magnitudes.astype(np.float32), "angle_codes": quantized}

    def decode(self, code: dict) -> np.ndarray:
        angles_per_level = [
            cb[idx] for idx, cb in zip(code["angle_codes"], self.codebooks)
        ]
        return self._inverse_transform(code["magnitudes"], angles_per_level)

    def ip_estimate(self, Q: np.ndarray, code: dict) -> np.ndarray:
        X_hat = self.decode(code)
        return Q @ X_hat.T

    def bytes_per_vector(self) -> float:
        # (d-1) angles * angle_bits bits + 32 bits for magnitude
        total_bits = (self.d - 1) * self.angle_bits + 32
        return total_bits / 8.0
