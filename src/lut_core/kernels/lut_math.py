# src/kernels/lut_math.py
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def nextafter_less(x: float) -> float:
    """Largest float32 strictly less than x."""
    return float(np.nextafter(np.float32(x), np.float32(-np.inf), dtype=np.float32))


def in_domain_mask(x: np.ndarray, x_min: float, x_max: float, boundary_mode: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if boundary_mode == "closed":
        return (x >= x_min) & (x <= x_max)
    return (x >= x_min) & (x < x_max)


def clip_for_indexing(x: np.ndarray, x_min: float, x_max: float, boundary_mode: str) -> np.ndarray:
    """
    Clips x for index computations.

    For 'half_open' boundary, values equal to x_max are clipped to nextafter(x_max, -inf)
    so that segment indexing and interpolation remain well-defined.
    """
    x = np.asarray(x, dtype=np.float32)
    hi = x_max if boundary_mode == "closed" else nextafter_less(x_max)
    return np.clip(x, np.float32(x_min), np.float32(hi)).astype(np.float32, copy=False)


def base_fn(x: np.ndarray, base_kind: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    bk = (base_kind or "none").lower().strip()
    if bk in ("none", "identity", "id"):
        return x
    if bk == "silu":
        # SiLU(x) = x * sigmoid(x)
        return x / (1.0 + np.exp(-x, dtype=np.float32))
    raise ValueError(f"Unsupported base_kind='{base_kind}'")


def segment_params_uniform(
    x_clip: np.ndarray, x_min: float, dx: float, K: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For uniform knots: returns (k, u) such that x in [x_k, x_{k+1}) and u in [0,1].
    """
    x_clip = np.asarray(x_clip, dtype=np.float32)
    t = (x_clip - np.float32(x_min)) / np.float32(dx)
    k = np.floor(t).astype(np.int32)
    k = np.clip(k, 0, K - 1)
    t0 = np.float32(x_min) + k.astype(np.float32) * np.float32(dx)
    u = (x_clip - t0) / np.float32(dx)
    # Guard: numerical drift
    u = np.clip(u, np.float32(0.0), np.float32(1.0))
    return k, u


def segment_params_nonuniform(x_clip: np.ndarray, knots: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    For arbitrary knots: returns (k, u) such that x in [knots[k], knots[k+1}) and u in [0,1].
    """
    x_clip = np.asarray(x_clip, dtype=np.float32)
    knots = np.asarray(knots, dtype=np.float32)
    K = int(knots.size - 1)

    # k = searchsorted(knots, x, side="right") - 1, clipped.
    k = np.searchsorted(knots, x_clip, side="right").astype(np.int32) - 1
    k = np.clip(k, 0, K - 1)

    x0 = knots[k]
    x1 = knots[k + 1]
    denom = np.maximum(x1 - x0, np.float32(1e-12))
    u = (x_clip - x0) / denom
    u = np.clip(u, np.float32(0.0), np.float32(1.0))
    return k, u


def lut_interp_indices(u: np.ndarray, L: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts u in [0,1] to LUT indices r0, r1 and interpolation weight w in [0,1].
    Uses L samples per segment with endpoints included (pos = u*(L-1)).
    """
    u = np.asarray(u, dtype=np.float32)
    pos = u * np.float32(L - 1)
    r0 = np.floor(pos).astype(np.int32)
    r0 = np.clip(r0, 0, L - 1)
    r1 = np.clip(r0 + 1, 0, L - 1)
    w = (pos - r0.astype(np.float32)).astype(np.float32)
    return r0, r1, w


def dequant(q: np.ndarray, y_min: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """
    Dequantizes q into float32 using per-segment affine parameters.

    y = y_min + scale * q
    """
    return (y_min.astype(np.float32) + scale.astype(np.float32) * q.astype(np.float32)).astype(np.float32, copy=False)
