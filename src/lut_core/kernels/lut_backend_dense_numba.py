# src/kernels/lut_backend_dense_numba.py
from __future__ import annotations

import numpy as np

from src.lut_core.kernels.lut_contract import PackedLUT

try:
    import numba as nb
    _NUMBA_AVAILABLE = True
except Exception:
    nb = None
    _NUMBA_AVAILABLE = False


def numba_available() -> bool:
    return bool(_NUMBA_AVAILABLE)


def _interp_code(interp: str) -> int:
    m = (interp or "linear").lower().strip()
    if m == "nearest":
        return 1
    return 0  # linear


def _base_code(base_kind: str) -> int:
    bk = (base_kind or "none").lower().strip()
    if bk in ("none", "identity", "id"):
        return 0
    if bk == "silu":
        return 1
    raise ValueError(f"Unsupported base_kind='{base_kind}' for numba backend")


if _NUMBA_AVAILABLE:

    @nb.njit(cache=True, fastmath=False)
    def _binary_search_knots(x: float, knots: np.ndarray) -> int:
        K = knots.size - 1
        lo = 0
        hi = K
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if knots[mid] <= x:
                lo = mid
            else:
                hi = mid
        if lo < 0:
            lo = 0
        elif lo > K - 1:
            lo = K - 1
        return lo

    @nb.njit(cache=True, fastmath=False)
    def _silu(x: float) -> float:
        return x / (1.0 + np.exp(-x))

    @nb.njit(cache=True, fastmath=False)
    def _forward_dense_numba(
        x: np.ndarray,
        q_flat: np.ndarray,
        scale: np.ndarray,
        y_min: np.ndarray,
        knots: np.ndarray,
        coef_base: np.ndarray,
        coef_lut: np.ndarray,
        coef_out: np.ndarray,
        L: int,
        K: int,
        interp_code: int,
        is_phi: int,
        oob_zero: int,
        boundary_closed: int,
        base_code: int,
        uniform_dx: float,
        has_uniform: int,
        x_min: float,
        x_max: float,
    ) -> np.ndarray:
        N = x.shape[0]
        in_dim = x.shape[1]
        out_dim = q_flat.shape[1]

        y = np.zeros((N, out_dim), dtype=np.float32)

        x_hi = x_max if boundary_closed == 1 else np.nextafter(np.float32(x_max), np.float32(-np.inf))

        for n in range(N):
            for i in range(in_dim):
                x_raw = float(x[n, i])

                # Clip for indexing
                xc = x_raw
                if xc < x_min:
                    xc = x_min
                elif xc > x_hi:
                    xc = float(x_hi)

                # Segment k and u
                if has_uniform == 1:
                    dx = uniform_dx
                    t = (xc - x_min) / dx
                    k = int(np.floor(t))
                    if k < 0:
                        k = 0
                    elif k > K - 1:
                        k = K - 1
                    t0 = x_min + k * dx
                    u = (xc - t0) / dx
                else:
                    k = _binary_search_knots(xc, knots)
                    x0 = float(knots[k])
                    x1 = float(knots[k + 1])
                    denom = x1 - x0
                    if denom <= 1e-12:
                        denom = 1e-12
                    u = (xc - x0) / denom

                if u < 0.0:
                    u = 0.0
                elif u > 1.0:
                    u = 1.0

                # LUT indices within the segment (endpoints included)
                pos = u * (L - 1)
                r0 = int(np.floor(pos))
                if r0 < 0:
                    r0 = 0
                elif r0 > L - 1:
                    r0 = L - 1
                r1 = r0 + 1
                if r1 > L - 1:
                    r1 = L - 1
                w = pos - r0

                idx0 = k * L + r0
                idx1 = k * L + r1

                # Base value (evaluated on raw x)
                if is_phi == 1:
                    base_val = 0.0
                else:
                    if base_code == 0:
                        base_val = x_raw
                    else:
                        base_val = _silu(x_raw)

                # In-range mask for OOB=zero
                in_range = 1
                if oob_zero == 1:
                    if boundary_closed == 1:
                        if not (x_raw >= x_min and x_raw <= x_max):
                            in_range = 0
                    else:
                        if not (x_raw >= x_min and x_raw < x_max):
                            in_range = 0

                for j in range(out_dim):
                    q0 = float(q_flat[i, j, idx0])
                    q1 = float(q_flat[i, j, idx1])

                    ym = float(y_min[i, j, k])
                    sc = float(scale[i, j, k])

                    y0 = ym + sc * q0
                    y1 = ym + sc * q1

                    if interp_code == 1:
                        lut_val = y0 if w < 0.5 else y1
                    else:
                        lut_val = (1.0 - w) * y0 + w * y1

                    if oob_zero == 1 and in_range == 0:
                        lut_val = 0.0

                    if is_phi == 1:
                        y[n, j] += lut_val
                    else:
                        y[n, j] += float(coef_out[i, j]) * (float(coef_base[i, j]) * base_val + float(coef_lut[i, j]) * lut_val)

        return y


def warmup_numba(packed: PackedLUT, *, in_dim: int, out_dim: int) -> None:
    if not _NUMBA_AVAILABLE:
        return
    x = np.zeros((2, in_dim), dtype=np.float32)
    _ = forward_dense_numba(x, packed)


def forward_dense_numba(x: np.ndarray, packed: PackedLUT) -> np.ndarray:
    if not _NUMBA_AVAILABLE:
        raise RuntimeError("Numba is not available (pip install numba).")

    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected x shape [N,in_dim], got {x.shape}")
    if not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)

    q_flat = np.ascontiguousarray(packed.q_flat)
    scale = np.ascontiguousarray(packed.scale, dtype=np.float32)
    y_min = np.ascontiguousarray(packed.y_min, dtype=np.float32)
    knots = np.ascontiguousarray(packed.knots, dtype=np.float32)

    coef_base = np.ascontiguousarray(packed.coef_base, dtype=np.float32)
    coef_lut = np.ascontiguousarray(packed.coef_lut, dtype=np.float32)
    coef_out = np.ascontiguousarray(packed.coef_out, dtype=np.float32)

    interp_code = _interp_code(packed.interp)
    is_phi = 1 if packed.value_representation == "phi" else 0
    oob_zero = 1 if packed.oob_behavior == "zero" else 0
    boundary_closed = 1 if packed.boundary_mode == "closed" else 0
    base_code = _base_code(packed.base_kind)

    has_uniform = 1 if packed.uniform_dx is not None else 0
    uniform_dx = float(packed.uniform_dx) if packed.uniform_dx is not None else 0.0

    return _forward_dense_numba(
        x=x,
        q_flat=q_flat,
        scale=scale,
        y_min=y_min,
        knots=knots,
        coef_base=coef_base,
        coef_lut=coef_lut,
        coef_out=coef_out,
        L=int(packed.L),
        K=int(packed.K),
        interp_code=int(interp_code),
        is_phi=int(is_phi),
        oob_zero=int(oob_zero),
        boundary_closed=int(boundary_closed),
        base_code=int(base_code),
        uniform_dx=float(uniform_dx),
        has_uniform=int(has_uniform),
        x_min=float(packed.x_min),
        x_max=float(packed.x_max),
    )
