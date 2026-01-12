# src/kernels/lut_backend_dense_numpy.py
from __future__ import annotations

import numpy as np

from src.lut_core.kernels.lut_contract import PackedLUT
from src.lut_core.kernels.lut_math import (
    base_fn,
    clip_for_indexing,
    in_domain_mask,
    lut_interp_indices,
    segment_params_nonuniform,
    segment_params_uniform,
)


def forward_dense_numpy(x: np.ndarray, packed: PackedLUT) -> np.ndarray:
    """
    Dense NumPy backend.

    Vectorizes over out_dim for each input coordinate i.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected x shape [N,in_dim], got {x.shape}")

    N, in_dim = x.shape
    in_dim_p, out_dim, KL = packed.q_flat.shape
    if in_dim != in_dim_p:
        raise ValueError(f"in_dim mismatch: x has {in_dim}, packed has {in_dim_p}")
    if KL != packed.K * packed.L:
        raise ValueError("PackedLUT q_flat has inconsistent last dimension.")

    y = np.zeros((N, out_dim), dtype=np.float32)

    for i in range(in_dim):
        xi = x[:, i]
        xi_clip = clip_for_indexing(xi, packed.x_min, packed.x_max, packed.boundary_mode)

        if packed.uniform_dx is not None:
            k, u = segment_params_uniform(xi_clip, packed.x_min, packed.uniform_dx, packed.K)
        else:
            k, u = segment_params_nonuniform(xi_clip, packed.knots)

        r0, r1, w = lut_interp_indices(u, packed.L)

        idx0 = (k * packed.L + r0).astype(np.int32)  # [N]
        idx1 = (k * packed.L + r1).astype(np.int32)  # [N]

        # q_flat for all j at once: [out_dim, K*L]
        qij = packed.q_flat[i]  # [out_dim, KL]

        # Gather q0/q1: shape [out_dim, N]
        q0 = qij[:, idx0]
        q1 = qij[:, idx1]

        # Gather y_min/scale per segment: [out_dim, N]
        y_min = packed.y_min[i][:, k]
        scale = packed.scale[i][:, k]

        y0 = y_min + scale * q0.astype(np.float32)
        y1 = y_min + scale * q1.astype(np.float32)
        w2 = w.astype(np.float32)[None, :]

        if packed.interp == "nearest":
            lut_val = np.where(w2 < 0.5, y0, y1)
        else:
            lut_val = (1.0 - w2) * y0 + w2 * y1  # [out_dim, N]

        if packed.oob_behavior == "zero":
            mask = in_domain_mask(xi, packed.x_min, packed.x_max, packed.boundary_mode).astype(np.float32)
            lut_val = lut_val * mask[None, :]

        if packed.value_representation == "phi":
            y += lut_val.T
        else:
            base = base_fn(xi, packed.base_kind).astype(np.float32)
            base2 = base[None, :]
            y += (packed.coef_out[i][:, None] * (packed.coef_base[i][:, None] * base2 + packed.coef_lut[i][:, None] * lut_val)).T

    return y
