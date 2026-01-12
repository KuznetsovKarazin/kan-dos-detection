# src/quant/lut_builder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np


ValueRepresentation = Literal["phi", "spline_component"]
InterpMode = Literal["nearest", "linear"]
OOBBehavior = Literal["clip", "zero"]  # semantic OOB behavior for LUT value (NOT for indexing)
QuantDType = Literal["uint8", "int8"]
QuantScheme = Literal["asymmetric", "symmetric"]
BoundaryMode = Literal["half_open", "closed"]  # domain membership definition


@dataclass
class LUTArtifact:
    """
    Persisted LUT artifact.

    Key design choice (for long-term clarity):
      - Indexing is always safe: segment indices are computed from a clipped x.
      - OOB behavior is semantic and explicit:
          * 'clip' : use boundary LUT values (from clipped x)
          * 'zero' : LUT value becomes 0 outside the domain (according to boundary_mode)

    value_representation:
      - 'phi'             : LUT stores full edge function φ_ij(x)
      - 'spline_component': LUT stores only the spline component s_ij(x), and runtime reconstructs:
            φ_ij(x) = out_scale * ( base_scale * base_fn(x) + spline_scale * s_ij(x) )

    Quantization model used by all backends (CPU/GPU):
      dequant(q) = y_min + scale * q
      where q is int8/uint8 stored in q_table.

    For symmetric quantization:
      - y_min is expected to be 0 for all segments.
      - q_table is int8.
    For asymmetric quantization:
      - q_table is uint8, y_min stores segment-wise offset.

    Arrays:
      knots:   float32 [K+1] shared knot vector
      q_table: int8/uint8 [E, K, L] values per edge, per segment, per sample
      scale:   float16/float32 [E, K]
      y_min:   float16/float32 [E, K]
    """
    
    format_version: int

    # Domain / grid
    knots: np.ndarray
    L: int
    interp: InterpMode
    boundary_mode: BoundaryMode

    # Semantic behavior for values outside the domain
    oob_behavior: OOBBehavior

    # Stored values
    q_table: np.ndarray
    scale: np.ndarray
    y_min: np.ndarray

    # Quantization metadata
    dtype: QuantDType
    scheme: QuantScheme
    qmin: int
    qmax: int

    # Representation metadata
    value_representation: ValueRepresentation
    base_kind: str  # e.g. "silu" or "none"

    # Reconstruction coefficients (optional; required for spline_component)
    edge_base_scale: Optional[np.ndarray] = None   # float32 [E]
    edge_spline_scale: Optional[np.ndarray] = None # float32 [E]
    edge_out_scale: Optional[np.ndarray] = None    # float32 [E]



def build_segment_grid(knots: np.ndarray, L: int) -> np.ndarray:
    """
    Build sample grid within each segment.
    Returns x_grid [K, L] where K = len(knots) - 1.
    Uses a half-open sampling convention inside segments to avoid hitting exact right boundaries.
    """
    knots = np.asarray(knots, dtype=np.float32)
    if knots.ndim != 1 or knots.size < 2:
        raise ValueError("knots must be 1D and have at least 2 elements")
    if L < 2:
        raise ValueError("L must be >= 2")

    K = knots.size - 1
    x_grid = np.empty((K, L), dtype=np.float32)

    for k in range(K):
        a = float(knots[k])
        b = float(knots[k + 1])
        # sample L points in [a, b) (half-open) to avoid boundary ambiguity
        # step = (b-a)/L, points: a + step * (0..L-1)
        step = (b - a) / float(L)
        x_grid[k, :] = a + step * np.arange(L, dtype=np.float32)

    return x_grid


def _quant_params_from_range(
    y_min: np.ndarray,
    y_max: np.ndarray,
    scheme: QuantScheme,
    dtype: QuantDType,
    qmin: int,
    qmax: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute (scale, y_min_for_dequant) for dequant: y = y_min + scale*q.
    For symmetric: y_min_for_dequant = 0.
    """
    y_min = np.asarray(y_min, dtype=np.float32)
    y_max = np.asarray(y_max, dtype=np.float32)

    if scheme == "symmetric":
        # symmetric around zero: y = scale * q, with q in [qmin, qmax] and y_min=0
        max_abs = np.maximum(np.abs(y_min), np.abs(y_max))
        denom = float(max(abs(qmin), abs(qmax)))
        denom = max(denom, 1.0)
        scale = np.where(max_abs > 0, max_abs / denom, 1.0).astype(np.float32)
        y0 = np.zeros_like(scale, dtype=np.float32)
        return scale, y0

    # asymmetric (offset-ymin): y = y_min + scale*q, q in [0..qmax] typically
    span = (y_max - y_min).astype(np.float32)
    denom = float(qmax - qmin)
    denom = max(denom, 1.0)
    scale = np.where(span > 0, span / denom, 1.0).astype(np.float32)
    return scale, y_min.astype(np.float32)


def build_lut_for_edges(
    edges: List,
    L: int,
    interp: InterpMode,
    y_range_method: Literal["minmax", "percentile"],
    lower_pct: float,
    upper_pct: float,
    dtype: QuantDType,
    scheme: QuantScheme,
    qmin: int,
    qmax: int,
    meta_dtype: Literal["float16", "float32"] = "float16",
    value_representation: ValueRepresentation = "phi",
    oob_behavior: OOBBehavior = "clip",
    boundary_mode: BoundaryMode = "half_open",
) -> LUTArtifact:
    """
    Build LUT artifact for a list of edges (EdgeSpec-like objects).

    Expectations for edges:
      - edges[0].knots exists and is shared by all edges
      - edge.eval_phi(x) exists
      - for value_representation='spline_component':
          edge.eval_spline(x) exists and coefficients are present:
            edge.sb (base_scale), edge.ss (spline_scale), edge.m (out_scale), edge.base_kind

    Note: oob_behavior is stored for runtime; it does not affect LUT sampling since LUT is sampled
    exactly on the knot-defined grid.
    """
    if not edges:
        raise ValueError("edges must be non-empty")
    if L < 2:
        raise ValueError("L must be >= 2")

    # Shared knots
    knots = np.asarray(edges[0].knots, dtype=np.float32)
    if knots.ndim != 1 or knots.size < 2:
        raise ValueError("edges[0].knots must be 1D with len>=2")
    for e in edges[1:]:
        k2 = np.asarray(e.knots, dtype=np.float32)
        if k2.shape != knots.shape or not np.allclose(k2, knots):
            raise ValueError("All edges must share identical knots (for this artifact format).")

    K = int(knots.size - 1)
    E = int(len(edges))

    # Choose evaluation function
    rep = str(value_representation).strip().lower()
    if rep not in ("phi", "spline_component"):
        raise ValueError(f"value_representation must be 'phi' or 'spline_component', got {value_representation!r}")

    def _eval_edge(e, x: np.ndarray) -> np.ndarray:
        if rep == "spline_component":
            if getattr(e, "eval_spline", None) is None:
                raise ValueError("spline_component requested but edge.eval_spline is missing")
            return e.eval_spline(x)
        return e.eval_phi(x)

    x_grid = build_segment_grid(knots, L)  # [K,L]

    float_lut = np.empty((E, K, L), dtype=np.float32)
    for ei, e in enumerate(edges):
        for k in range(K):
            float_lut[ei, k, :] = _eval_edge(e, x_grid[k, :]).astype(np.float32, copy=False)

    # Segment-wise y_min/y_max
    if y_range_method == "percentile":
        lo = np.percentile(float_lut, lower_pct, axis=2).astype(np.float32)  # [E,K]
        hi = np.percentile(float_lut, upper_pct, axis=2).astype(np.float32)  # [E,K]
        y_lo, y_hi = lo, hi
    else:
        y_lo = np.min(float_lut, axis=2).astype(np.float32)
        y_hi = np.max(float_lut, axis=2).astype(np.float32)

    scale, y_min = _quant_params_from_range(y_lo, y_hi, scheme=scheme, dtype=dtype, qmin=qmin, qmax=qmax)

    # Quantize: q = round((y - y_min)/scale)  (asymmetric) or q=round(y/scale) (symmetric because y_min=0)
    q = np.empty_like(float_lut, dtype=np.int32)
    # broadcast [E,K,1]
    scale_b = scale[:, :, None]
    y_min_b = y_min[:, :, None]
    q_float = (float_lut - y_min_b) / scale_b
    q[:, :, :] = np.rint(q_float).astype(np.int32)
    q = np.clip(q, qmin, qmax)

    if dtype == "uint8":
        q_table = q.astype(np.uint8)
    else:
        q_table = q.astype(np.int8)

    md = np.float16 if meta_dtype == "float16" else np.float32
    scale_md = scale.astype(md)
    y_min_md = y_min.astype(md)

    # Reconstruction coefficients
    base_kind = "none"
    edge_base_scale = None
    edge_spline_scale = None
    edge_out_scale = None

    if rep == "spline_component":
        # Check and store coefficients
        bk = getattr(edges[0], "base_kind", None) or "unknown"
        for e in edges[1:]:
            if (getattr(e, "base_kind", None) or "unknown") != bk:
                bk = "mixed"
                break
        base_kind = str(bk)

        edge_base_scale = np.empty((E,), dtype=np.float32)
        edge_spline_scale = np.empty((E,), dtype=np.float32)
        edge_out_scale = np.empty((E,), dtype=np.float32)
        for ei, e in enumerate(edges):
            sb = getattr(e, "sb", None)
            ss = getattr(e, "ss", None)
            mm = getattr(e, "m", None)
            if sb is None or ss is None or mm is None:
                raise ValueError("spline_component requested but edge coefficients (sb, ss, m) are missing")
            edge_base_scale[ei] = float(sb)
            edge_spline_scale[ei] = float(ss)
            edge_out_scale[ei] = float(mm)

    return LUTArtifact(
        format_version=1,
        knots=knots,
        L=int(L),
        interp=interp,
        boundary_mode=boundary_mode,
        oob_behavior=oob_behavior,
        q_table=q_table,
        scale=scale_md,
        y_min=y_min_md,
        dtype=dtype,
        scheme=scheme,
        qmin=int(qmin),
        qmax=int(qmax),
        value_representation=rep,  # normalized
        base_kind=base_kind,
        edge_base_scale=edge_base_scale,
        edge_spline_scale=edge_spline_scale,
        edge_out_scale=edge_out_scale,
    )


def artifact_memory_bytes(art: LUTArtifact) -> int:
    total = 0
    for arr in [art.knots, art.q_table, art.scale, art.y_min, art.edge_base_scale, art.edge_spline_scale, art.edge_out_scale]:
        if isinstance(arr, np.ndarray):
            total += int(arr.nbytes)
    return total
