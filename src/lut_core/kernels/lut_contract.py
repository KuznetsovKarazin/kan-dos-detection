# src/kernels/lut_contract.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.lut_core.quant.lut_builder import LUTArtifact


@dataclass(frozen=True)
class PackedLUT:
    """
    Runtime contract for LUT inference backends.

    Shapes:
      q_flat:   [in_dim, out_dim, K*L]  (stored dtype: int8/uint8)
      scale:    [in_dim, out_dim, K]    float32
      y_min:    [in_dim, out_dim, K]    float32
      knots:    [K+1]                   float32

      coef_base:  [in_dim, out_dim]     float32
      coef_lut:   [in_dim, out_dim]     float32
      coef_out:   [in_dim, out_dim]     float32

    Semantics:
      value_representation:
        - "phi": LUT stores the full edge function φ_ij(x)
        - "spline_component": LUT stores only the spline component; φ_ij(x) is reconstructed at inference

      oob_behavior:
        - "clip": evaluate using x clipped to the knot domain
        - "zero": return 0 for the LUT component outside the knot domain (masking is boundary_mode-aware)
          NOTE: for spline_component, "zero" masks ONLY the LUT (spline) component; base branch is preserved.

      boundary_mode:
        - "half_open": in-range is [x_min, x_max)
        - "closed": in-range is [x_min, x_max]
    """
    q_flat: np.ndarray
    scale: np.ndarray
    y_min: np.ndarray
    knots: np.ndarray
    L: int
    K: int
    interp: str
    q_dtype: str

    base_kind: str
    coef_base: np.ndarray
    coef_lut: np.ndarray
    coef_out: np.ndarray

    value_representation: str
    oob_behavior: str
    boundary_mode: str
    x_min: float
    x_max: float
    uniform_dx: Optional[float]


def _edge_matrix(edges: List, in_dim: int, out_dim: int) -> np.ndarray:
    edge_ids = -np.ones((in_dim, out_dim), dtype=np.int32)
    for e in edges:
        i = int(getattr(e, "src_idx"))
        j = int(getattr(e, "dst_idx"))
        edge_ids[i, j] = int(getattr(e, "edge_id"))
    if np.any(edge_ids < 0):
        raise ValueError("Edges do not cover full (in_dim,out_dim) connectivity for a dense layer.")
    return edge_ids


def _normalize_value_representation(art: LUTArtifact) -> str:
    # New name
    vr = getattr(art, "value_representation", None)
    if vr is not None:
        s = str(vr).lower().strip()
        if s in ("phi",):
            return "phi"
        if s in ("spline_component", "spline", "component"):
            return "spline_component"

    # Legacy name
    vk = str(getattr(art, "value_kind", "phi") or "phi").lower().strip()
    return "spline_component" if vk in ("spline", "spline_component") else "phi"


def _normalize_oob_behavior(art: LUTArtifact) -> Tuple[str, str]:
    """
    Returns (oob_behavior, oob_semantics_hint)

    oob_behavior: "clip" | "zero"
    oob_semantics_hint: "zero_spline" | "zero_phi" | ""  (used only for legacy compatibility if needed)
    """
    # New name
    ob = getattr(art, "oob_behavior", None)
    if ob is not None:
        s = str(ob).lower().strip()
        if s in ("clip", "clip_x"):
            return "clip", ""
        if s in ("zero", "zero_spline", "zero_phi", "zero_lut"):
            # We keep "zero" in runtime; hint is empty in new format.
            return "zero", ""

    # Legacy name
    mode = str(getattr(art, "oob_mode", "clip_x") or "clip_x").lower().strip()
    if mode in ("zero_spline", "zero"):
        return "zero", "zero_spline"
    if mode in ("zero_phi",):
        return "zero", "zero_phi"
    return "clip", ""


def _read_coefficients(art: LUTArtifact, E: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns (base_scale, spline_scale, out_scale) arrays of shape (E,), or (None,None,None) if not present.
    Supports both new and legacy field names.
    """
    # New names
    ebs = getattr(art, "edge_base_scale", None)
    ess = getattr(art, "edge_spline_scale", None)
    eos = getattr(art, "edge_out_scale", None)

    if ebs is not None or ess is not None or eos is not None:
        if ebs is None or ess is None or eos is None:
            raise ValueError("Partial edge_*_scale present; need edge_base_scale, edge_spline_scale, edge_out_scale.")
        ebs = np.asarray(ebs, dtype=np.float32).reshape(-1)
        ess = np.asarray(ess, dtype=np.float32).reshape(-1)
        eos = np.asarray(eos, dtype=np.float32).reshape(-1)
        if ebs.size != E or ess.size != E or eos.size != E:
            raise ValueError("edge_*_scale arrays must have shape (E,).")
        return ebs, ess, eos

    # Legacy names
    sb = getattr(art, "edge_sb", None)
    ss = getattr(art, "edge_ss", None)
    m = getattr(art, "edge_m", None)
    if sb is None or ss is None or m is None:
        return None, None, None

    sb = np.asarray(sb, dtype=np.float32).reshape(-1)
    ss = np.asarray(ss, dtype=np.float32).reshape(-1)
    m = np.asarray(m, dtype=np.float32).reshape(-1)
    if sb.size != E or ss.size != E or m.size != E:
        raise ValueError("legacy edge_sb/edge_ss/edge_m arrays must have shape (E,).")
    return sb, ss, m


def pack_dense_layer(
    art: LUTArtifact,
    *,
    edges: List,
    in_dim: int,
    out_dim: int,
    boundary_mode: str = "half_open",
) -> PackedLUT:
    boundary_mode = boundary_mode if boundary_mode in ("half_open", "closed") else "half_open"

    edge_ids = _edge_matrix(edges, in_dim=in_dim, out_dim=out_dim)

    # Knots
    knots = np.asarray(art.knots, dtype=np.float32)
    if knots.ndim != 1 or knots.size < 2:
        raise ValueError("art.knots must be a 1D array with size >= 2.")
    K = int(knots.size - 1)
    x_min = float(knots[0])
    x_max = float(knots[-1])

    # Table params
    L = int(art.L)
    q_table = np.asarray(art.q_table)
    scale = np.asarray(art.scale, dtype=np.float32)
    y_min = np.asarray(art.y_min, dtype=np.float32)

    if q_table.ndim != 3 or q_table.shape[1] != K or q_table.shape[2] != L:
        raise ValueError(f"Unexpected q_table shape {q_table.shape}, expected [E,{K},{L}].")

    E = int(q_table.shape[0])

    # Dense packing per (i,j)
    q_dense = np.empty((in_dim, out_dim, K, L), dtype=q_table.dtype)
    s_dense = np.empty((in_dim, out_dim, K), dtype=np.float32)
    y_dense = np.empty((in_dim, out_dim, K), dtype=np.float32)

    for i in range(in_dim):
        for j in range(out_dim):
            eid = int(edge_ids[i, j])
            if eid < 0 or eid >= E:
                raise ValueError(f"edge_id out of range for (i={i},j={j}): {eid}")
            q_dense[i, j, :, :] = q_table[eid, :, :]
            s_dense[i, j, :] = scale[eid, :]
            y_dense[i, j, :] = y_min[eid, :]

    q_flat = q_dense.reshape(in_dim, out_dim, K * L)

    value_representation = _normalize_value_representation(art)
    oob_behavior, _oob_hint = _normalize_oob_behavior(art)

    interp = str(getattr(art, "interp", "linear") or "linear").lower().strip()
    q_dtype = str(getattr(art, "dtype", "uint8") or "uint8").lower().strip()

    # Base kind
    base_kind = str(getattr(art, "base_kind", "none") or "none").lower().strip()
    if value_representation == "phi":
        base_kind = "none"

    # Coefficients
    if value_representation == "phi":
        coef_base = np.zeros((in_dim, out_dim), dtype=np.float32)
        coef_lut = np.ones((in_dim, out_dim), dtype=np.float32)
        coef_out = np.ones((in_dim, out_dim), dtype=np.float32)
    else:
        ebs, ess, eos = _read_coefficients(art, E)
        if ebs is None or ess is None or eos is None:
            raise ValueError(
                "spline_component representation requires per-edge coefficients "
                "(edge_base_scale/edge_spline_scale/edge_out_scale or legacy edge_sb/edge_ss/edge_m)."
            )

        coef_base = np.empty((in_dim, out_dim), dtype=np.float32)
        coef_lut = np.empty((in_dim, out_dim), dtype=np.float32)
        coef_out = np.empty((in_dim, out_dim), dtype=np.float32)

        for i in range(in_dim):
            for j in range(out_dim):
                eid = int(edge_ids[i, j])
                coef_base[i, j] = float(ebs[eid])
                coef_lut[i, j] = float(ess[eid])
                coef_out[i, j] = float(eos[eid])

    # Uniform grid detection
    if np.allclose(np.diff(knots), np.diff(knots)[0], rtol=0.0, atol=1e-7):
        uniform_dx = float(knots[1] - knots[0])
    else:
        uniform_dx = None

    return PackedLUT(
        q_flat=q_flat,
        scale=s_dense,
        y_min=y_dense,
        knots=knots,
        L=L,
        K=K,
        interp=interp,
        q_dtype=q_dtype,
        base_kind=base_kind,
        coef_base=coef_base,
        coef_lut=coef_lut,
        coef_out=coef_out,
        value_representation=value_representation,
        oob_behavior=oob_behavior,
        boundary_mode=boundary_mode,
        x_min=x_min,
        x_max=x_max,
        uniform_dx=uniform_dx,
    )
