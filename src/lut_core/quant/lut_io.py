# src/quant/lut_io.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.lut_core.quant.lut_builder import LUTArtifact


def save_lut_npz(path: str | Path, art: LUTArtifact) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        format_version=np.int32(art.format_version),
        knots=np.asarray(art.knots, dtype=np.float32),
        L=np.int32(art.L),
        interp=np.asarray(art.interp),
        boundary_mode=np.asarray(art.boundary_mode),
        oob_behavior=np.asarray(art.oob_behavior),

        q_table=art.q_table,
        scale=art.scale,
        y_min=art.y_min,

        dtype=np.asarray(art.dtype),
        scheme=np.asarray(art.scheme),
        qmin=np.int32(art.qmin),
        qmax=np.int32(art.qmax),

        value_representation=np.asarray(art.value_representation),
        base_kind=np.asarray(art.base_kind),

        edge_base_scale=(art.edge_base_scale if art.edge_base_scale is not None else np.array([], dtype=np.float32)),
        edge_spline_scale=(art.edge_spline_scale if art.edge_spline_scale is not None else np.array([], dtype=np.float32)),
        edge_out_scale=(art.edge_out_scale if art.edge_out_scale is not None else np.array([], dtype=np.float32)),
    )


def _get(d: Dict[str, Any], key: str, default: Any) -> Any:
    return d[key] if key in d else default


def load_lut_npz(path: str | Path) -> LUTArtifact:
    """
    Loads LUTArtifact with backward compatibility.

    Old fields mapping (legacy):
      - value_kind: 'phi' or 'spline'  -> value_representation ('phi' or 'spline_component')
      - oob_mode:  'clip_x' or 'zero_spline' -> oob_behavior ('clip' or 'zero')
      - edge_sb/edge_ss/edge_m -> edge_base_scale/edge_spline_scale/edge_out_scale
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}

    # New-format keys
    if "value_representation" in data:
        format_version = int(np.asarray(_get(data, "format_version", np.int32(1))).item())
        knots = np.asarray(data["knots"], dtype=np.float32)
        L = int(np.asarray(data["L"]).item())
        interp = str(np.asarray(data["interp"]).item())
        boundary_mode = str(np.asarray(_get(data, "boundary_mode", np.asarray("half_open"))).item())
        oob_behavior = str(np.asarray(_get(data, "oob_behavior", np.asarray("clip"))).item())

        q_table = data["q_table"]
        scale = data["scale"]
        y_min = data["y_min"]

        dtype = str(np.asarray(data["dtype"]).item())
        scheme = str(np.asarray(data["scheme"]).item())
        qmin = int(np.asarray(data["qmin"]).item())
        qmax = int(np.asarray(data["qmax"]).item())

        value_representation = str(np.asarray(data["value_representation"]).item())
        base_kind = str(np.asarray(_get(data, "base_kind", np.asarray("none"))).item())

        ebs = np.asarray(_get(data, "edge_base_scale", np.array([], dtype=np.float32)), dtype=np.float32)
        ess = np.asarray(_get(data, "edge_spline_scale", np.array([], dtype=np.float32)), dtype=np.float32)
        eom = np.asarray(_get(data, "edge_out_scale", np.array([], dtype=np.float32)), dtype=np.float32)

        edge_base_scale = ebs if ebs.size else None
        edge_spline_scale = ess if ess.size else None
        edge_out_scale = eom if eom.size else None

        return LUTArtifact(
            format_version=format_version,
            knots=knots,
            L=L,
            interp=interp,  # type: ignore[arg-type]
            boundary_mode=boundary_mode,  # type: ignore[arg-type]
            oob_behavior=oob_behavior,  # type: ignore[arg-type]
            q_table=q_table,
            scale=scale,
            y_min=y_min,
            dtype=dtype,  # type: ignore[arg-type]
            scheme=scheme,  # type: ignore[arg-type]
            qmin=qmin,
            qmax=qmax,
            value_representation=value_representation,  # type: ignore[arg-type]
            base_kind=base_kind,
            edge_base_scale=edge_base_scale,
            edge_spline_scale=edge_spline_scale,
            edge_out_scale=edge_out_scale,
        )

    # Legacy load path
    knots = np.asarray(data["knots"], dtype=np.float32)
    L = int(np.asarray(data["L"]).item())

    interp = str(np.asarray(_get(data, "interp", np.asarray("linear"))).item())
    dtype = str(np.asarray(_get(data, "dtype", np.asarray("uint8"))).item())
    scheme = str(np.asarray(_get(data, "scheme", np.asarray("asymmetric"))).item())
    qmin = int(np.asarray(_get(data, "qmin", np.int32(0))).item())
    qmax = int(np.asarray(_get(data, "qmax", np.int32(255))).item())

    # legacy names
    value_kind = str(np.asarray(_get(data, "value_kind", np.asarray("phi"))).item()).strip().lower()
    if value_kind == "spline":
        value_representation = "spline_component"
    else:
        value_representation = "phi"

    oob_mode = str(np.asarray(_get(data, "oob_mode", np.asarray("clip_x"))).item()).strip().lower()
    if oob_mode in ("zero_spline", "zero"):
        oob_behavior = "zero"
    else:
        oob_behavior = "clip"

    q_table = data["q_table"]
    scale = data["scale"]
    y_min = data["y_min"]

    base_kind = str(np.asarray(_get(data, "base_kind", np.asarray("none"))).item())

    # coefficients
    edge_sb = np.asarray(_get(data, "edge_sb", np.array([], dtype=np.float32)), dtype=np.float32)
    edge_ss = np.asarray(_get(data, "edge_ss", np.array([], dtype=np.float32)), dtype=np.float32)
    edge_m = np.asarray(_get(data, "edge_m", np.array([], dtype=np.float32)), dtype=np.float32)

    edge_base_scale = edge_sb if edge_sb.size else None
    edge_spline_scale = edge_ss if edge_ss.size else None
    edge_out_scale = edge_m if edge_m.size else None

    return LUTArtifact(
        format_version=0,
        knots=knots,
        L=L,
        interp=interp,  # type: ignore[arg-type]
        boundary_mode="half_open",
        oob_behavior=oob_behavior,  # type: ignore[arg-type]
        q_table=q_table,
        scale=scale,
        y_min=y_min,
        dtype=dtype,  # type: ignore[arg-type]
        scheme=scheme,  # type: ignore[arg-type]
        qmin=qmin,
        qmax=qmax,
        value_representation=value_representation,  # type: ignore[arg-type]
        base_kind=base_kind,
        edge_base_scale=edge_base_scale,
        edge_spline_scale=edge_spline_scale,
        edge_out_scale=edge_out_scale,
    )
