# src/lut_v2/artifact.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class LUTLayerArtifact:
    """
    Dense KAN activation layer LUT artifact (per-edge, per-segment LUT).

    q_table: shape [E, S, L], dtype int8 (symmetric) or uint8 (asymmetric)
    scale : shape [E, S], float32
    y_min : shape [E, S], float32 (only for asymmetric; else None)
    knots : shape [K+1], float32 (shared across inputs)
    sb, ss, m: shape [E], float32
    """
    in_dim: int
    out_dim: int
    L: int
    degree: int
    interp: str                 # "nearest" | "linear"
    value_repr: str             # "spline_component" | "phi"
    scheme: str                 # "symmetric" | "asymmetric"
    dtype: str                  # "int8" | "uint8"
    boundary_mode: str          # "closed" | "half_open"
    oob_policy: str             # "clip_x" | "zero_spline"
    base_kind: str              # "silu" | "none"

    knots: np.ndarray
    q_table: np.ndarray
    scale: np.ndarray
    y_min: Optional[np.ndarray]
    sb: np.ndarray
    ss: np.ndarray
    m: np.ndarray

    @property
    def edges(self) -> int:
        return int(self.in_dim * self.out_dim)

    @property
    def segments(self) -> int:
        return int(self.knots.shape[0] - 1)

    def to_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "L": self.L,
            "degree": self.degree,
            "interp": self.interp,
            "value_repr": self.value_repr,
            "scheme": self.scheme,
            "dtype": self.dtype,
            "boundary_mode": self.boundary_mode,
            "oob_policy": self.oob_policy,
            "base_kind": self.base_kind,
        }
        np.savez_compressed(
            str(path),
            meta_json=np.array([json.dumps(meta)], dtype=object),
            knots=self.knots.astype(np.float32),
            q_table=self.q_table,
            scale=self.scale.astype(np.float32),
            y_min=(self.y_min.astype(np.float32) if self.y_min is not None else np.array([], dtype=np.float32)),
            sb=self.sb.astype(np.float32),
            ss=self.ss.astype(np.float32),
            m=self.m.astype(np.float32),
        )

    @staticmethod
    def from_npz(path: Path) -> "LUTLayerArtifact":
        z = np.load(str(path), allow_pickle=True)
        meta = json.loads(str(z["meta_json"][0]))
        y_min_arr = z["y_min"]
        y_min = None if y_min_arr.size == 0 else y_min_arr.astype(np.float32)
        return LUTLayerArtifact(
            in_dim=int(meta["in_dim"]),
            out_dim=int(meta["out_dim"]),
            L=int(meta["L"]),
            degree=int(meta["degree"]),
            interp=str(meta["interp"]),
            value_repr=str(meta["value_repr"]),
            scheme=str(meta["scheme"]),
            dtype=str(meta["dtype"]),
            boundary_mode=str(meta["boundary_mode"]),
            oob_policy=str(meta["oob_policy"]),
            base_kind=str(meta["base_kind"]),
            knots=z["knots"].astype(np.float32),
            q_table=z["q_table"],
            scale=z["scale"].astype(np.float32),
            y_min=y_min,
            sb=z["sb"].astype(np.float32),
            ss=z["ss"].astype(np.float32),
            m=z["m"].astype(np.float32),
        )

    def memory_bytes(self) -> Dict[str, int]:
        def nbytes(a: Optional[np.ndarray]) -> int:
            return 0 if a is None else int(a.nbytes)

        return {
            "q_table": int(self.q_table.nbytes),
            "scale": int(self.scale.nbytes),
            "y_min": nbytes(self.y_min),
            "knots": int(self.knots.nbytes),
            "sb": int(self.sb.nbytes),
            "ss": int(self.ss.nbytes),
            "m": int(self.m.nbytes),
            "total": int(
                self.q_table.nbytes
                + self.scale.nbytes
                + (0 if self.y_min is None else self.y_min.nbytes)
                + self.knots.nbytes
                + self.sb.nbytes
                + self.ss.nbytes
                + self.m.nbytes
            ),
        }
