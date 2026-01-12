# src/models/kan_wrapper.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np


@dataclass(frozen=True)
class EdgeSpec:
    """
    Canonical edge specification for a single dense KAN layer.

    Float semantics (PyKAN-style):
        phi_ij(x) = m_ij * ( sb_ij * base(x) + ss_ij * spline(x) )

    Notes for LUT conversion (v2):
      - value_representation="phi": store full phi_ij(x) in LUT (no reconstruction coefficients required)
      - value_representation="spline_component": store only spline(x) in LUT and reconstruct at inference:
            phi_ij(x) = m_ij * ( sb_ij * base(x) + ss_ij * lut_spline(x) )

    The builder currently expects the legacy coefficient names:
      - sb : base scale
      - ss : spline scale
      - m  : output scale / mask
    """
    edge_id: int
    src_idx: int
    dst_idx: int

    knots: np.ndarray  # shared [K+1], float32

    eval_phi: Callable[[np.ndarray], np.ndarray]
    eval_spline: Optional[Callable[[np.ndarray], np.ndarray]]

    base_kind: str
    sb: float
    ss: float
    m: float


class PyKANSingleLayerAdapter:
    """
    Minimal adapter around a PyKAN model that exposes:
      - extract_edges(): list[EdgeSpec] for a single activation layer
      - forward_float(x): float forward using PyKAN model

    Contract assumptions:
      - We export ONE layer (layer_idx) and assume dense connectivity.
      - We require a single shared knot vector across all input channels
        (current LUT artifact format stores one knots[] per layer).
    """

    def __init__(self, model: Any, *, layer_idx: int = 0, device: str = "cpu"):
        self.model = model
        self.layer_idx = int(layer_idx)
        self.device = str(device)

        try:
            import torch  # type: ignore
        except Exception as e:
            raise ImportError("PyKANSingleLayerAdapter requires torch.") from e

        self.model.to(self.device)
        self.model.eval()

        # Infer dims from layer tensors
        layer = self.model.act_fun[self.layer_idx]
        grid = getattr(layer, "grid")
        coef = getattr(layer, "coef")

        grid_t = self._as_tensor(grid).to(self.device)
        coef_t = self._as_tensor(coef).to(self.device)

        if grid_t.ndim == 3:
            # [in_dim, 1, K+1] -> [in_dim, K+1]
            grid_t = grid_t[:, 0, :]
        if coef_t.ndim == 4:
            # [in_dim, out_dim, 1, Kcoef] -> [in_dim, out_dim, Kcoef]
            coef_t = coef_t[:, :, 0, :]

        self.in_dim = int(grid_t.shape[0])
        self.out_dim = int(coef_t.shape[1])

    @staticmethod
    def from_arch(arch: dict, checkpoint: Optional[str] = None, device: str = "cpu") -> "PyKANSingleLayerAdapter":
        try:
            import torch  # type: ignore
        except Exception as e:
            raise ImportError("PyKANSingleLayerAdapter requires torch.") from e

        try:
            from kan import KAN  # type: ignore
        except Exception as e:
            raise ImportError(
                "Cannot import 'KAN' from module 'kan'. This project uses PyKAN.\n"
                "Install it via:\n"
                "  pip install pykan\n"
                "or (recommended):\n"
                "  pip install git+https://github.com/KindXiaoming/pykan.git"
            ) from e
        

        width = arch.get("width", None)
        if width is None:
            raise ValueError("arch.width is required (e.g., [8,8]).")
        grid = int(arch.get("grid", 5))
        k = int(arch.get("k", 3))
        seed = int(arch.get("seed", 0))

        torch.manual_seed(seed)
        np.random.seed(seed)

        model = KAN(width=width, grid=grid, k=k, seed=seed, device=device)

        if checkpoint:
            ckpt_path = Path(checkpoint)
            if ckpt_path.exists():
                state = torch.load(str(ckpt_path), map_location=device)
                sd = None
                if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
                    sd = state["state_dict"]
                elif isinstance(state, dict):
                    sd = state
                if isinstance(sd, dict):
                    try:
                        model.load_state_dict(sd, strict=False)
                    except Exception:
                        # best-effort load; keep going for sanity experiments
                        pass

        return PyKANSingleLayerAdapter(model=model, layer_idx=int(arch.get("layer_idx", 0)), device=device)

    def _as_tensor(self, x: Any):
        import torch  # type: ignore
        if isinstance(x, torch.Tensor):
            return x
        return torch.as_tensor(x)

    @staticmethod
    def _infer_base_kind(base_fun: Any) -> str:
        try:
            import torch.nn as nn  # type: ignore
            if isinstance(base_fun, nn.SiLU):
                return "silu"
        except Exception:
            pass
        name = type(base_fun).__name__.lower() if base_fun is not None else ""
        if "silu" in name:
            return "silu"
        if "identity" in name:
            return "none"
        # In this repo we only implement "none" and "silu" in kernels.
        return "none"

    @staticmethod
    def _edge_scalar_2d(mat: Any, i: int, j: int) -> float:
        import torch  # type: ignore
        t = mat
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t)
        return float(t[i, j].detach().cpu().item())

    @staticmethod
    def _grid_1xK_for_input(grid_t, in_idx: int):
        # grid_t is [in_dim, K+1]
        return grid_t[in_idx : in_idx + 1, :]

    @staticmethod
    def _coef_1x1xK_for_edge(coef_t, in_idx: int, out_idx: int):
        # coef_t is [in_dim, out_dim, Kcoef]
        return coef_t[in_idx : in_idx + 1, out_idx : out_idx + 1, :]

    @staticmethod
    def _shared_knots_from_grid(grid_t) -> np.ndarray:
        """
        Enforce the current artifact constraint: one shared knots[] per layer.
        PyKAN stores grid per input channel; we only support the case when they are equal.
        """
        grid_np = grid_t.detach().cpu().numpy().astype(np.float32, copy=False)
        if grid_np.ndim != 2 or grid_np.shape[0] < 1 or grid_np.shape[1] < 2:
            raise ValueError(f"Unexpected grid shape {grid_np.shape}; expected [in_dim, K+1].")

        ref = grid_np[0]
        for i in range(1, grid_np.shape[0]):
            if not np.allclose(grid_np[i], ref, rtol=0.0, atol=1e-7):
                raise ValueError(
                    "PyKAN grid is not shared across inputs. Current LUTArtifact format stores a single knots[] "
                    "vector per layer, so this layer cannot be exported without extending the artifact format "
                    "(e.g., per-input knots or per-edge knots)."
                )

        # Ensure strictly increasing knots
        if not np.all(np.diff(ref) > 0):
            raise ValueError("Invalid grid/knots: expected strictly increasing knot vector.")
        return ref.astype(np.float32, copy=False)

    def extract_edges(self) -> List[EdgeSpec]:
        try:
            import torch  # type: ignore
        except Exception as e:
            raise ImportError("PyKANSingleLayerAdapter requires torch.") from e

        try:
            from kan.spline import coef2curve  # type: ignore
        except Exception as e:
            raise ImportError("Cannot import kan.spline.coef2curve; PyKAN install may be incomplete.") from e

        layer = self.model.act_fun[self.layer_idx]

        grid = getattr(layer, "grid")
        coef = getattr(layer, "coef")
        scale_base = getattr(layer, "scale_base")
        scale_sp = getattr(layer, "scale_sp")
        mask = getattr(layer, "mask")

        base_fun = getattr(layer, "base_fun", None)
        base_kind = self._infer_base_kind(base_fun)

        grid_t = self._as_tensor(grid).to(self.device)
        coef_t = self._as_tensor(coef).to(self.device)

        if grid_t.ndim == 3:
            grid_t = grid_t[:, 0, :]
        if coef_t.ndim == 4:
            coef_t = coef_t[:, :, 0, :]

        in_dim = int(grid_t.shape[0])
        out_dim = int(coef_t.shape[1])

        # Shared knots per layer (artifact constraint)
        knots_shared = self._shared_knots_from_grid(grid_t)

        k_degree = int(getattr(layer, "k", 3))

        def _base_torch(x_t):
            if base_fun is None:
                return x_t
            return base_fun(x_t)

        edges: List[EdgeSpec] = []
        for out_idx in range(out_dim):
            for in_idx in range(in_dim):
                edge_id = out_idx * in_dim + in_idx

                sb = self._edge_scalar_2d(scale_base, in_idx, out_idx)
                ss = self._edge_scalar_2d(scale_sp, in_idx, out_idx)
                m = self._edge_scalar_2d(mask, in_idx, out_idx)

                g_1xK = self._grid_1xK_for_input(grid_t, in_idx)
                c_1x1xK = self._coef_1x1xK_for_edge(coef_t, in_idx, out_idx)

                def _eval_spline(
                    x_np: np.ndarray,
                    *,
                    g=g_1xK,
                    c=c_1x1xK,
                    k=k_degree,
                ) -> np.ndarray:
                    x_np = np.asarray(x_np, dtype=np.float32).reshape(-1)
                    xt = torch.from_numpy(x_np).to(self.device).view(-1, 1)
                    with torch.no_grad():
                        y = coef2curve(x_eval=xt, grid=g, coef=c, k=k).view(-1)
                    return y.detach().cpu().numpy().astype(np.float32, copy=False)

                def _eval_phi(
                    x_np: np.ndarray,
                    *,
                    sb_=float(sb),
                    ss_=float(ss),
                    m_=float(m),
                    g=g_1xK,
                    c=c_1x1xK,
                    k=k_degree,
                ) -> np.ndarray:
                    x_np = np.asarray(x_np, dtype=np.float32).reshape(-1)
                    xt = torch.from_numpy(x_np).to(self.device).view(-1, 1)
                    with torch.no_grad():
                        base = _base_torch(xt).view(-1)
                        spl = coef2curve(x_eval=xt, grid=g, coef=c, k=k).view(-1)
                        y = m_ * (sb_ * base + ss_ * spl)
                    return y.detach().cpu().numpy().astype(np.float32, copy=False)

                edges.append(
                    EdgeSpec(
                        edge_id=int(edge_id),
                        src_idx=int(in_idx),
                        dst_idx=int(out_idx),
                        knots=knots_shared,
                        eval_phi=_eval_phi,
                        eval_spline=_eval_spline,
                        base_kind=base_kind,
                        sb=float(sb),
                        ss=float(ss),
                        m=float(m),
                    )
                )

        return edges

    def forward_float(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError(f"Expected x shape [N,{self.in_dim}], got {x.shape}")

        import torch  # type: ignore

        xt = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            yt = self.model(xt)
        return yt.detach().cpu().numpy().astype(np.float32, copy=False)
