# src/lut_v2/evaluate_lut.py
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

from kan import KAN

from src.lut_core.models.kan_wrapper import PyKANSingleLayerAdapter
from src.lut_core.quant.lut_io import load_lut_npz
from src.lut_core.kernels.lut_contract import pack_dense_layer, PackedLUT
from src.lut_core.kernels.lut_backend_dense_numpy import forward_dense_numpy

try:
    import numba as nb  # noqa: F401
    from src.lut_core.kernels.lut_backend_dense_numba import forward_dense_numba
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False


# -----------------------------
# Utility
# -----------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x, dtype=np.float32))


def _load_json(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_manifest(lut_dir: Path) -> Dict[str, Any]:
    p = lut_dir / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"manifest.json not found in lut_dir={lut_dir}")
    return json.loads(p.read_text(encoding="utf-8"))


def _resolve_layer_paths(lut_dir: Path, manifest: Dict[str, Any]) -> List[Path]:
    out: List[Path] = []
    for layer in manifest.get("layers", []):
        raw = str(layer.get("path", ""))
        p = Path(raw)

        # absolute
        if p.is_absolute() and p.exists():
            out.append(p)
            continue

        # relative by filename
        p2 = lut_dir / p.name
        if p2.exists():
            out.append(p2)
            continue

        # try normalize windows path -> just take name
        p3 = lut_dir / Path(raw.replace("\\", "/")).name
        if p3.exists():
            out.append(p3)
            continue

        raise FileNotFoundError(f"Cannot resolve LUT layer artifact: {raw}")
    return out


def torch_model_memory_bytes(model: Any) -> Dict[str, int]:
    params_bytes = 0
    buffers_bytes = 0
    params_count = 0
    buffers_count = 0
    for p in model.parameters():
        params_count += p.numel()
        params_bytes += p.numel() * p.element_size()
    for b in model.buffers():
        buffers_count += b.numel()
        buffers_bytes += b.numel() * b.element_size()
    return {
        "params_bytes": int(params_bytes),
        "buffers_bytes": int(buffers_bytes),
        "total_bytes": int(params_bytes + buffers_bytes),
        "params_count": int(params_count),
        "buffers_count": int(buffers_count),
    }


def lut_memory_report(art: Any) -> Dict[str, Any]:
    knots_b = int(np.asarray(art.knots).nbytes)
    q_b = int(np.asarray(art.q_table).nbytes)
    s_b = int(np.asarray(art.scale).nbytes)
    y_b = int(np.asarray(art.y_min).nbytes)
    total = knots_b + q_b + s_b + y_b
    return {
        "lut_total_bytes": int(total),
        "breakdown": {
            "knots_bytes": int(knots_b),
            "q_table_bytes": int(q_b),
            "scale_bytes": int(s_b),
            "y_min_bytes": int(y_b),
        },
    }


def measure_latency(fn: Callable[[], None], warmup_iters: int, measure_iters: int) -> Dict[str, float]:
    warmup_iters = int(warmup_iters)
    measure_iters = int(measure_iters)

    for _ in range(warmup_iters):
        fn()

    t0 = time.perf_counter()
    for _ in range(measure_iters):
        fn()
    t1 = time.perf_counter()

    total = t1 - t0
    per_iter = total / max(1, measure_iters)
    return {
        "total_s": float(total),
        "per_iter_ms": float(per_iter * 1000.0),
        "iters": float(measure_iters),
    }


def timing_block(lat: Dict[str, float], batch_size: int) -> Dict[str, float]:
    per_iter_ms = float(lat["per_iter_ms"])
    return {
        "per_iter_ms": per_iter_ms,
        "ms_per_sample": per_iter_ms / float(batch_size),
        "iters": float(lat["iters"]),
        "batch_size": float(batch_size),
        "total_s": float(lat["total_s"]),
    }


def classification_metrics_from_logits(logits: np.ndarray, y_true: np.ndarray, threshold: float) -> Dict[str, float]:
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1).astype(int)

    proba = _sigmoid(logits)
    y_pred = (proba > float(threshold)).astype(int)

    acc = float((y_pred == y_true).mean())
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# -----------------------------
# Forward paths
# -----------------------------
def prepare_packed_layers(model: Any, layer_paths: List[Path], device: str) -> List[PackedLUT]:
    packed_layers: List[PackedLUT] = []
    for layer_idx, p in enumerate(layer_paths):
        adapter = PyKANSingleLayerAdapter(model, layer_idx=layer_idx, device=device)
        edges = adapter.extract_edges()
        art = load_lut_npz(p)
        packed = pack_dense_layer(
            art,
            edges=edges,
            in_dim=int(adapter.in_dim),
            out_dim=int(adapter.out_dim),
            boundary_mode=str(getattr(art, "boundary_mode", "half_open")),
        )
        packed_layers.append(packed)
    return packed_layers


def forward_packed_numpy(x: np.ndarray, packed_layers: List[PackedLUT]) -> np.ndarray:
    h = np.asarray(x, dtype=np.float32)
    for packed in packed_layers:
        h = forward_dense_numpy(h, packed)
    return h


def forward_packed_numba(x: np.ndarray, packed_layers: List[PackedLUT]) -> np.ndarray:
    h = np.asarray(x, dtype=np.float32)
    for packed in packed_layers:
        h = forward_dense_numba(h, packed)
    return h


def torch_forward_logits(model: Any, x: np.ndarray, device: str) -> np.ndarray:
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    with torch.no_grad():
        y = model(xt)
    return y.detach().cpu().numpy().astype(np.float32, copy=False)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--lut-dir", required=True)
    ap.add_argument("--device", default="cpu")

    ap.add_argument("--threads-torch", type=int, default=1)
    ap.add_argument("--threads-numba", type=int, default=1)

    ap.add_argument("--threshold", type=float, default=0.5)

    # timing config
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--warmup-iters", type=int, default=50)
    ap.add_argument("--measure-iters", type=int, default=200)

    # optional diagnostics (kept off by default)
    ap.add_argument("--phi-error", action="store_true", help="Compute expensive phi_error diagnostics (optional).")
    ap.add_argument("--phi-points", type=int, default=256)
    ap.add_argument("--phi-topk", type=int, default=10)

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    lut_dir = Path(args.lut_dir)

    # Align thread settings
    try:
        torch.set_num_threads(int(args.threads_torch))
    except Exception:
        pass

    if NUMBA_OK:
        try:
            import numba as nb2
            nb2.set_num_threads(int(args.threads_numba))
        except Exception:
            pass

    # Load dataset/model
    dataset: Dict[str, Any] = torch.load(run_dir / "dataset.pt", map_location="cpu")
    model_blob: Dict[str, Any] = torch.load(run_dir / "trained_model.pt", map_location="cpu")

    model = KAN(**model_blob["architecture"])
    model.load_state_dict(model_blob["model_state_dict"])
    model.to(args.device)
    model.eval()

    X_test = dataset["test_input"].detach().cpu().numpy().astype(np.float32, copy=False)
    y_true = dataset["test_label"].detach().cpu().numpy().reshape(-1).astype(int)

    bs = min(int(args.batch_size), int(X_test.shape[0]))
    xb = X_test[:bs]

    # Load float metrics.json as "historical reference" (optional)
    float_metrics_ref = _load_json(run_dir / "metrics.json")

    # Load LUT manifest + resolve layer artifacts
    manifest = _load_manifest(lut_dir)
    layer_paths = _resolve_layer_paths(lut_dir, manifest)

    # Memory reports
    float_mem = torch_model_memory_bytes(model)
    lut_layers_mem: List[Dict[str, Any]] = []
    lut_total = 0
    lut_break_total = {"knots_bytes": 0, "q_table_bytes": 0, "scale_bytes": 0, "y_min_bytes": 0}

    for layer_idx, p in enumerate(layer_paths):
        art = load_lut_npz(p)
        rep = lut_memory_report(art)
        lut_total += int(rep["lut_total_bytes"])
        for k, v in rep["breakdown"].items():
            lut_break_total[k] += int(v)
        lut_layers_mem.append(
            {"layer_idx": int(layer_idx), "artifact": p.name, **rep}
        )

    # Prepare packed layers once (for infer-only)
    packed_layers = prepare_packed_layers(model, layer_paths, device=args.device)

    # Optional: warm up numba compilation with correct shape
    if NUMBA_OK:
        _ = forward_packed_numba(xb, packed_layers)
        _ = forward_packed_numba(xb, packed_layers)

    # Quality metrics
    logits_float = torch_forward_logits(model, X_test, device=args.device)
    float_quality = classification_metrics_from_logits(logits_float, y_true, threshold=float(args.threshold))

    logits_lut_np = forward_packed_numpy(X_test, packed_layers)
    lut_quality_numpy = classification_metrics_from_logits(logits_lut_np, y_true, threshold=float(args.threshold))

    lut_quality_numba = None
    if NUMBA_OK:
        logits_lut_nb = forward_packed_numba(X_test, packed_layers)
        lut_quality_numba = classification_metrics_from_logits(logits_lut_nb, y_true, threshold=float(args.threshold))

    # Timing: (1) float infer-only (PyTorch)
    def fn_float_infer() -> None:
        _ = torch_forward_logits(model, xb, device=args.device)

    lat = measure_latency(fn_float_infer, warmup_iters=int(args.warmup_iters), measure_iters=int(args.measure_iters))
    timing_float_infer_only = timing_block(lat, batch_size=bs)

    # Timing: (2) LUT infer-only numpy (packed)
    def fn_lut_infer_numpy() -> None:
        _ = forward_packed_numpy(xb, packed_layers)

    lat = measure_latency(fn_lut_infer_numpy, warmup_iters=int(args.warmup_iters), measure_iters=int(args.measure_iters))
    timing_lut_infer_numpy = timing_block(lat, batch_size=bs)

    # Timing: (3) LUT infer-only numba (packed)
    timing_lut_infer_numba = None
    if NUMBA_OK:
        def fn_lut_infer_numba() -> None:
            _ = forward_packed_numba(xb, packed_layers)

        lat = measure_latency(fn_lut_infer_numba, warmup_iters=int(args.warmup_iters), measure_iters=int(args.measure_iters))
        timing_lut_infer_numba = timing_block(lat, batch_size=bs)

    # Timing: end2end includes packing inside the timed function
    # Use fewer iters to keep it practical; still comparable within itself
    e2e_warm = 3
    e2e_it = 20

    def _prepare_and_forward_numpy() -> None:
        pl = prepare_packed_layers(model, layer_paths, device=args.device)
        _ = forward_packed_numpy(xb, pl)

    lat = measure_latency(_prepare_and_forward_numpy, warmup_iters=e2e_warm, measure_iters=e2e_it)
    timing_lut_e2e_numpy = timing_block(lat, batch_size=bs)

    timing_lut_e2e_numba = None
    if NUMBA_OK:
        def _prepare_and_forward_numba() -> None:
            pl = prepare_packed_layers(model, layer_paths, device=args.device)
            # compile per-run: warmup included
            _ = forward_packed_numba(xb, pl)

        lat = measure_latency(_prepare_and_forward_numba, warmup_iters=e2e_warm, measure_iters=e2e_it)
        timing_lut_e2e_numba = timing_block(lat, batch_size=bs)

    # Optional phi_error diagnostics (kept off by default; expensive)
    phi_error = None
    if args.phi_error:
        # Local import to avoid overhead/extra deps; will use edges + LUT
        from src.metrics.phi_error import evaluate_phi_error_on_grid  # if present in your project

        phi_error = []
        for layer_idx, p in enumerate(layer_paths):
            adapter = PyKANSingleLayerAdapter(model, layer_idx=layer_idx, device=args.device)
            edges = adapter.extract_edges()
            art = load_lut_npz(p)
            rep = evaluate_phi_error_on_grid(edges, art, num_points=int(args.phi_points), topk=int(args.phi_topk))
            phi_error.append(
                {"layer_idx": int(layer_idx), "artifact": p.name, **rep}
            )

    # Build unified report
    report: Dict[str, Any] = {
        "task": "kan_dos_detection_lut_eval_unified",
        "run_dir": str(run_dir),
        "lut_dir": str(lut_dir),
        "device": str(args.device),
        "threads": {
            "torch": int(args.threads_torch),
            "numba": int(args.threads_numba),
            "numba_available": bool(NUMBA_OK),
        },
        "threshold": float(args.threshold),
        "split": "test",
        "lut_config": {
            "L": int(manifest.get("L", -1)),
            "scheme": str(manifest.get("scheme", "")),
            "dtype": str(manifest.get("dtype", "")),
            "interp": str(manifest.get("interp", "")),
            "boundary_mode": str(manifest.get("boundary_mode", "")),
            "oob_policy": str(manifest.get("oob_policy", "")),
            "value_repr": str(manifest.get("value_repr", "")),
            "calibration": manifest.get("calibration", {}),
        },
        "quality": {
            "float": float_quality,
            "lut_numpy": lut_quality_numpy,
            "lut_numba": lut_quality_numba,
            "float_metrics_reference_file": float_metrics_ref,  # optional; your old metrics.json
        },
        "timing": {
            # infer-only (comparable across methods)
            "infer_only": {
                "float_pytorch": timing_float_infer_only,
                "lut_numpy_packed": timing_lut_infer_numpy,
                "lut_numba_packed": timing_lut_infer_numba,
            },
            # end-to-end (useful for deployment budgets; not directly comparable to float_pytorch)
            "end2end": {
                "lut_numpy_prepare_plus_infer": timing_lut_e2e_numpy,
                "lut_numba_prepare_plus_infer": timing_lut_e2e_numba,
                "note": "end2end includes packing/adapter extraction each iteration; measured with fewer iters.",
            },
        },
        "memory": {
            "float_model": float_mem,
            "lut_total_bytes": int(lut_total),
            "lut_breakdown_total": lut_break_total,
            "lut_layers": lut_layers_mem,
        },
    }

    if phi_error is not None:
        report["phi_error"] = phi_error

    out_path = lut_dir / "lut_report_unified.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Console summary
    print(f"[OK] Saved: {out_path}")
    print("[QUALITY] float:", json.dumps(float_quality, indent=2))
    print("[QUALITY] lut_numpy:", json.dumps(lut_quality_numpy, indent=2))
    if lut_quality_numba is not None:
        print("[QUALITY] lut_numba:", json.dumps(lut_quality_numba, indent=2))
    print("[TIMING infer-only] float_pytorch:", json.dumps(timing_float_infer_only, indent=2))
    print("[TIMING infer-only] lut_numpy_packed:", json.dumps(timing_lut_infer_numpy, indent=2))
    if timing_lut_infer_numba is not None:
        print("[TIMING infer-only] lut_numba_packed:", json.dumps(timing_lut_infer_numba, indent=2))


if __name__ == "__main__":
    main()
