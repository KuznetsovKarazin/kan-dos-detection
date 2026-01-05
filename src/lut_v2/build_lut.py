# src/lut_v2/build_lut.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
from kan import KAN

from src.lut_core.models.kan_wrapper import PyKANSingleLayerAdapter
from src.lut_core.quant.lut_builder import build_lut_for_edges
from src.lut_core.quant.lut_io import save_lut_npz


def _bytes_total_npz_payload(art) -> int:
    b = 0
    b += int(np.asarray(art.q_table).nbytes)
    b += int(np.asarray(art.scale).nbytes)
    b += int(np.asarray(art.y_min).nbytes)
    b += int(np.asarray(art.knots).nbytes)
    if art.edge_base_scale is not None:
        b += int(np.asarray(art.edge_base_scale).nbytes)
    if art.edge_spline_scale is not None:
        b += int(np.asarray(art.edge_spline_scale).nbytes)
    if art.edge_out_scale is not None:
        b += int(np.asarray(art.edge_out_scale).nbytes)
    return b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--value-repr", choices=["spline_component", "phi"], default="spline_component")
    ap.add_argument("--scheme", choices=["symmetric", "asymmetric"], default="symmetric")
    ap.add_argument("--dtype", choices=["int8", "uint8"], default="int8")
    ap.add_argument("--interp", choices=["nearest", "linear"], default="linear")
    ap.add_argument("--boundary-mode", choices=["closed", "half_open"], default="closed")
    ap.add_argument("--oob-policy", choices=["clip_x", "zero_spline"], default="clip_x")
    ap.add_argument("--calib-split", choices=["train", "test"], default="train")
    ap.add_argument("--num-samples", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load run bundle
    dataset: Dict[str, Any] = torch.load(run_dir / "dataset.pt", map_location="cpu")
    model_data: Dict[str, Any] = torch.load(run_dir / "trained_model.pt", map_location="cpu")
    model = KAN(**model_data["architecture"])
    model.load_state_dict(model_data["model_state_dict"])
    model.to(args.device)
    model.eval()

    # Keep calib metadata for manifest (builder is grid-driven, but we store the intent)
    split_key = "train_input" if args.calib_split == "train" else "test_input"
    X = dataset[split_key].detach().cpu().numpy().astype(np.float32, copy=False)
    rng = np.random.default_rng(args.seed)
    if X.shape[0] > args.num_samples:
        idx = rng.choice(X.shape[0], size=args.num_samples, replace=False)
        Xc = X[idx]
    else:
        Xc = X

    # Map CLI -> lut_core semantics
    oob_behavior = "clip" if args.oob_policy == "clip_x" else "zero"

    # Quant ranges (match typical “254 levels” choice for int8 symmetric)
    if args.dtype == "int8":
        qmin, qmax = -127, 127
    else:
        qmin, qmax = 0, 255

    layer_count = len(model.act_fun)

    manifest = {
        "run_dir": str(run_dir),
        "L": args.L,
        "value_repr": args.value_repr,
        "scheme": args.scheme,
        "dtype": args.dtype,
        "interp": args.interp,
        "boundary_mode": args.boundary_mode,
        "oob_policy": args.oob_policy,
        "calibration": {
            "split": args.calib_split,
            "num_samples": int(Xc.shape[0]),
            "seed": int(args.seed),
        },
        "device": args.device,
        "layers": [],
    }

    for layer_idx in range(layer_count):
        adapter = PyKANSingleLayerAdapter(model, layer_idx=layer_idx, device=args.device)
        edges = adapter.extract_edges()

        art = build_lut_for_edges(
            edges=edges,
            L=args.L,
            interp=args.interp,
            y_range_method="minmax",
            lower_pct=0.1,
            upper_pct=99.9,
            dtype=args.dtype,
            scheme=args.scheme,
            qmin=qmin,
            qmax=qmax,
            meta_dtype="float16",
            value_representation=args.value_repr,
            oob_behavior=oob_behavior,
            boundary_mode=args.boundary_mode,
        )

        # Save layer artifact (lut_core format)
        layer_path = out_dir / (
            f"layer{layer_idx}_L{args.L}_{args.scheme}_{args.dtype}_{args.interp}_"
            f"{args.boundary_mode}_{args.oob_policy}_{args.value_repr}.npz"
        )
        save_lut_npz(layer_path, art)

        bytes_total = _bytes_total_npz_payload(art)
        print(f"[OK] layer {layer_idx}: saved {layer_path.name}, bytes_total={bytes_total}")

        manifest["layers"].append(
            {
                "layer_idx": int(layer_idx),
                "path": str(layer_path),
                "in_dim": int(adapter.in_dim),
                "out_dim": int(adapter.out_dim),
                "bytes_total": int(bytes_total),
            }
        )

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
