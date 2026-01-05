# src/lut_v2/sweep_lut.py
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-root", default=None, help="If omitted: <run-dir>/lut_sweeps")
    ap.add_argument("--backend", choices=["numpy", "numba"], default="numpy")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num-samples", type=int, default=4096)
    ap.add_argument("--Ls", default="16,32,64,128")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_root = Path(args.out_root) if args.out_root else (run_dir / "lut_sweeps")
    out_root.mkdir(parents=True, exist_ok=True)

    Ls = [int(x) for x in args.Ls.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    value_reprs = ["spline_component"]
    interps = ["linear"]
    boundary_modes = ["closed", "half_open"]
    oob_policies = ["clip_x", "zero_spline"]
    schemes = [("symmetric", "int8"), ("asymmetric", "uint8")]

    for seed in seeds:
        for L, (scheme, dtype), interp, bm, oob, vr in itertools.product(
            Ls, schemes, interps, boundary_modes, oob_policies, value_reprs
        ):
            tag = f"seed{seed}_L{L}_{scheme}_{dtype}_{interp}_{bm}_{oob}_{vr}"
            lut_dir = out_root / tag
            lut_dir.mkdir(parents=True, exist_ok=True)

            # build
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "src.lut_v2.build_lut",
                    "--run-dir",
                    str(run_dir),
                    "--out",
                    str(lut_dir),
                    "--L",
                    str(L),
                    "--value-repr",
                    vr,
                    "--scheme",
                    scheme,
                    "--dtype",
                    dtype,
                    "--interp",
                    interp,
                    "--boundary-mode",
                    bm,
                    "--oob-policy",
                    oob,
                    "--calib-split",
                    "train",
                    "--num-samples",
                    str(args.num_samples),
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                ]
            )

            # evaluate
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "src.lut_v2.evaluate_lut",
                    "--run-dir",
                    str(run_dir),
                    "--lut-dir",
                    str(lut_dir),
                    "--backend",
                    args.backend,
                    "--threads",
                    str(args.threads),
                    "--device",
                    args.device,
                ]
            )

            print(f"[OK] {tag}")


if __name__ == "__main__":
    main()
