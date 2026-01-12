# src/lut_v2/sweep_all.py
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional


@dataclass(frozen=True)
class QuantSpec:
    scheme: str  # symmetric | asymmetric
    dtype: str   # int8 | uint8

    @property
    def tag(self) -> str:
        return f"{self.scheme}_{self.dtype}"


def _parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out


def _parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _run(cmd: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> None:
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None, env=env)


def _safe_tag(s: str) -> str:
    # minimal sanitization for Windows paths
    return s.replace(" ", "_").replace(":", "-").replace("/", "_").replace("\\", "_")


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--run-dir", required=True, help="Path to run directory containing trained_model.pt and dataset.pt")
    ap.add_argument("--out-root", default=None, help="Default: <run-dir>/lut_sweeps_v2")
    ap.add_argument("--device", default="cpu")

    # Sweep grid
    ap.add_argument("--Ls", default="4,8,16,32,64,128,256")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--boundary-modes", default="closed,half_open")
    ap.add_argument("--oob-policies", default="clip_x,zero_spline")
    ap.add_argument("--value-repr", default="spline_component")  # keep fixed (recommended)
    ap.add_argument("--interp", default="linear")               # keep fixed unless you want ablation
    ap.add_argument("--quant", default="symmetric:int8,asymmetric:uint8")

    # Calibration
    ap.add_argument("--calib-split", default="train", choices=["train", "test"])
    ap.add_argument("--num-samples", type=int, default=4096)

    # Evaluation protocol
    ap.add_argument("--batch-sizes", default="1,256", help="For IoT-J: include 1 (latency) and 256 (throughput)")
    ap.add_argument("--threads-torch", default="1")
    ap.add_argument("--threads-numba", default="1")
    ap.add_argument("--warmup-iters", type=int, default=50)
    ap.add_argument("--measure-iters", type=int, default=200)
    ap.add_argument("--threshold", type=float, default=0.5)

    # Behavior
    ap.add_argument("--force", action="store_true", help="Recompute even if report exists")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--python", default=sys.executable, help="Python executable (default: current)")

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if args.out_root is None:
        out_root = run_dir / "lut_sweeps_v2"
    else:
        out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Parse lists
    Ls = _parse_int_list(args.Ls)
    seeds = _parse_int_list(args.seeds)
    boundary_modes = _parse_str_list(args.boundary_modes)
    oob_policies = _parse_str_list(args.oob_policies)
    batch_sizes = _parse_int_list(args.batch_sizes)
    threads_torch_list = _parse_int_list(args.threads_torch)
    threads_numba_list = _parse_int_list(args.threads_numba)

    # Parse quant specs
    quants: List[QuantSpec] = []
    for q in _parse_str_list(args.quant):
        # format: scheme:dtype
        if ":" not in q:
            raise ValueError(f"--quant item must be scheme:dtype, got: {q}")
        scheme, dtype = q.split(":", 1)
        scheme = scheme.strip()
        dtype = dtype.strip()
        quants.append(QuantSpec(scheme=scheme, dtype=dtype))

    index_path = out_root / "sweep_index.csv"
    fieldnames = [
        "status",
        "tag",
        "seed",
        "L",
        "scheme",
        "dtype",
        "interp",
        "boundary_mode",
        "oob_policy",
        "value_repr",
        "calib_split",
        "num_samples",
        "lut_dir",
        "eval_report",
        "build_seconds",
        "error",
    ]

    # Prepare CSV (append if exists)
    write_header = not index_path.exists()
    with index_path.open("a", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        total = 0
        ok = 0

        for seed in seeds:
            for L in Ls:
                for q in quants:
                    for bm in boundary_modes:
                        for oob in oob_policies:
                            total += 1

                            tag = _safe_tag(
                                f"seed{seed}_L{L}_{q.tag}_{args.interp}_{bm}_{oob}_{args.value_repr}"
                            )
                            lut_dir = out_root / tag
                            lut_dir.mkdir(parents=True, exist_ok=True)

                            # We will generate one report per (batch_size, threads_torch, threads_numba)
                            # but also keep a "default" pointer.
                            # For paper: batch=1, threads=1 is primary.
                            # We store reports under: <lut_dir>/eval/bs{bs}_tt{tt}_tn{tn}/lut_report_unified.json
                            try:
                                # Skip if already computed (for primary setting only), unless --force
                                primary_report = (
                                    lut_dir / "eval" / f"bs{batch_sizes[0]}_tt{threads_torch_list[0]}_tn{threads_numba_list[0]}"
                                    / "lut_report_unified.json"
                                )
                                if primary_report.exists() and not args.force:
                                    writer.writerow({
                                        "status": "SKIP",
                                        "tag": tag,
                                        "seed": seed,
                                        "L": L,
                                        "scheme": q.scheme,
                                        "dtype": q.dtype,
                                        "interp": args.interp,
                                        "boundary_mode": bm,
                                        "oob_policy": oob,
                                        "value_repr": args.value_repr,
                                        "calib_split": args.calib_split,
                                        "num_samples": args.num_samples,
                                        "lut_dir": str(lut_dir),
                                        "eval_report": str(primary_report),
                                        "build_seconds": "",
                                        "error": "",
                                    })
                                    continue

                                # --------------------
                                # 1) BUILD LUT
                                # --------------------
                                t0 = time.perf_counter()
                                _run([
                                    args.python,
                                    "-m",
                                    "src.lut_v2.build_lut",
                                    "--run-dir", str(run_dir),
                                    "--out", str(lut_dir),
                                    "--L", str(L),
                                    "--value-repr", args.value_repr,
                                    "--scheme", q.scheme,
                                    "--dtype", q.dtype,
                                    "--interp", args.interp,
                                    "--boundary-mode", bm,
                                    "--oob-policy", oob,
                                    "--calib-split", args.calib_split,
                                    "--num-samples", str(args.num_samples),
                                    "--seed", str(seed),
                                    "--device", args.device,
                                ])
                                t1 = time.perf_counter()
                                build_s = t1 - t0

                                # --------------------
                                # 2) EVALUATE (multiple batch sizes / threads)
                                # --------------------
                                for bs in batch_sizes:
                                    for tt in threads_torch_list:
                                        for tn in threads_numba_list:
                                            eval_dir = lut_dir / "eval" / f"bs{bs}_tt{tt}_tn{tn}"
                                            eval_dir.mkdir(parents=True, exist_ok=True)

                                            # evaluator writes into lut_dir by default;
                                            # we want it under eval_dir, so we run it and then move/copy the produced file.
                                            _run([
                                                args.python,
                                                "-m",
                                                "src.lut_v2.evaluate_lut",
                                                "--run-dir", str(run_dir),
                                                "--lut-dir", str(lut_dir),
                                                "--device", args.device,
                                                "--threads-torch", str(tt),
                                                "--threads-numba", str(tn),
                                                "--batch-size", str(bs),
                                                "--warmup-iters", str(args.warmup_iters),
                                                "--measure-iters", str(args.measure_iters),
                                                "--threshold", str(args.threshold),
                                            ])

                                            produced = lut_dir / "lut_report_unified.json"
                                            if not produced.exists():
                                                raise RuntimeError(f"evaluate_lut did not produce {produced}")

                                            target = eval_dir / "lut_report_unified.json"
                                            # overwrite
                                            target.write_text(produced.read_text(encoding="utf-8"), encoding="utf-8")

                                ok += 1
                                writer.writerow({
                                    "status": "OK",
                                    "tag": tag,
                                    "seed": seed,
                                    "L": L,
                                    "scheme": q.scheme,
                                    "dtype": q.dtype,
                                    "interp": args.interp,
                                    "boundary_mode": bm,
                                    "oob_policy": oob,
                                    "value_repr": args.value_repr,
                                    "calib_split": args.calib_split,
                                    "num_samples": args.num_samples,
                                    "lut_dir": str(lut_dir),
                                    "eval_report": str(primary_report),
                                    "build_seconds": f"{build_s:.3f}",
                                    "error": "",
                                })
                                print(f"[OK] {tag}")

                            except Exception as e:
                                writer.writerow({
                                    "status": "FAIL",
                                    "tag": tag,
                                    "seed": seed,
                                    "L": L,
                                    "scheme": q.scheme,
                                    "dtype": q.dtype,
                                    "interp": args.interp,
                                    "boundary_mode": bm,
                                    "oob_policy": oob,
                                    "value_repr": args.value_repr,
                                    "calib_split": args.calib_split,
                                    "num_samples": args.num_samples,
                                    "lut_dir": str(lut_dir),
                                    "eval_report": "",
                                    "build_seconds": "",
                                    "error": repr(e),
                                })
                                print(f"[FAIL] {tag}: {e}")
                                if args.stop_on_error:
                                    raise

        print(f"[DONE] total={total}, ok={ok}, out_root={out_root}")
        print(f"[INDEX] {index_path}")


if __name__ == "__main__":
    main()
