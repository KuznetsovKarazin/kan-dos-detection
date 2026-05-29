# src/lut_v2/sweep_all.py
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any


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


def _safe_tag(s: str) -> str:
    return s.replace(" ", "_").replace(":", "-").replace("/", "_").replace("\\", "_")


def _run(cmd: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> None:
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None, env=env)


def _json_load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_dump(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _iters_for_batch(bs: int, warmup_default: int, measure_default: int,
                    warmup_small: int, measure_small: int) -> tuple[int, int]:
    """
    Batch=1..4 is extremely overhead-sensitive; use larger iters to reduce timer noise.
    """
    if bs <= 4:
        return warmup_small, measure_small
    return warmup_default, measure_default


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--run-dir", required=True, help="Run directory containing trained_model.pt and dataset.pt")
    ap.add_argument("--out-root", default=None, help="Default: <run-dir>/lut_sweeps_v2")
    ap.add_argument("--device", default="cpu")

    # Sweep grid
    ap.add_argument("--Ls", default="4,8,16,32,64,128,256")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--boundary-modes", default="closed,half_open")
    ap.add_argument("--oob-policies", default="clip_x,zero_spline")
    ap.add_argument("--value-repr", default="spline_component")
    ap.add_argument("--interp", default="linear")
    ap.add_argument("--quant", default="symmetric:int8,asymmetric:uint8")

    # Calibration
    ap.add_argument("--calib-split", default="train", choices=["train", "test"])
    ap.add_argument("--num-samples", type=int, default=4096)

    # Evaluation protocol
    ap.add_argument("--batch-sizes", default="1,256")
    ap.add_argument("--threads-torch", default="1")
    ap.add_argument("--threads-numba", default="1")
    ap.add_argument("--warmup-iters", type=int, default=50)
    ap.add_argument("--measure-iters", type=int, default=200)

    # Strongly recommended for bs=1..4
    ap.add_argument("--warmup-iters-small-batch", type=int, default=1000)
    ap.add_argument("--measure-iters-small-batch", type=int, default=20000)

    ap.add_argument("--threshold", type=float, default=0.5)

    # Behavior
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--python", default=sys.executable)

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_root = (run_dir / "lut_sweeps_v2") if args.out_root is None else Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    Ls = _parse_int_list(args.Ls)
    seeds = _parse_int_list(args.seeds)
    boundary_modes = _parse_str_list(args.boundary_modes)
    oob_policies = _parse_str_list(args.oob_policies)
    batch_sizes = _parse_int_list(args.batch_sizes)
    threads_torch_list = _parse_int_list(args.threads_torch)
    threads_numba_list = _parse_int_list(args.threads_numba)

    quants: List[QuantSpec] = []
    for q in _parse_str_list(args.quant):
        if ":" not in q:
            raise ValueError(f"--quant item must be scheme:dtype, got: {q}")
        scheme, dtype = q.split(":", 1)
        quants.append(QuantSpec(scheme=scheme.strip(), dtype=dtype.strip()))

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
        "primary_eval_report",
        "build_seconds",
        "error",
    ]

    write_header = not index_path.exists()
    with index_path.open("a", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        total = 0
        ok = 0

        # Primary setting for quick existence check in --force/skip logic
        primary_bs = batch_sizes[0]
        primary_tt = threads_torch_list[0]
        primary_tn = threads_numba_list[0]

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

                            primary_report = (
                                lut_dir / "eval" / f"bs{primary_bs}_tt{primary_tt}_tn{primary_tn}" / "lut_report_unified.json"
                            )

                            try:
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
                                        "primary_eval_report": str(primary_report),
                                        "build_seconds": "",
                                        "error": "",
                                    })
                                    continue

                                # 1) BUILD LUT
                                t0 = time.perf_counter()
                                _run([
                                    args.python, "-m", "src.lut_v2.build_lut",
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
                                build_s = time.perf_counter() - t0

                                # 2) EVALUATE
                                # IMPORTANT: evaluate_lut writes lut_report_unified.json into lut_dir.
                                # We immediately copy+augment it into eval_dir to avoid accidental overwrites.
                                produced = lut_dir / "lut_report_unified.json"

                                for bs in batch_sizes:
                                    warmup_it, measure_it = _iters_for_batch(
                                        bs,
                                        args.warmup_iters,
                                        args.measure_iters,
                                        args.warmup_iters_small_batch,
                                        args.measure_iters_small_batch,
                                    )

                                    for tt in threads_torch_list:
                                        for tn in threads_numba_list:
                                            eval_dir = lut_dir / "eval" / f"bs{bs}_tt{tt}_tn{tn}"
                                            eval_dir.mkdir(parents=True, exist_ok=True)

                                            # Optional: keep per-setting logs (useful for debugging)
                                            log_path = eval_dir / "eval_stdout.log"
                                            with log_path.open("w", encoding="utf-8") as f_log:
                                                subprocess.check_call(
                                                    [
                                                        args.python, "-m", "src.lut_v2.evaluate_lut",
                                                        "--run-dir", str(run_dir),
                                                        "--lut-dir", str(lut_dir),
                                                        "--device", args.device,
                                                        "--threads-torch", str(tt),
                                                        "--threads-numba", str(tn),
                                                        "--batch-size", str(bs),
                                                        "--warmup-iters", str(warmup_it),
                                                        "--measure-iters", str(measure_it),
                                                        "--threshold", str(args.threshold),
                                                    ],
                                                    stdout=f_log,
                                                    stderr=subprocess.STDOUT,
                                                )

                                            if not produced.exists():
                                                raise RuntimeError(f"evaluate_lut did not produce: {produced}")

                                            report = _json_load(produced)

                                            # Augment with sweep metadata (so collector can trust it)
                                            report.setdefault("sweep", {})
                                            report["sweep"].update({
                                                "tag": tag,
                                                "seed": seed,
                                                "L": L,
                                                "scheme": q.scheme,
                                                "dtype": q.dtype,
                                                "interp": args.interp,
                                                "boundary_mode": bm,
                                                "oob_policy": oob,
                                                "value_repr": args.value_repr,
                                                "batch_size": bs,
                                                "threads_torch": tt,
                                                "threads_numba": tn,
                                                "warmup_it": warmup_it,
                                                "measure_it": measure_it,
                                            })

                                            # Mark potential confusion: reference timing might be from another batch
                                            ref = report.get("quality", {}).get("float_metrics_reference_file") or {}
                                            ref_timing = ref.get("timing", {})
                                            #ref = report.get("quality", {}).get("float_metrics_reference_file", {})
                                            #ref_timing = ref.get("timing", {})
                                            if isinstance(ref_timing, dict) and "batch_size" in ref_timing:
                                                if int(ref_timing["batch_size"]) != int(bs):
                                                    report.setdefault("notes", [])
                                                    report["notes"].append(
                                                        f"float_metrics_reference_file.timing.batch_size={ref_timing['batch_size']} "
                                                        f"differs from current eval batch_size={bs}. Use timing.infer_only.float_pytorch for this eval."
                                                    )

                                            target = eval_dir / "lut_report_unified.json"
                                            _json_dump(target, report)

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
                                    "primary_eval_report": str(primary_report),
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
                                    "primary_eval_report": "",
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
