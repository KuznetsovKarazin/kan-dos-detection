from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _collect_reports(root: Path) -> List[Path]:
    # We collect the eval copies (not the lut_dir root file) to avoid overwrites.
    return sorted(root.glob("**/eval/**/lut_report_unified.json"))


def _flatten_one(rep: Dict[str, Any], path: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    row["report_path"] = str(path)

    sweep = rep.get("sweep", {})
    # If sweep metadata exists (from sweep_all.py), prefer it.
    row["tag"] = sweep.get("tag") or path.parent.parent.parent.name
    row["seed"] = sweep.get("seed")
    row["L"] = sweep.get("L") or _safe_get(rep, ["lut_config", "L"])

    row["scheme"] = sweep.get("scheme") or _safe_get(rep, ["lut_config", "scheme"])
    row["dtype"] = sweep.get("dtype") or _safe_get(rep, ["lut_config", "dtype"])
    row["interp"] = sweep.get("interp") or _safe_get(rep, ["lut_config", "interp"])
    row["boundary_mode"] = sweep.get("boundary_mode") or _safe_get(rep, ["lut_config", "boundary_mode"])
    row["oob_policy"] = sweep.get("oob_policy") or _safe_get(rep, ["lut_config", "oob_policy"])
    row["value_repr"] = sweep.get("value_repr") or _safe_get(rep, ["lut_config", "value_repr"])

    row["batch_size"] = sweep.get("batch_size") or _safe_get(rep, ["timing", "infer_only", "float_pytorch", "batch_size"])
    row["threads_torch"] = sweep.get("threads_torch") or _safe_get(rep, ["threads", "torch"])
    row["threads_numba"] = sweep.get("threads_numba") or _safe_get(rep, ["threads", "numba"])

    # Quality: float
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        row[f"float_{key}"] = _safe_get(rep, ["quality", "float", key])

    # Quality: LUT numpy
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        row[f"lut_numpy_{key}"] = _safe_get(rep, ["quality", "lut_numpy", key])

    # Quality: LUT numba
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        row[f"lut_numba_{key}"] = _safe_get(rep, ["quality", "lut_numba", key])

    # Timing infer-only
    row["float_ms_per_sample"] = _safe_get(rep, ["timing", "infer_only", "float_pytorch", "ms_per_sample"])
    row["lut_numpy_ms_per_sample"] = _safe_get(rep, ["timing", "infer_only", "lut_numpy_packed", "ms_per_sample"])
    row["lut_numba_ms_per_sample"] = _safe_get(rep, ["timing", "infer_only", "lut_numba_packed", "ms_per_sample"])

    # Timing end2end (diagnostic)
    row["lut_numpy_e2e_ms_per_sample"] = _safe_get(rep, ["timing", "end2end", "lut_numpy_prepare_plus_infer", "ms_per_sample"])
    row["lut_numba_e2e_ms_per_sample"] = _safe_get(rep, ["timing", "end2end", "lut_numba_prepare_plus_infer", "ms_per_sample"])

    # Memory
    row["float_params_bytes"] = _safe_get(rep, ["memory", "float_model", "params_bytes"])
    row["float_total_bytes"] = _safe_get(rep, ["memory", "float_model", "total_bytes"])
    row["lut_total_bytes"] = _safe_get(rep, ["memory", "lut_total_bytes"])
    row["lut_knots_bytes"] = _safe_get(rep, ["memory", "lut_breakdown_total", "knots_bytes"])
    row["lut_q_table_bytes"] = _safe_get(rep, ["memory", "lut_breakdown_total", "q_table_bytes"])
    row["lut_scale_bytes"] = _safe_get(rep, ["memory", "lut_breakdown_total", "scale_bytes"])
    row["lut_y_min_bytes"] = _safe_get(rep, ["memory", "lut_breakdown_total", "y_min_bytes"])

    return row


def _agg_mean_std(df: pd.DataFrame, group_cols: List[str], value_cols: List[str]) -> pd.DataFrame:
    g = df.groupby(group_cols, dropna=False)
    mean = g[value_cols].mean(numeric_only=True).add_suffix("_mean")
    std = g[value_cols].std(numeric_only=True).add_suffix("_std")
    out = pd.concat([mean, std], axis=1).reset_index()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root", required=True, help=".../lut_sweeps_v2")
    ap.add_argument("--out", default=None, help="Output directory; default: <sweep-root>/tables")
    ap.add_argument("--only-batch", default=None, help="Optional: filter by batch size, e.g. 1 or 256")
    args = ap.parse_args()

    sweep_root = Path(args.sweep_root)
    out_dir = Path(args.out) if args.out else (sweep_root / "tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = _collect_reports(sweep_root)
    if not paths:
        raise SystemExit(f"No reports found under: {sweep_root} (expected **/eval/**/lut_report_unified.json)")

    rows: List[Dict[str, Any]] = []
    for p in paths:
        rep = _read_json(p)
        rows.append(_flatten_one(rep, p))

    df = pd.DataFrame(rows)

    # Optional filter
    if args.only_batch is not None:
        df = df[df["batch_size"].astype(float) == float(args.only_batch)].copy()

    raw_path = out_dir / "raw_reports.csv"
    df.to_csv(raw_path, index=False)

    group_cols = [
        "L", "scheme", "dtype", "interp", "boundary_mode", "oob_policy", "value_repr",
        "batch_size", "threads_torch", "threads_numba",
    ]

    value_cols = [
        # quality
        "float_accuracy", "float_f1", "float_roc_auc", "float_pr_auc",
        "lut_numpy_accuracy", "lut_numpy_f1", "lut_numpy_roc_auc", "lut_numpy_pr_auc",
        "lut_numba_accuracy", "lut_numba_f1", "lut_numba_roc_auc", "lut_numba_pr_auc",
        # timing
        "float_ms_per_sample", "lut_numpy_ms_per_sample", "lut_numba_ms_per_sample",
        "lut_numpy_e2e_ms_per_sample", "lut_numba_e2e_ms_per_sample",
        # memory
        "float_total_bytes", "lut_total_bytes",
        "lut_q_table_bytes", "lut_scale_bytes", "lut_y_min_bytes", "lut_knots_bytes",
    ]

    # keep only columns that exist
    value_cols = [c for c in value_cols if c in df.columns]

    main = _agg_mean_std(df, group_cols=group_cols, value_cols=value_cols)
    main_path = out_dir / "table_main_mean_std.csv"
    main.to_csv(main_path, index=False)

    # Convenience: speed table
    speed_cols = [c for c in [
        "float_ms_per_sample", "lut_numpy_ms_per_sample", "lut_numba_ms_per_sample",
        "lut_numpy_e2e_ms_per_sample", "lut_numba_e2e_ms_per_sample",
    ] if c in df.columns]
    speed = _agg_mean_std(df, group_cols=group_cols, value_cols=speed_cols)
    speed.to_csv(out_dir / "table_speed_mean_std.csv", index=False)

    # Convenience: accuracy/AUC table
    acc_cols = [c for c in [
        "float_accuracy", "float_f1", "float_roc_auc", "float_pr_auc",
        "lut_numpy_accuracy", "lut_numpy_f1", "lut_numpy_roc_auc", "lut_numpy_pr_auc",
        "lut_numba_accuracy", "lut_numba_f1", "lut_numba_roc_auc", "lut_numba_pr_auc",
    ] if c in df.columns]
    acc = _agg_mean_std(df, group_cols=group_cols, value_cols=acc_cols)
    acc.to_csv(out_dir / "table_quality_mean_std.csv", index=False)

    # Convenience: memory table
    mem_cols = [c for c in [
        "float_total_bytes", "lut_total_bytes",
        "lut_q_table_bytes", "lut_scale_bytes", "lut_y_min_bytes", "lut_knots_bytes",
    ] if c in df.columns]
    mem = _agg_mean_std(df, group_cols=group_cols, value_cols=mem_cols)
    mem.to_csv(out_dir / "table_memory_mean_std.csv", index=False)

    print(f"[OK] Raw: {raw_path}")
    print(f"[OK] Main: {main_path}")
    print(f"[OK] Speed: {out_dir / 'table_speed_mean_std.csv'}")
    print(f"[OK] Quality: {out_dir / 'table_quality_mean_std.csv'}")
    print(f"[OK] Memory: {out_dir / 'table_memory_mean_std.csv'}")
    print(f"[DIR] {out_dir}")


if __name__ == "__main__":
    main()
