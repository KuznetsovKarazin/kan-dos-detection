import json, glob
from pathlib import Path

run_dir = Path(r"experiment_data\runs\20260522_123241_seed42_TON_IoT_binary_w32-16_g5_k3")
reports = list(run_dir.glob("lut_sweeps_v2/*/lut_report_unified.json"))
print(f"Found {len(reports)} reports")
for p in sorted(reports)[:5]:
    d = json.loads(p.read_text())
    cfg = d.get("lut_config", {})
    q = d.get("quality", {}).get("lut_numba", {})
    t = d.get("timing", {}).get("infer_only", {}).get("lut_numba_packed", {})
    print(f"L={cfg.get('L')} {cfg.get('boundary_mode')} {cfg.get('oob_policy')}: "
          f"F1={q.get('f1','?'):.4f}  "
          f"ms/sample={t.get('ms_per_sample','?'):.4f}")