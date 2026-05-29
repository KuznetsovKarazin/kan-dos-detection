"""
Print TON_IoT LUT sweep summary table for the paper.
Usage: python scripts/toniot_summary.py
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

TON_RUN = Path(r"experiment_data\runs\20260522_123241_seed42_TON_IoT_binary_w32-16_g5_k3")
root = TON_RUN / "lut_sweeps_v2"

# Float baseline from any report
sample = next(root.glob("*/eval/bs1_tt1_tn1/lut_report_unified.json"))
d0 = json.loads(sample.read_text())
float_f1   = d0["quality"]["float"]["f1"]
float_acc  = d0["quality"]["float"]["accuracy"]
float_rec  = d0["quality"]["float"]["recall"]
float_prec = d0["quality"]["float"]["precision"]
float_roc  = d0["quality"]["float"]["roc_auc"]
float_ms1  = d0["timing"]["infer_only"]["float_pytorch"]["ms_per_sample"]

print("=" * 70)
print("TON_IoT — Float baseline (KAN, no LUT)")
print("=" * 70)
print(f"  Accuracy  : {float_acc:.4f}")
print(f"  Precision : {float_prec:.4f}")
print(f"  Recall    : {float_rec:.4f}")
print(f"  F1        : {float_f1:.4f}")
print(f"  ROC-AUC   : {float_roc:.4f}")
print(f"  ms/sample (bs=1, PyTorch): {float_ms1:.2f}")
print()

# Collect per-L stats (symmetric int8 only, matching paper Table III style)
data = defaultdict(list)
for p in root.glob("*/eval/bs1_tt1_tn1/lut_report_unified.json"):
    d = json.loads(p.read_text())
    cfg = d.get("lut_config", {})
    if cfg.get("scheme") != "symmetric":
        continue
    L = cfg["L"]
    q  = d["quality"]["lut_numba"]
    t1 = d["timing"]["infer_only"]["lut_numba_packed"]["ms_per_sample"]
    ft = d["timing"]["infer_only"]["float_pytorch"]["ms_per_sample"]

    p256 = p.parent.parent / "bs256_tt1_tn1" / "lut_report_unified.json"
    if p256.exists():
        d256  = json.loads(p256.read_text())
        t256  = d256["timing"]["infer_only"]["lut_numba_packed"]["ms_per_sample"]
        ft256 = d256["timing"]["infer_only"]["float_pytorch"]["ms_per_sample"]
    else:
        t256 = ft256 = None

    data[L].append({
        "acc": q["accuracy"], "prec": q["precision"],
        "rec": q["recall"],   "f1":   q["f1"],
        "roc": q["roc_auc"],
        "ms1": t1, "ms256": t256, "ft1": ft, "ft256": ft256,
    })

print("=" * 90)
print("Detection quality (sym int8, mean over 5 seeds)  — Table III style")
print("=" * 90)
header = f"{'L':>4}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'dF1':>7}  {'ROC':>7}"
print(header)
print("-" * len(header))
for L in sorted(data.keys()):
    v = data[L]
    acc  = statistics.mean([x["acc"]  for x in v])
    prec = statistics.mean([x["prec"] for x in v])
    rec  = statistics.mean([x["rec"]  for x in v])
    f1   = statistics.mean([x["f1"]   for x in v])
    roc  = statistics.mean([x["roc"]  for x in v])
    df1  = f1 - float_f1
    print(f"{L:>4}  {acc:.4f}   {prec:.4f}   {rec:.4f}   {f1:.4f}  {df1:>+.4f}   {roc:.4f}")

print()
print("=" * 90)
print("Inference latency (sym int8, mean over 5 seeds)  — Table V/VI style")
print("=" * 90)
header2 = f"{'L':>4}  {'ms bs=1':>10}  {'spdup bs=1':>12}  {'ms bs=256':>11}  {'spdup bs=256':>14}"
print(header2)
print("-" * len(header2))
for L in sorted(data.keys()):
    v = data[L]
    ms1   = statistics.mean([x["ms1"] for x in v])
    ft1   = statistics.mean([x["ft1"] for x in v])
    spd1  = ft1 / ms1

    has256 = [x for x in v if x["ms256"] is not None]
    if has256:
        ms256  = statistics.mean([x["ms256"]  for x in has256])
        ft256  = statistics.mean([x["ft256"]  for x in has256])
        spd256 = ft256 / ms256
        ms256_s  = f"{ms256:.5f}"
        spd256_s = f"{spd256:>8.0f}x"
    else:
        ms256_s = spd256_s = "    n/a   "

    print(f"{L:>4}  {ms1:.5f}     {spd1:>8.0f}x   {ms256_s}    {spd256_s}")

# Also print L=8 highlighted (recommended default)
print()
v8 = data.get(8, [])
if v8:
    f1_8  = statistics.mean([x["f1"]  for x in v8])
    roc_8 = statistics.mean([x["roc"] for x in v8])
    ms1_8 = statistics.mean([x["ms1"] for x in v8])
    ft1_8 = statistics.mean([x["ft1"] for x in v8])
    print("=" * 50)
    print("RECOMMENDED CONFIG L=8 (for export_lut_c_header):")
    print(f"  F1      = {f1_8:.4f}")
    print(f"  ROC-AUC = {roc_8:.4f}")
    print(f"  Speedup bs=1 = {ft1_8/ms1_8:.0f}x")
    print("=" * 50)
