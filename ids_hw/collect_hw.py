"""
Flash LUT-KAN firmware and collect benchmark results from ESP32-C3.
Python replacement for flash_lutkan.ps1 — no PowerShell issues.

Usage:
    python collect_hw.py                      # auto-detect port, monitor mode
    python collect_hw.py --collect 20         # collect 20 cycles, print stats
    python collect_hw.py --port COM4 --collect 20
    python collect_hw.py --skip-flash --collect 20
    python collect_hw.py --monitor            # just open serial monitor

Requirements:  pip install pyserial
PlatformIO must be on PATH (it is if pio works in your terminal).
"""

import argparse
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Serial port helpers
# ---------------------------------------------------------------------------
def list_ports():
    try:
        import serial.tools.list_ports
        return [p.device for p in serial.tools.list_ports.comports()]
    except Exception:
        return []


def auto_detect_port():
    try:
        import serial.tools.list_ports
        candidates = []
        for p in serial.tools.list_ports.comports():
            desc = (p.description or "") + (p.manufacturer or "")
            if any(x in desc for x in ["USB", "CH34", "CP21", "FTDI", "ESP32"]):
                candidates.append(p.device)
        return candidates[0] if candidates else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Flash via PlatformIO
# ---------------------------------------------------------------------------
def flash(env: str, port: str, script_dir: Path):
    cmd = ["pio", "run", "-e", env, "-t", "upload", "--upload-port", port]
    print(f"  >> Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(script_dir))
    if result.returncode != 0:
        print(" ERR Build/upload failed")
        sys.exit(1)
    print("  OK Flashed successfully")


# ---------------------------------------------------------------------------
# Collect serial data
# ---------------------------------------------------------------------------
def collect(port: str, baud: int, n_cycles: int, timeout_s: int = 180):
    try:
        import serial
    except ImportError:
        print("ERROR: pyserial not installed. Run: pip install pyserial")
        sys.exit(1)

    print(f"  >> Opening {port} at {baud} baud...")
    ser = serial.Serial(port, baud, timeout=8)
    time.sleep(1.0)

    rows_needed = n_cycles * 2
    rows = []
    meta = []
    deadline = time.time() + timeout_s

    print(f"  >> Collecting {rows_needed} rows ({n_cycles} cycles)...")
    while len(rows) < rows_needed and time.time() < deadline:
        try:
            line = ser.readline().decode("utf-8", errors="replace").strip()
            if line.startswith("ATTACK,") or line.startswith("NORMAL,"):
                rows.append(line)
                pct = len(rows) * 100 // rows_needed
                print(f"\r     {len(rows)}/{rows_needed} ({pct}%)  ", end="", flush=True)
            elif line:
                meta.append(line)
        except Exception:
            continue

    print()
    ser.close()
    return rows, meta


# ---------------------------------------------------------------------------
# Monitor mode
# ---------------------------------------------------------------------------
def monitor(port: str, baud: int):
    try:
        import serial
    except ImportError:
        print("ERROR: pyserial not installed. Run: pip install pyserial")
        sys.exit(1)

    print(f"  >> Monitor {port} @ {baud} — press Ctrl+C to exit")
    ser = serial.Serial(port, baud, timeout=2)
    try:
        while True:
            line = ser.readline().decode("utf-8", errors="replace").strip()
            if line:
                print(line)
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()


# ---------------------------------------------------------------------------
# Statistics + report
# ---------------------------------------------------------------------------
def make_report(rows, meta, script_dir: Path):
    latencies = []
    correct = 0
    total = 0
    for r in rows:
        parts = r.split(",")
        if len(parts) >= 6:
            try:
                latencies.append(int(parts[3]))
                correct += int(parts[5])
                total += 1
            except ValueError:
                pass

    if not latencies:
        print(" ERR No valid rows parsed")
        return

    mean_us = sum(latencies) / len(latencies)
    min_us  = min(latencies)
    max_us  = max(latencies)
    std_us  = math.sqrt(sum((x - mean_us)**2 for x in latencies) / len(latencies))
    hw_acc  = round(correct / total, 4) if total else 0.0

    def get_meta(key):
        for line in meta:
            if line.startswith(key + "="):
                return line.split("=", 1)[1]
        return ""

    model_name  = get_meta("MODEL")    or "LUT_KAN"
    lut_l       = get_meta("LUT_L")    or "?"
    f1_val      = get_meta("F1")       or "?"
    roc_val     = get_meta("ROC_AUC")  or "?"
    dataset_val = get_meta("DATASET")  or "test"
    size_kb     = get_meta("FLASH_KB") or get_meta("SIZE_KB") or "?"
    sram_raw    = get_meta("SRAM_USED") or ""
    try:
        sram_b = int(sram_raw)
        sram_val = f"{sram_b} B ({sram_b/1024:.1f} KB)"
    except (ValueError, TypeError):
        sram_val = sram_raw if sram_raw else "< 512 B runtime"
    board       = get_meta("BOARD") or "ESP32-C3-SuperMini"
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")

    sep  = "=" * 60
    sep2 = "-" * 60

    size_label = f"{size_kb} KB" if size_kb != "?" else "see Flash usage"
    lines = [
        sep,
        "  HARDWARE BENCHMARK RESULTS",
        sep,
        f"  Timestamp  : {ts}",
        f"  Model      : {model_name} (L={lut_l})",
        f"  Board      : {board}",
        f"  F1 (test)  : {f1_val}     ROC-AUC : {roc_val}",
        f"  Size       : {size_label}",
        f"  SRAM used  : {sram_val}",
        f"  Samples    : {total}   Accuracy on hw : {hw_acc}",
        "",
        "  Latency per inference:",
        f"    mean = {mean_us:.1f} us    std = {std_us:.1f} us",
        f"    min  = {min_us} us    max = {max_us} us",
        "",
        sep2,
        "  PAPER-READY BLOCK",
        sep2,
        f"  Model: {model_name} (L={lut_l}) | Board: {board}",
        f"  Inference latency: {mean_us:.1f} +/- {std_us:.1f} us (min {min_us}, max {max_us})",
        f"  SRAM usage: {sram_val}",
        f"  F1 ({dataset_val} test set): {f1_val}  |  ROC-AUC: {roc_val}",
        f"  Model size: {size_label}",
        sep2,
    ]

    report_text = "\n".join(lines)
    print("\n" + report_text + "\n")

    results_dir = script_dir / "results"
    results_dir.mkdir(exist_ok=True)
    fname = f"{ts}_{board}_{model_name}_L{lut_l}_report.txt"
    report_path = results_dir / fname
    report_path.write_text(report_text, encoding="utf-8")
    print(f"  OK Report saved: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Flash LUT-KAN to ESP32-C3 and collect benchmarks")
    ap.add_argument("--port",       default="",    help="COM port (auto-detected if omitted)")
    ap.add_argument("--collect",    type=int, default=0, help="Number of cycles to collect")
    ap.add_argument("--monitor",    action="store_true",  help="Open serial monitor")
    ap.add_argument("--skip-flash", action="store_true",  help="Skip pio build/upload")
    ap.add_argument("--env",        default="esp32_lut_kan", help="PlatformIO environment name")
    ap.add_argument("--baud",       type=int, default=115200)
    args = ap.parse_args()

    script_dir = Path(__file__).parent.resolve()

    # Resolve port
    port = args.port
    if not port:
        port = auto_detect_port()
        if port:
            print(f"  OK Auto-detected port: {port}")
        else:
            ports = list_ports()
            print(f" ERR Could not auto-detect port.")
            print(f"     Available: {ports if ports else 'none found'}")
            print(f"     Use: python collect_hw.py --port COM<N>")
            sys.exit(1)

    # Flash
    if not args.skip_flash:
        flash(args.env, port, script_dir)
        time.sleep(2)

    # Monitor or collect
    if args.monitor:
        monitor(port, args.baud)
    elif args.collect > 0:
        rows, meta = collect(port, args.baud, args.collect)
        if rows:
            make_report(rows, meta, script_dir)
        else:
            print(" ERR No data received. Try --monitor to see raw output.")
    else:
        print("  >> Nothing to do. Use --collect 20 or --monitor")
        print("     Example: python collect_hw.py --collect 20")


if __name__ == "__main__":
    main()
