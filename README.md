# LUT-KAN: Lookup-Table Compiled KAN for IoT Intrusion Detection

> **First deployment of a Kolmogorov-Arnold Network on microcontroller hardware.**
> LUT-compiled KAN runs on an ESP32-C3 (RISC-V, 320 KB SRAM) with **18 KB runtime SRAM** and on an Arduino Mega 2560 (AVR, 8 KB SRAM) with only **2 KB runtime SRAM** — no Python, no PyTorch, no deep-learning runtime.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PlatformIO](https://img.shields.io/badge/PlatformIO-ESP32--C3%20%7C%20AVR-orange)](https://platformio.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20IoT%20Journal-blue)](https://doi.org/10.1016/j.neucom.2026.133774)

---

## Overview

Kolmogorov-Arnold Networks (KANs) achieve competitive accuracy with fewer parameters and offer **post-hoc symbolic interpretability** — but their runtime B-spline evaluation makes them impractical on edge hardware.

This repository implements **LUT-KAN**: a post-training compilation pipeline that replaces every B-spline edge function with a precomputed, quantized lookup table and linear interpolation. The result is a self-contained **C inference engine** with no Python dependency, enabling KAN deployment on microcontrollers for the first time.

### Key Results at a Glance

#### Software (AMD Ryzen 7 7840HS, single-threaded Numba)

| Dataset | Architecture | Float F1 | LUT L=8 F1 | Speedup (bs=1) | Speedup (bs=256) |
|---|---|---|---|---|---|
| **CICIDS2017** (DoS, 78 feat.) | 78→32→16→1 | 0.9900 | **0.9896** (ΔF1 −0.0004) | **6333×** | **63×** |
| **TON_IoT** (10 attacks, 91 feat.) | 91→32→16→1 | 0.9869 | **0.9867** (ΔF1 −0.0002) | **6193×** | **67×** |

#### Hardware (physical MCU measurements)

| Model | Board | Latency | SRAM | Flash | F1 |
|---|---|---|---|---|---|
| LUT-KAN L=8 | ESP32-C3-SuperMini | 25.8 ms | **18 KB** | 519 KB | 0.9896 |
| LUT-KAN L=2 | ESP32-C3-SuperMini | 19.4 ms | 212 KB | 324 KB | 0.9874 |
| LUT-KAN L=2 | **Arduino Mega 2560** | 246 ms | **2 KB** | 209 KB | 0.9874 |
| LUT-KAN L=8 (TON_IoT) | ESP32-C3-SuperMini | 29.2 ms | **18 KB** | 591 KB | 0.9867 |
| MLP TFLite INT8 (baseline) | ESP32-C3-SuperMini | 0.8 ms | 398 KB | 13 KB | 0.9959 |
| LightGBM INT8 (baseline) | ESP32-C3-SuperMini | 5.6 ms | 104 KB | 74 KB | 0.9992 |
| XGBoost INT8 (baseline) | ESP32-C3-SuperMini | 8.2 ms | 98 KB | 370 KB | 0.9989 |

> **Why LUT-KAN on MCU?** LUT-KAN uses only 18 KB runtime SRAM (Flash/XIP mode) vs 398 KB for MLP TFLite — enabling KAN deployment on devices where neural IDS methods cannot run at all. On the Arduino Mega 2560 (8 KB total SRAM), LUT-KAN is the **only** neural-architecture IDS that fits.

---

## Repository Structure

```
kan-dos-detection/
├── src/
│   ├── train.py                    # Train KAN on CICIDS2017
│   ├── analyze.py                  # Evaluate float model
│   ├── feature_analysis.py         # Feature importance / correlation
│   └── lut_v2/
│       ├── build_lut.py            # Compile trained KAN → LUT artifacts (.npz)
│       ├── evaluate_lut.py         # Quality + latency + memory report (JSON)
│       └── sweep_all.py            # Full resolution/scheme/policy sweep
│
├── scripts/
│   ├── train_toniot.py             # Train KAN on TON_IoT dataset
│   ├── export_lut_c_header.py      # Generate C header for ESP32 or AVR
│   ├── collect_results.py          # Aggregate sweep → paper-ready CSV tables
│   ├── toniot_summary.py           # Print TON_IoT sweep summary table
│   └── analyze_toniot.py           # Generate TON_IoT analysis figures
│
├── ids_hw/                         # Embedded hardware deployment
│   ├── platformio.ini              # PlatformIO: esp32_lut_kan + mega_lut_kan envs
│   ├── collect_hw.py               # Flash + collect latency measurements
│   ├── ids_esp32_lut_kan/
│   │   └── main.cpp                # ESP32-C3 inference firmware
│   ├── ids_mega_lut_kan/
│   │   └── main.cpp                # Arduino Mega 2560 inference firmware
│   └── results/                    # Hardware benchmark reports (.txt)
│
├── experiment_data/
│   └── runs/
│       ├── 20260102_..._DoS_Hulk_w32-16_g5_k3/    # CICIDS2017 run
│       │   ├── dataset.pt
│       │   ├── trained_model.pt
│       │   ├── lut_sweeps_v2/       # LUT sweep results
│       │   ├── lut_phi_L2/          # phi-repr LUT for ESP32/Mega
│       │   ├── lut_phi_L8/          # phi-repr LUT L=8 for ESP32
│       │   ├── analysis/            # Figures, reports
│       │   └── lut_results_summary.csv/
│       └── 20260522_..._TON_IoT_binary_w32-16_g5_k3/   # TON_IoT run
│
├── data/                            # Dataset CSV (not committed)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/KuznetsovKarazin/kan-dos-detection.git
cd kan-dos-detection
git checkout lut-v2

python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows PowerShell:
venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

**Hardware deployment** (optional):
```bash
pip install platformio pyserial
```

---

## Datasets

### CICIDS2017 — DoS Scenario

Download from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html) and place the Wednesday traffic file:

```
data/Wednesday-workingHours.pcap_ISCX.csv
```

### TON_IoT Network Dataset

Download `train_test_network.csv` from the [TON_IoT dataset page](https://research.unsw.edu.au/projects/toniot-datasets) and place it:

```
data/train_test_network.csv
```

---

## Usage

### Step 1 — Train the KAN Model

**CICIDS2017:**
```bash
python src/train.py
# Output: experiment_data/runs/<RUN_ID>/
```

**TON_IoT:**
```bash
python scripts/train_toniot.py --csv data/train_test_network.csv
# Output: experiment_data/runs/<RUN_ID_TONIOT>/
```

### Step 2 — Run the LUT Sweep

Sweeps all combinations of L ∈ {2,4,8,16,32,64,128,256}, quantization schemes,
boundary modes, and OOB policies:

```bash
# CICIDS2017 — full sweep (recommended settings):
python -m src.lut_v2.sweep_all \
    --run-dir experiment_data/runs/<RUN_ID> \
    --Ls 2,4,8,16,32,64,128,256 \
    --seeds 0,1,2,3,4 \
    --boundary-modes half_open \
    --oob-policies zero_spline \
    --batch-sizes 1,256 \
    --warmup-iters 10 --measure-iters 100 \
    --warmup-iters-small-batch 50 --measure-iters-small-batch 200
```

```bash
# TON_IoT — same command, different run-dir:
python -m src.lut_v2.sweep_all \
    --run-dir experiment_data/runs/<RUN_ID_TONIOT> \
    ...
```

### Step 3 — Collect Results

```bash
python scripts/collect_results.py \
    --sweep-root experiment_data/runs/<RUN_ID>/lut_sweeps_v2 \
    --out experiment_data/runs/<RUN_ID>/lut_results_summary.csv

# TON_IoT summary table:
python scripts/toniot_summary.py
```

---

## Embedded Hardware Deployment

The `ids_hw/` directory contains the full embedded pipeline.

### Prerequisites

- [PlatformIO](https://platformio.org/) (CLI or VS Code extension)
- `pip install pyserial`
- ESP32-C3-SuperMini and/or Arduino Mega 2560

### Step 4 — Generate C Inference Header

**ESP32-C3 (CICIDS2017, L=8, Flash mode — recommended):**
```bash
python scripts/export_lut_c_header.py \
    --run-dir experiment_data/runs/<RUN_ID> \
    --lut-dir experiment_data/runs/<RUN_ID>/lut_phi_L8 \
    --out ids_hw/ids_esp32_lut_kan/lut_kan_model.h \
    --dataset-name CICIDS2017 --f1 0.9896 --roc-auc 0.9991
```

**Arduino Mega 2560 (CICIDS2017, L=2, AVR PROGMEM):**
```bash
python scripts/export_lut_c_header.py \
    --run-dir experiment_data/runs/<RUN_ID> \
    --lut-dir experiment_data/runs/<RUN_ID>/lut_phi_L2 \
    --out ids_hw/ids_mega_lut_kan/lut_kan_model.h \
    --target avr \
    --dataset-name CICIDS2017 --f1 0.9874 --roc-auc 0.9991
```

**ESP32-C3 SRAM mode (L=2, fastest):**
```bash
python scripts/export_lut_c_header.py \
    --run-dir experiment_data/runs/<RUN_ID> \
    --lut-dir experiment_data/runs/<RUN_ID>/lut_phi_L2 \
    --out ids_hw/ids_esp32_lut_kan/lut_kan_model.h \
    --sram --dataset-name CICIDS2017 --f1 0.9874 --roc-auc 0.9991
```

### Step 5 — Flash and Collect Benchmarks

```bash
cd ids_hw

# ESP32-C3:
python collect_hw.py --env esp32_lut_kan --collect 20

# Arduino Mega 2560:
python collect_hw.py --env mega_lut_kan --collect 20

# Specify port manually if auto-detection fails:
python collect_hw.py --env esp32_lut_kan --port COM5 --collect 20
```

Results are saved to `ids_hw/results/<TIMESTAMP>_<BOARD>_<MODEL>_report.txt`.

---

## Replicating Paper Results

### Table III — LUT Quality (both datasets)

```bash
# After completing Steps 1-3 for both datasets:
python scripts/toniot_summary.py   # TON_IoT
python scripts/collect_results.py --sweep-root <CICIDS_RUN>/lut_sweeps_v2 --out summary/  # CICIDS2017
```

### Table IX — Hardware Benchmarks

Run Steps 4-5 for each combination:

| Config | `--lut-dir` | `--target` | `--sram` | Board |
|---|---|---|---|---|
| CICIDS L=8 Flash | `lut_phi_L8` | esp32 | no | ESP32-C3 |
| CICIDS L=2 SRAM | `lut_phi_L2` | esp32 | yes | ESP32-C3 |
| TON_IoT L=8 Flash | `lut_phi_L8` (TON) | esp32 | no | ESP32-C3 |
| CICIDS L=2 AVR | `lut_phi_L2` | avr | n/a | Arduino Mega |

### LUT phi Compilation (needed before export)

The phi-representation LUTs (used for hardware) are built separately:

```bash
python -m src.lut_v2.build_lut \
    --run-dir experiment_data/runs/<RUN_ID> \
    --out experiment_data/runs/<RUN_ID>/lut_phi_L8 \
    --L 8 --value-repr phi --scheme symmetric --dtype int8 \
    --interp linear --boundary-mode half_open --oob-policy zero_spline --seed 0
```

---

## Method: How LUT-KAN Works

Each KAN edge function φ_ij(x) is a B-spline with K = G + 2k = 11 segments.
LUT compilation precomputes L sample points per segment, quantizes them to int8,
and stores one float32 scale factor per segment (segment-wise quantization).

```
Trained KAN (float32 B-spline)
          │
          ▼
  build_lut.py  ──►  layer_i.npz  ┐
                  q_table [E×K×L] │  → lut_kan_model.h  →  C inference engine
                  scale   [E×K]   ┘         (AVR PROGMEM / ESP32 XIP)
```

**At inference time** (C code, no Python):
1. Find knot segment containing x
2. Compute normalized position λ ∈ [0,1]
3. Dequantize: v = scale[e,seg] × q_table[e,seg,⌊λ⌋]
4. Linear interpolation: φ = (1−λ)·v₀ + λ·v₁
5. Accumulate: h_j += φ for all input edges i

**Memory modes:**
- `--target esp32` (default): arrays live in Flash/XIP cache → 18 KB runtime SRAM
- `--target esp32 --sram`: arrays loaded to DRAM → faster on cache-miss-heavy models
- `--target avr`: PROGMEM + far-pointer addressing → 2 KB runtime SRAM on AVR

---

## Why KAN for IDS?

Unlike MLPs, KAN's learned φ_ij edge functions can be symbolically approximated
post-hoc (e.g., via `kan.auto_symbolic()`), enabling human-readable detection rules:

```python
# After training:
kan_model.auto_symbolic()
# May yield: φ(x) ≈ 0.47·sin(2.1x) + 0.23·x²
# → interpretable rule for a specific network flow feature
```

This interpretability property is preserved after LUT compilation: the LUT
faithfully approximates the original spline, and symbolic analysis can be
performed on the float model.

---

## Citation

If you use this work, please cite both:

**This paper (MCU deployment + second dataset):**
```bibtex
@article{kuznetsov2026lutkan_iot,
  author  = {Kuznetsov, Oleksandr},
  title   = {{LUT}-Compiled {Kolmogorov}-{Arnold} Networks for Lightweight {DoS} Detection on {IoT} Edge Devices},
  journal = {IEEE Internet of Things Journal},
  year    = {2026},
  note    = {Under review}
}
```

**Companion toolkit paper (CPU inference, Neurocomputing):**
```bibtex
@article{kuznetsov2026neucom,
  author  = {Kuznetsov, Oleksandr},
  title   = {{LUT-KAN}: An open software toolkit for {LUT}-compiled and quantized {CPU} inference of {Kolmogorov}--{Arnold} networks},
  journal = {Neurocomputing},
  year    = {2026},
  month   = apr,
  pages   = {133774},
  doi     = {10.1016/j.neucom.2026.133774}
}
```

---

## Hardware Photos

| ESP32-C3-SuperMini | Arduino Mega 2560 |
|:---:|:---:|
| RISC-V 160 MHz · 320 KB SRAM | AVR 16 MHz · 8 KB SRAM |
| LUT-KAN L=8: **25.8 ms**, **18 KB SRAM** | LUT-KAN L=2: **246 ms**, **2 KB SRAM** |

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Related Work

- **LUT-KAN toolkit** (Neurocomputing 2026): [doi:10.1016/j.neucom.2026.133774](https://doi.org/10.1016/j.neucom.2026.133774)
- **KAN original paper**: Liu et al., [arXiv:2404.19756](https://arxiv.org/abs/2404.19756)
- **CICIDS2017 dataset**: Sharafaldin et al., ICISSP 2018
- **TON_IoT dataset**: Alsaedi et al., IEEE Access 2020
