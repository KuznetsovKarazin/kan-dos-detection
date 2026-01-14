# DoS Attack Detection using Kolmogorov-Arnold Networks (KAN)

## Overview

This repository implements an Intrusion Detection System (IDS) for IoT based on Kolmogorov-Arnold Networks (KAN). The goal is accurate Denial of Service (DoS) detection with low computational overhead, targeting CPU/edge deployments.

In addition to the baseline PyKAN/PyTorch inference, the repository includes an optional LUT-KAN (v2) pipeline that compiles KAN spline components into segment-wise lookup tables (LUTs) and evaluates them with NumPy or Numba backends.

## Key Features

Baseline IDS (KAN):
- Lightweight KAN architecture (≈50K parameters; small float model footprint)
- High detection accuracy on CICIDS2017 DoS traffic (see Results)
- Training, evaluation, and feature analysis scripts

LUT-KAN (v2) acceleration:
- LUT compilation from a trained KAN run directory
- Two execution backends for LUT inference:
  - NumPy vectorized evaluation
  - Numba JIT evaluation (fast CPU inference)
- Unified evaluation report (quality + latency + memory) written as JSON for reproducible tables/plots

## Project Structure

```text
/
├── data/                              # Dataset directory (not committed)
│   └── Wednesday-workingHours.pcap_ISCX.csv
│
├── experiment_data/                   # Default output directory
│   ├── figures/                       # Training/evaluation figures
│   ├── analysis/                      # Feature analysis artifacts (optional)
│   └── runs/                          # Recommended: per-run directories (v2)
│       └── <RUN_ID>/                  # e.g., 20260102_165326_seed42_DoS_Hulk_w32-16_g5_k3
│           ├── dataset.pt
│           ├── trained_model.pt
│           ├── metrics.json           # baseline (float) metrics (optional)
│           └── lut/
│               └── <LUT_ID>/          # e.g., L64_sym_int8
│                   ├── layer*.npz
│                   ├── manifest.json
│                   └── lut_report_unified.json
│
├── figures/                           # KAN visualization files (optional)
│   ├── sp_0_*.png
│   ├── sp_1_*.png
│   └── sp_2_*.png
│
├── src/
│   ├── train.py                       # Baseline training pipeline
│   ├── analyze.py                     # Baseline performance analysis
│   ├── feature_analysis.py            # Baseline feature analysis
│   └── lut_v2/                        # LUT-KAN (v2) pipeline
│       ├── build_lut.py               # Build LUT artifacts for a trained model
│       ├── evaluate_lut.py            # Unified eval: quality + speed + memory
│       └── sweep_lut.py               # Optional: run multiple LUT configs
│
└── requirements.txt
```

Notes:
- Large artifacts (dataset CSV, trained models, LUT .npz files) should not be committed to git. Use per-run directories and/or GitHub Releases for reproducibility bundles.

## Installation

1) Clone the repository:
```bash
git clone https://github.com/KuznetsovKarazin/kan-dos-detection.git
cd kan-dos-detection
```

2) Create and activate a virtual environment:
```bash
python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows (PowerShell):
venv\Scripts\Activate.ps1
```

3) Install dependencies:
```bash
pip install -r requirements.txt
```

Optional (recommended for LUT speed):
```bash
pip install numba
```

## Dataset

This project uses the CICIDS2017 dataset (Wednesday traffic). Download it from the official source and place the CSV here:

```text
data/Wednesday-workingHours.pcap_ISCX.csv
```

Dataset reference: CIC (Canadian Institute for Cybersecurity) IDS 2017.

## Baseline Usage (PyKAN/PyTorch)

1) Train the KAN model:
```bash
python src/train.py
```

2) Analyze model performance:
```bash
python src/analyze.py
```

3) Analyze feature importance:
```bash
python src/feature_analysis.py
```

Outputs are written under `experiment_data/` by default. If you prefer per-run directories, create a run folder and adjust the save path (or wrap the scripts with a small runner).

## LUT-KAN (v2): Build and Evaluate LUT Inference

The LUT pipeline operates on a "run directory" that contains:
- `dataset.pt`
- `trained_model.pt`
Optionally:
- `metrics.json` (baseline float timing/metrics)

You can use either:
- `--run-dir experiment_data` (if your baseline scripts save there), or
- `--run-dir experiment_data/runs/<RUN_ID>` (recommended, cleaner for sweeps).

### Step A: Build LUT artifacts

Example: build L=64, symmetric int8, linear interpolation, closed boundary, clip_x OOB policy.

```bash
python -m src.lut_v2.build_lut   --run-dir experiment_data/runs/<RUN_ID>   --out experiment_data/runs/<RUN_ID>/lut/L64_sym_int8   --L 64   --value-repr spline_component   --scheme symmetric --dtype int8   --interp linear   --boundary-mode closed   --oob-policy clip_x   --calib-split train --num-samples 4096   --device cpu
```

This writes:
- per-layer LUT artifacts (`layer*.npz`)
- `manifest.json` describing the LUT configuration

### Step B: Evaluate LUT (unified report)

Run a single command to compute:
- quality: float vs LUT (NumPy and, if installed, Numba)
- latency: infer-only (comparable) + end-to-end (prepare+infer)
- memory: float model footprint vs LUT artifacts (total and per layer)

```bash
python -m src.lut_v2.evaluate_lut   --run-dir experiment_data/runs/<RUN_ID>   --lut-dir experiment_data/runs/<RUN_ID>/lut/L64_sym_int8   --device cpu   --threads-torch 1   --threads-numba 1   --batch-size 256   --warmup-iters 50   --measure-iters 200   --threshold 0.5
```

Output:
- `experiment_data/runs/<RUN_ID>/lut/L64_sym_int8/lut_report_unified.json`

Interpretation guide:
- `timing.infer_only.*` measures pure forward execution (best for fair comparisons)
- `timing.end2end.*` includes packing/adapter preparation inside the timed loop (useful for deployment budgeting; slower by design)

### Optional: LUT sweeps

If you want to run multiple LUT configurations (L, scheme/dtype, OOB/boundary modes), use `src/lut_v2/sweep_lut.py` or generate YAML/CLI lists and iterate. A common sweep:
- L ∈ {16, 32, 64, 128}
- scheme/dtype: symmetric int8 vs asymmetric uint8
- boundary_mode ∈ {closed, half_open}
- oob_policy ∈ {clip_x, zero_spline}

After the sweep, aggregate `lut_report_unified.json` files into tables (mean/std over seeds). A dedicated `collect_results.py` is recommended for paper-ready tables.

## Results (baseline)

Example (typical) performance metrics:
- Accuracy: 0.990
- Precision: 0.984
- Recall: 0.996
- F1-Score: 0.990

Resource summary (baseline float model):
- Total Parameters: ~50K
- Model size: ~0.19 MB (float)

Note: LUT-KAN changes the memory/latency trade-off: LUT artifacts are typically larger than float parameters, but can substantially reduce infer-only latency on CPU.

## Hardware / Experimental Setup (paper reference)

Example setup used for experiments:
- CPU: AMD Ryzen 7 7840HS (3.80 GHz)
- RAM: 64 GB
- OS: Windows

When reporting speed, specify:
- backend (float PyTorch vs LUT NumPy vs LUT Numba)
- whether timing is infer-only or end-to-end
- batch size, warmup/iters, and thread settings

## Citation

If you use this work in your research, please cite:

```bibtex
@article{Kuznetsov_2026,
  title   = {LUT-Compiled Kolmogorov-Arnold Networks for Lightweight DoS Detection on IoT Edge Devices},
  url     = {http://arxiv.org/abs/2601.08044},
  doi     = {10.48550/arXiv.2601.08044},
  note    = {arXiv:2601.08044 [cs]},
  author  = {Kuznetsov, Oleksandr},
  year    = {2026},
  month   = jan
}

```

## License

This project is licensed under the MIT License.

## Contact

- Oleksandr Kuznetsov - oleksandr.o.kuznetsov@gmail.com
- Project Link: https://github.com/KuznetsovKarazin/kan-dos-detection/tree/lut-v2

## Acknowledgments

- Canadian Institute for Cybersecurity for the CICIDS2017 dataset: https://www.unb.ca/cic/datasets/ids-2017.html
- KAN implementation based on pykan: https://github.com/KindXiaoming/pykan

