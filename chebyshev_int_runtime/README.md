# Integer-Only Inference Extension (Chebyshev KAN)

This directory contains a reference implementation of **integer-only LUT inference** for Kolmogorov-Arnold Networks on microcontrollers without a hardware floating-point unit (FPU). It is a companion contribution to the main project [`kan-dos-detection`](../) by O. Kuznetsov, published in *Neurocomputing* (Elsevier, 2026), and addresses one specific aspect that is left open in the main LUT compilation pipeline: the residual floating-point multiplication during run-time dequantization.

In the main pipeline, the LUT is stored as `int8` quantized values and a `float32` per-segment scale factor; at inference time, the dequantization step is `v = scale[e,seg] × q_table[e,seg,⌊λ⌋]`, an `int8 × float32` multiplication. On boards with a hardware FPU (e.g. ESP32-C3 RISC-V) this is cheap; on boards without an FPU (e.g. AVR-class Arduino Mega 2560) it is software-emulated and dominates the inference cost.

The technique implemented here removes that residual floating-point operation by **pre-scaling the LUT entries to `int16` at export time**, so that the dequantization is absorbed into the table itself and the entire runtime — lookup, accumulation, decision — is performed in pure integer arithmetic.

---

## What this contribution adds

A self-contained reference pipeline:

- **Training**: a Chebyshev-basis KAN (single-layer and two-layer with tabulated `tanh`), trained on the TON_IoT dataset for both binary and 10-class intrusion detection.
- **Export**: scripts (`export_lut_int.py` and variants) that compile the trained model to pre-scaled `int16` LUTs and emit C headers ready to flash.
- **Embedded runtime**: minimal C/C++ firmware for Arduino Mega 2560 and ESP32-C3 that performs end-to-end integer inference (Q16.16 inputs, `int16` LUT entries, `int32` accumulation, integer-comparison decision).
- **Wokwi validation**: simulated execution on both target boards with latency and accuracy measurements.

The implementation is **architecturally distinct** from the main project: it uses Chebyshev polynomials (vs. B-spline) on a unified 10-feature subset of TON_IoT (vs. 78/91 feature CICIDS2017/TON_IoT). It is therefore not a drop-in replacement for the deep B-spline models of `lut-v2`, but a parallel demonstration that the same integer-only principle can be applied across KAN variants.

---

## Directory layout

```
chebyshev_int_runtime/
├── src/                              # Python models (NumPy / PyTorch)
│   ├── kan_chebyshev.py              # binary single-layer KAN
│   ├── kan_chebyshev_multiclass.py   # 10-class single-layer KAN
│   ├── kan_multilayer_numpy.py       # multi-layer KAN, NumPy forward
│   ├── kan_torch.py                  # PyTorch training (multi-layer)
│   └── kan_bspline.py                # B-spline variant (comparison)
│
├── scripts/                          # Export to integer LUT and C headers
│   ├── export_lut_int.py             # binary, single-layer
│   ├── export_lut_int_multiclass.py  # 10-class, single-layer
│   ├── export_ml_int.py              # multi-layer with tabulated tanh
│   ├── ml_stage2_layer1_tanh.py      # multi-layer staged quantization
│   └── ml_stage3_full.py             # multi-layer full forward export
│
├── mcu/                              # C/C++ firmware and generated headers
│   ├── main_kan_wokwi_fullint.cpp    # binary, integer-only firmware
│   ├── main_kan_wokwi_int.cpp        # binary, mixed-int (for comparison)
│   ├── main_kan_wokwi.cpp            # binary, float (baseline)
│   ├── main_kan_mc_wokwi.cpp         # multiclass single-layer firmware
│   ├── main_kan_ml_wokwi.cpp         # multi-layer firmware (forward only)
│   ├── kan_ids_layer.h               # generated LUT (binary, float)
│   ├── kan_ids_layer_int.h           # generated LUT (binary, int16)
│   ├── kan_ids_mc_int.h              # generated LUT (multiclass)
│   ├── kan_ml_layer1.h               # generated LUT (multi-layer L1)
│   ├── kan_ml_layer2.h               # generated LUT (multi-layer L2)
│   ├── kan_ml_tanh.h                 # generated LUT (tabulated tanh)
│   ├── test_vectors*.h               # sanity-check test vectors
│   └── WOKWI_GUIDE*.md               # step-by-step Wokwi simulation guide
│
└── results/
    └── wokwi_measurements.md         # measured latencies and accuracies
```

---

## Method: integer-only pipeline

At export time, for each Chebyshev edge function `φ_e(x)` and each LUT segment `seg`:

```
LUT_int[e, seg] = round( S × dequant( q_table[e, seg] ) )
```

where `S` is a global integer scale chosen so that the LUT values fit `int16`. The per-segment `float32` scale factor is folded into the stored values; no separate scale array is shipped to the device.

At run time, given an input `x` already in Q16.16 fixed-point:

1. Locate the segment and the normalized position `λ ∈ [0, 1]` using integer arithmetic.
2. Read the two adjacent LUT entries and linearly interpolate, producing an `int32` contribution.
3. Accumulate per-output `int32` partial sums across all input edges.
4. Decide by integer comparison (sign of the accumulator for binary; argmax for multiclass).

No floating-point operation is performed on the device.

---

## Measured results

All measurements are performed in **Wokwi simulation** on the two target boards (Arduino Mega 2560, AVR 16 MHz, 8 KB SRAM; ESP32-C3, RISC-V 160 MHz, 320 KB SRAM). Each configuration is verified end-to-end against the Python reference (logit-level match on all test vectors, integer argmax identical to the float argmax in ≥ 99.7% of cases).

| Configuration              | macro-F1 | Edges | LUT size | Latency (ESP32-C3) | On-device accuracy* |
|----------------------------|----------|-------|----------|--------------------|---------------------|
| Binary, single-layer       | 0.969    | 10    | 10 KB    | 38 µs              | 97.5 %              |
| Multiclass, single-layer   | 0.858    | 100   | 100 KB   | 118 µs             | 90 %                |
| Multiclass, multi-layer    | 0.916    | 320   | 320 KB   | 691 µs             | 95 %                |

*On-device accuracy measured on the 40-vector sanity-check set used for Wokwi validation, not on the full test set. The full-test macro-F1 figures in the first column come from the Python reference forward, which has been verified to match the C runtime bit-for-bit on the same vectors.*

On the Arduino Mega 2560 (binary, single-layer), the same firmware achieves **357 µs** (vs. **2851 µs** for the float baseline), giving an end-to-end speedup of **8×**. On ESP32-C3 the speedup is **44×**. The Mega/ESP32-C3 latency ratio, originally 1.7× under the float baseline (where both boards spent most of their time emulating floating-point in software), grows to 9.4× under integer-only inference — closely matching the underlying clock ratio (160/16 MHz = 10×). This shows that under integer-only inference the measured latency reflects the physical capability of each platform, rather than the overhead of software-emulated floating-point.

See [`results/wokwi_measurements.md`](results/wokwi_measurements.md) for the complete per-board breakdown and the full test-set macro-F1 of the multi-layer model (0.9118 on 12k test samples; 0.9177 on all 211k samples of the TON_IoT dataset).

---

## How to apply this principle to the main `lut-v2` B-spline models

The integer-only pre-scaling principle is **basis-agnostic** and architecture-agnostic: it depends only on the structure of the LUT (per-edge, per-segment quantized values plus a scale factor). Adapting `src/lut_v2/build_lut.py` to emit pre-scaled `int16` tables in place of the current `int8 + float32 scale` representation would be a direct port of the technique implemented here in `scripts/export_lut_int.py`. The C runtime in `ids_hw/ids_mega_lut_kan/main.cpp` would similarly drop the floating-point multiplication step from its inner loop.

On the Mega 2560, where the dequantization is the dominant cost, this is expected to yield a substantial reduction in inference latency. The integration is left as a follow-up: this contribution provides the reference implementation and validates the principle on a smaller model where the speedup can be measured cleanly.

---

## Scope and limitations

- **Simulation, not physical hardware.** All measurements reported here are obtained in Wokwi simulation. Physical-board validation is the natural next step.
- **Different model family.** The Chebyshev single/multi-layer architectures used here are simpler than the deep B-spline models of `lut-v2`. The absolute accuracies (0.969 binary, 0.92 multiclass) are below those of the deep B-spline models on the same task; the goal here is not to compete on accuracy but to demonstrate that an integer-only runtime is feasible and to quantify the speedup it provides.
- **Different dataset framing.** TON_IoT is used as a 10-class problem on a 10-feature unified subspace (selected by mutual information). The main project uses the binary CICIDS2017 task on the full 78-feature space and TON_IoT on 91 features. Direct numerical comparison between the two pipelines is therefore not meaningful.

---

## Reproducing the results

The full reproducibility pipeline (training, LUT export, header generation, Wokwi simulation) is documented in the companion repository [`IDS-KAN`](https://github.com/emanuelepiodebernardis/IDS-KAN), which contains the dataset preprocessing, training scripts, and configuration files used here. The files in this directory are extracted from that companion repository at the state described in `results/wokwi_measurements.md`.

---

## License

MIT, consistent with the parent project.
