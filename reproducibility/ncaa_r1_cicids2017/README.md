# NCA R1 CICIDS2017 DoS Hulk Reproducibility Record

This directory records the dataset provenance, leakage-free preprocessing,
training configuration, and held-out evaluation results used for the
CICIDS2017 DoS Hulk case study in the Neural Computing and Applications
R1 revision.

## Dataset

File:

`Wednesday-workingHours.pcap_ISCX.csv`

SHA-256:

`893c27dc968bf7a8adef1689f90be55ca4a4dc3088fb63d6ff247ac56856df2a`

The raw CSV contains 692,703 flows, 78 numerical input features, and one
class-label column.

Raw class counts are recorded in `preprocessing_manifest.json`.

## Binary cohort

The experiment uses all available DoS Hulk flows and an equally sized
random sample of BENIGN flows:

- BENIGN: 231,073
- DoS Hulk: 231,073
- total: 462,146

A stratified 80/20 split with random seed 42 gives:

- training set: 369,716 flows
- held-out test set: 92,430 flows
- test BENIGN: 46,215
- test DoS Hulk: 46,215

The training and test source-row indices are disjoint.

## Leakage-free preprocessing

The train/test split is performed before fitting any feature transformation.

The following quantities are estimated exclusively from the training
partition:

1. median values used for missing-value imputation;
2. Q1/Q3 statistics and 3xIQR clipping bounds;
3. StandardScaler parameters.

The frozen training statistics are subsequently applied to the held-out
test partition.

Outliers are clipped (winsorized); rows are not removed.

Detailed train-fitted preprocessing statistics are provided in
`feature_stats.csv`.

## KAN configuration

- architecture: 78 -> 32 -> 16 -> 1
- grid: 5
- spline order k: 3
- optimizer: Adam
- learning rate: 0.001
- epochs: 200
- random seed: 42
- classification threshold: 0.5

Training follows a fixed 200-epoch schedule. The held-out test partition is
not inspected during training.

## Software provenance

The model was trained from repository commit:

`fb4fe662dd40867c9f9e4f841ed4f70ddbb706ef`

PyKAN:

- version: 0.2.8
- Git commit: `ecde4ec3274d3bef1ad737479cf126aed38ab530`

Exact Python, PyTorch, NumPy, pandas, scikit-learn, platform, dataset, and
Git provenance is recorded in `run_meta.json`.

## Held-out test results

The fixed classification threshold is 0.5.

| Metric | Value |
|---|---:|
| Accuracy | 0.98995997 |
| Precision | 0.98438402 |
| Recall | 0.99571568 |
| F1 | 0.99001743 |
| ROC-AUC | 0.99910626 |
| PR-AUC | 0.99905396 |

Confusion matrix:

- TN = 45,485
- FP = 730
- FN = 198
- TP = 46,017

Machine-readable results are stored in `metrics.json`.

The corresponding confusion matrix and ROC/PR curves are stored under
`figures/`.

## Archived trained model

The trained `trained_model.pt` used for the R1 LUT evaluation has SHA-256:

`7E81C8BF6CF5861821CAAEDD27A3534FB8F3B738A98DD4BE15E8B2E930B790C3`

The model itself, the CICIDS2017 CSV, prepared tensors, preprocessing
objects, and other large experiment artifacts are intentionally not tracked
in Git.

## Reproduction

Prepare the leakage-free split and preprocessing artifacts:

    python -m src.train `
      --data "data\Wednesday-workingHours.pcap_ISCX.csv" `
      --attack-type "DoS Hulk" `
      --max-samples-per-class 231073 `
      --test-size 0.2 `
      --seed 42 `
      --epochs 200 `
      --run-dir "experiment_data\runs\NCA_R1_CICIDS2017_DoS_Hulk_seed42" `
      --prepare-only

Train using the exact prepared split:

    python -m src.train `
      --data "data\Wednesday-workingHours.pcap_ISCX.csv" `
      --attack-type "DoS Hulk" `
      --max-samples-per-class 231073 `
      --test-size 0.2 `
      --seed 42 `
      --epochs 200 `
      --run-dir "experiment_data\runs\NCA_R1_CICIDS2017_DoS_Hulk_seed42" `
      --reuse-prepared

Recompute the final held-out evaluation:

    python -m src.analyze `
      --load-dir "experiment_data\runs\NCA_R1_CICIDS2017_DoS_Hulk_seed42" `
      --threshold 0.5

## LUT evaluation

The revised LUT compilation, matched NumPy/Numba B-spline baselines,
endpoint-inclusive LUT construction, and latency/memory experiments are
performed using the canonical `KuznetsovKarazin/lut-kan` repository.

The historical LUT implementation retained in this repository is not used
to generate the revised NCA R1 LUT results.
