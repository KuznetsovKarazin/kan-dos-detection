# Feature Analysis Report for DoS Detection

## Overview
- Total features analyzed: 79
- Dataset size: 369,716 samples
- Positive class (attacks) ratio: 50.00%

## Top 10 Most Important Features
|                        |   correlation |        std |          mean |        min |       max |   zero_fraction |
|:-----------------------|--------------:|-----------:|--------------:|-----------:|----------:|----------------:|
| target                 |      1        | nan        | nan           | nan        | nan       |             nan |
| Avg Bwd Segment Size   |      0.631018 |   1.00013  |   0.000259173 |  -0.81848  |   3.55328 |               0 |
| Bwd Packet Length Mean |      0.631018 |   1.00013  |   0.000259173 |  -0.81848  |   3.55328 |               0 |
| Bwd Packet Length Std  |      0.617967 |   0.999839 |  -0.000178437 |  -0.724286 |   4.76045 |               0 |
| Bwd Packet Length Max  |      0.617608 |   0.999627 |  -0.000232059 |  -0.759416 |   4.21958 |               0 |
| Fwd IAT Std            |      0.615174 |   0.999721 |  -0.000293528 |  -0.695335 |   3.94457 |               0 |
| Packet Length Std      |      0.613591 |   0.999632 |  -0.000200197 |  -0.788665 |   4.23826 |               0 |
| Idle Max               |      0.609978 |   0.99971  |  -0.000227065 |  -0.710255 |   2.08499 |               0 |
| Idle Mean              |      0.609504 |   0.999818 |  -0.000111613 |  -0.707884 |   2.10977 |               0 |
| Fwd IAT Max            |      0.606884 |   0.999728 |  -0.00024808  |  -0.715666 |   2.08414 |               0 |

## Feature Importance Statistics
- Features with strong correlation (|r| > 0.5): 20
- Features with moderate correlation (0.3 < |r| < 0.5): 10
- Features with weak correlation (|r| < 0.3): 29

## Key Findings
1. Most predictive features:
  - target (r = 1.000)
  -  Avg Bwd Segment Size (r = 0.631)
  -  Bwd Packet Length Mean (r = 0.631)

2. Feature ranges:
  - Maximum value overall: 6.22
  - Minimum value overall: -4.50
  - Most variable feature:  Subflow Bwd Bytes (std = 1.00)

3. Zero values analysis:
  - Features with >50% zeros: 20
  - Most sparse feature:  Active Std (100.0% zeros)

## Recommendations
1. Primary features for simplified model:
  target,  Avg Bwd Segment Size,  Bwd Packet Length Mean,  Bwd Packet Length Std, Bwd Packet Length Max

2. Features to consider removing:
  - Low correlation (|r| < 0.1):  Total Fwd Packets, Subflow Fwd Packets,  Total Backward Packets
  - High sparsity (>90% zeros):  Active Std,  Bwd Avg Bytes/Bulk,  Bwd Avg Packets/Bulk

3. Feature engineering opportunities:
  - Consider combining correlated features
  - Investigate non-linear transformations for weak correlations
  - Create interaction terms for top features

## Visualization Details
- Feature correlation plot: feature_correlations.png
- Correlation heatmap: correlation_heatmap.png
- Detailed statistics: feature_statistics.csv

Analysis timestamp: 2025-02-02 23:22:44
