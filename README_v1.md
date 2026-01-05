# DoS Attack Detection using Kolmogorov-Arnold Networks (KAN)

## Overview

Implementation of an advanced Intrusion Detection System (IDS) for IoT using Kolmogorov-Arnold Networks. The system is designed to detect Denial of Service (DoS) attacks with high accuracy while maintaining minimal computational overhead.

## Key Features

- Lightweight architecture (50K parameters, 0.19 MB)
- High detection accuracy (99%)
- Fast inference (2.00ms per sample)
- Resource-efficient design suitable for IoT/edge devices

## Project Structure

```markdown
/
├── data/                      # Dataset directory
│   └── Wednesday-workingHours.pcap_ISCX.csv    # CICIDS2017 dataset
│
├── experiment_data/           # Experimental results
│   ├── analysis/             # Analysis results
│   │   ├── feature_analysis_report.md
│   │   ├── feature_correlations.png
│   │   ├── correlation_heatmap.png
│   │   └── feature_statistics.csv
│   ├── figures/              # Training visualizations
│   │   ├── attack_distribution.png
│   │   └── training_curves.png
│   └── model/               # Saved models and checkpoints
│
├── figures/                  # KAN Visualization Files
│   ├── sp_0__.png         # Layer 0 spline visualizations
│   ├── sp_1__.png         # Layer 1 spline visualizations
│   └── sp_2__.png         # Layer 2 spline visualizations
│       # Format: sp_X_Y_Z.png where:
│       # X - layer index
│       # Y - neuron index
│       # Z - spline index
│
├── src/                      # Source code
│   ├── train.py             # Training pipeline
│   ├── analyze.py           # Performance analysis
│   └── feature_analysis.py  # Feature importance analysis
│
└── requirements.txt          # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/KuznetsovKarazin/kan-dos-detection.git
cd kan-dos-detection
```
2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model:
```bash
python src/train.py
```
2. Analyze model performance:
```bash
python src/analyze.py
```
3. Analyze feature importance:
```bash
python src/feature_analysis.py
```

## Results

### Performance Metrics

- Accuracy: 0.990
- Precision: 0.984
- Recall: 0.996
- F1-Score: 0.990

### Resource Requirements

- Total Parameters: 50,092
- Trainable Parameters: 42,336
- Model Size: 0.19 MB
- Average Inference Time: 2.00 ms per sample

### KAN Visualizations
The `figures/` directory contains detailed visualizations of the KAN network's internal structure:
- Spline visualizations for each layer (sp_X_Y_Z.png)
  - Layer 0: Input processing splines
  - Layer 1: Hidden layer feature transformations
  - Layer 2: Output layer decision boundaries
- Each visualization shows how individual neurons process and transform input features
- File naming convention: sp_X_Y_Z.png
  - X: Layer index
  - Y: Neuron index in the layer
  - Z: Individual spline index for the neuron

## Data

The project uses the CICIDS2017 dataset, focusing on Wednesday's traffic which contains various DoS attack types:

- DoS Hulk (231,073 samples)
- DoS GoldenEye (10,293 samples)
- DoS slowloris (5,796 samples)
- DoS Slowhttptest (5,499 samples)
- Heartbleed (11 samples)

## Model Architecture

- Input Layer: 78 features
- Hidden Layers: [32, 16]
- Output Layer: Binary classification
- Architecture: Kolmogorov-Arnold Network
- Spline Degree: 3
- Grid Points: 5

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{Kuznetsov_2025, title={Efficient Denial of Service Attack Detection in IoT using Kolmogorov-Arnold Networks}, url={http://arxiv.org/abs/2502.01835}, DOI={10.48550/arXiv.2502.01835}, note={arXiv:2502.01835 [cs]}, number={arXiv:2502.01835}, publisher={arXiv}, author={Kuznetsov, Oleksandr}, year={2025}, month=feb }
```

## Contact

- Oleksandr Kuznetsov - oleksandr.o.kuznetsov@gmail.com
- Project Link: https://github.com/KuznetsovKarazin/kan-dos-detection

## Acknowledgments

- Canadian Institute for Cybersecurity for the CICIDS2017 dataset (https://www.unb.ca/cic/datasets/ids-2017.html)
- KAN implementation based on pykan (https://github.com/KindXiaoming/pykan)