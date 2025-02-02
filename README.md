# DoS Attack Detection using KAN

## Project Description
This project implements a DoS attack detection system using Kolmogorov-Arnold Networks (KAN). The system is designed to detect DoS Hulk attacks in network traffic with high accuracy while maintaining low computational overhead.

## Results
- Accuracy: 99%
- Model Size: 0.19 MB
- Inference Time: 2.00 ms per sample

## Project Structure
- `data/`: Contains the CICIDS2017 dataset
- `experiment_data/`: Contains trained models and results
- `src/`: Contains source code
  - `train.py`: Training code
  - `analyze.py`: Analysis code

## Setup and Installation
1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Run training: `python src/train.py`
4. Run analysis: `python src/analyze.py`

## License
MIT