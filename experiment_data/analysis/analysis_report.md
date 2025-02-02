# DoS Detection Model Analysis Report
    
## Model Architecture
- Input Features: [78, 0]
- Hidden Layers: [[32, 0], [16, 0]]
- Output Layer: [1, 0]
- Grid Points: 5
- Spline Degree: 3

## Performance Metrics
- Accuracy: 0.990
- Precision: 0.984
- Recall: 0.996
- F1-Score: 0.990
- AUC-ROC: 0.977

## Model Characteristics
- Total Parameters: 50,092
- Trainable Parameters: 42,336
- Model Size: 0.19 MB
- Average Inference Time: 2.00 ms per sample

## Optimal Decision Thresholds
- Best F1-Score Threshold: 0.737
  - F1-Score: 0.991
- Best Accuracy Threshold: 0.737
  - Accuracy: 0.991

## Key Findings
1. Compact Architecture:
   - Uses [78, 0] input features
   - 2 hidden layers with [[32, 0], [16, 0]] neurons
   - Single output neuron for binary classification

2. Performance:
   - High accuracy (90%) with balanced precision/recall
   - Excellent AUC-ROC (0.977)
   - Fast inference (1.26ms per sample)

3. Resource Efficiency:
   - Small model size (0.19 MB)
   - Efficient parameter usage (42,336 trainable parameters)
   - Suitable for edge deployment

## Visualizations Generated
- Model architecture: model_structure.png
- ROC and PR curves: roc_pr_curves.png
- Training metrics: training_curves.png
- Confusion matrix: confusion_matrix.png
- Threshold analysis: threshold_analysis.png

## Analysis Timestamp
2025-02-02 21:30:55
