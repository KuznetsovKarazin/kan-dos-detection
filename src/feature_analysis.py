"""
DoS Detection using KAN - Feature Analysis Module
Analyzes feature importance and correlations for the trained model.

Author: Oleksandr Kuznetsov
Date: February 2025
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle

def load_experiment(load_dir='experiment_data'):
   """Load saved experiment data"""
   load_dir = Path(load_dir)
   
   dataset = torch.load(load_dir / 'dataset.pt')
   model_data = torch.load(load_dir / 'trained_model.pt')
   
   with open(load_dir / 'features.pkl', 'rb') as f:
       features = pickle.load(f)
   
   return dataset, model_data, features

def analyze_features(dataset, features, save_dir='experiment_data/analysis'):
   """Analyze feature importance and correlations"""
   save_dir = Path(save_dir)
   save_dir.mkdir(parents=True, exist_ok=True)
   
   # Get feature values from dataset
   X = dataset['train_input'].numpy()
   y = dataset['train_label'].numpy()
   
   # Create DataFrame
   df = pd.DataFrame(X, columns=features)
   df['target'] = y
   
   # 1. Feature correlations with target
   correlations = df.corr()['target'].sort_values(ascending=False)
   
   plt.figure(figsize=(12, 6))
   correlations[:-1].plot(kind='bar')
   plt.title('Feature Correlations with Target (DoS Attack Detection)')
   plt.xlabel('Features')
   plt.ylabel('Correlation Coefficient')
   plt.xticks(rotation=45, ha='right')
   plt.tight_layout()
   plt.savefig(save_dir / 'feature_correlations.png', dpi=300, bbox_inches='tight')
   plt.close()
   
   # 2. Correlation heatmap of top 15 features
   plt.figure(figsize=(12, 10))
   top_features = correlations[1:16].index  # Exclude target
   sns.heatmap(df[top_features].corr(), annot=True, fmt='.2f', cmap='RdBu_r')
   plt.title('Feature Correlation Heatmap (Top 15 Features)')
   plt.tight_layout()
   plt.savefig(save_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
   plt.close()
   
   # 3. Feature statistics
   stats = pd.DataFrame({
       'correlation': correlations[:-1],
       'std': df[features].std(),
       'mean': df[features].mean(),
       'min': df[features].min(),
       'max': df[features].max(),
       'zero_fraction': (df[features] == 0).mean()
   }).sort_values('correlation', ascending=False)
   
   # Save detailed statistics
   stats.to_csv(save_dir / 'feature_statistics.csv')
   
   return stats, df

def generate_feature_report(stats, df, save_dir='experiment_data/analysis'):
   """Generate comprehensive feature analysis report"""
   # Преобразуем строку в Path объект
   save_dir = Path(save_dir)
   save_dir.mkdir(parents=True, exist_ok=True)
   
   report = f"""# Feature Analysis Report for DoS Detection

## Overview
- Total features analyzed: {len(stats)}
- Dataset size: {len(df):,} samples
- Positive class (attacks) ratio: {df['target'].mean():.2%}

## Top 10 Most Important Features
{stats.head(10).to_markdown()}

## Feature Importance Statistics
- Features with strong correlation (|r| > 0.5): {sum(abs(stats['correlation']) > 0.5)}
- Features with moderate correlation (0.3 < |r| < 0.5): {sum((abs(stats['correlation']) > 0.3) & (abs(stats['correlation']) <= 0.5))}
- Features with weak correlation (|r| < 0.3): {sum(abs(stats['correlation']) <= 0.3)}

## Key Findings
1. Most predictive features:
  - {stats.index[0]} (r = {stats['correlation'].iloc[0]:.3f})
  - {stats.index[1]} (r = {stats['correlation'].iloc[1]:.3f})
  - {stats.index[2]} (r = {stats['correlation'].iloc[2]:.3f})

2. Feature ranges:
  - Maximum value overall: {stats['max'].max():.2f}
  - Minimum value overall: {stats['min'].min():.2f}
  - Most variable feature: {stats['std'].idxmax()} (std = {stats['std'].max():.2f})

3. Zero values analysis:
  - Features with >50% zeros: {sum(stats['zero_fraction'] > 0.5)}
  - Most sparse feature: {stats['zero_fraction'].idxmax()} ({stats['zero_fraction'].max():.1%} zeros)

## Recommendations
1. Primary features for simplified model:
  {', '.join(stats.head(5).index)}

2. Features to consider removing:
  - Low correlation (|r| < 0.1): {', '.join(stats[abs(stats['correlation']) < 0.1].head(3).index)}
  - High sparsity (>90% zeros): {', '.join(stats[stats['zero_fraction'] > 0.9].head(3).index)}

3. Feature engineering opportunities:
  - Consider combining correlated features
  - Investigate non-linear transformations for weak correlations
  - Create interaction terms for top features

## Visualization Details
- Feature correlation plot: feature_correlations.png
- Correlation heatmap: correlation_heatmap.png
- Detailed statistics: feature_statistics.csv

Analysis timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
   
   with open(save_dir / 'feature_analysis_report.md', 'w') as f:
       f.write(report)

if __name__ == "__main__":
   # Load data
   dataset, model_data, features = load_experiment()
   
   # Analyze features
   analysis_dir = Path('experiment_data/analysis')
   stats, df = analyze_features(dataset, features, analysis_dir)
   
   # Generate report
   generate_feature_report(stats, df, analysis_dir)
   
   print(f"\nAnalysis completed. Results saved to:")
   print(f"- Report: {analysis_dir / 'feature_analysis_report.md'}")
   print(f"- Feature correlations: {analysis_dir / 'feature_correlations.png'}")
   print(f"- Correlation heatmap: {analysis_dir / 'correlation_heatmap.png'}")
   print(f"- Statistics: {analysis_dir / 'feature_statistics.csv'}")