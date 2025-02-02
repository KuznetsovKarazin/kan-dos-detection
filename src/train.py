"""
DoS Detection using KAN - Enhanced Training Module
This module implements advanced data preparation and model training for DoS attack detection
using Kolmogorov-Arnold Networks (KAN).

Author: Oleksandr Kuznetsov
Date: February 2025
"""

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from kan import KAN
import pickle
import os
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

def prepare_dos_data(filepath, attack_type='DoS Hulk', max_samples_per_class=50000):
    """
    Prepare balanced dataset for DoS attack detection with enhanced data processing.
    """
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    print("\nUnique labels in dataset:")
    attack_stats = df[' Label'].value_counts()
    print(attack_stats)
    
    plt.figure(figsize=(10, 6))
    attack_stats.plot(kind='bar')
    plt.title('Distribution of Attack Types')
    plt.xticks(rotation=45)
    plt.tight_layout()
    Path('experiment_data/figures').mkdir(parents=True, exist_ok=True)
    plt.savefig('experiment_data/figures/attack_distribution.png')
    plt.close()
    
    # Balance classes with maximum available samples
    max_samples = min(
        max_samples_per_class,
        df[df[' Label'] == 'BENIGN'].shape[0],
        df[df[' Label'] == attack_type].shape[0]
    )
    
    print(f"\nUsing {max_samples} samples per class")
    
    benign_samples = df[df[' Label'] == 'BENIGN'].sample(n=max_samples, random_state=42)
    attack_samples = df[df[' Label'] == attack_type].sample(n=max_samples, random_state=42)
    
    df = pd.concat([benign_samples, attack_samples])
    
    df['attack'] = (df[' Label'] != 'BENIGN').astype(int)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'attack']
    
    # Enhanced data cleaning
    df = df.replace([np.inf, -np.inf], np.nan)
    
    feature_stats = []
    for col in numeric_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median)
        
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        df[col] = df[col].clip(lower_bound, upper_bound)
        
        feature_stats.append({
            'feature': col,
            'median': median,
            'q1': q1,
            'q3': q3,
            'outliers': outliers
        })
    
    # Save feature statistics
    pd.DataFrame(feature_stats).to_csv('experiment_data/feature_stats.csv', index=False)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(df[numeric_cols])
    y = df['attack'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    dataset = {
        'train_input': torch.FloatTensor(X_train),
        'train_label': torch.FloatTensor(y_train).reshape(-1, 1),
        'test_input': torch.FloatTensor(X_test),
        'test_label': torch.FloatTensor(y_test).reshape(-1, 1)
    }
    
    print(f"\nDataset shapes:")
    print(f"Train: {X_train.shape}")
    print(f"Test: {X_test.shape}")
    print(f"\nClass distribution in train:")
    print(pd.Series(y_train).value_counts(normalize=True))
    
    return dataset, scaler, numeric_cols

def train_kan_model(dataset, input_dim, epochs=100):
    """
    Enhanced KAN model training with advanced architecture and monitoring.
    """
    # Create model with enhanced architecture
    model = KAN(
        #width=[input_dim, 64, 32, 16, 1],
        width=[input_dim, 32, 16, 1],
        #grid=7,
        grid=5,
        k=3,
        seed=42
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epochs': []
    }
    
    print("\nModel architecture:")
    print(f"Input dimension: {input_dim}")
    print(f"Hidden layers: {model.width[1:-1]}")
    print(f"Grid points: {model.grid}")
    print(f"Spline degree: {model.k}")
    
    print("\nStarting training...")
    
    best_test_acc = 0
    patience = 10
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        outputs = model(dataset['train_input'])
        loss = criterion(outputs, dataset['train_label'])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            train_pred = (torch.sigmoid(outputs) > 0.5).float()
            train_acc = (train_pred == dataset['train_label']).float().mean()
            
            test_outputs = model(dataset['test_input'])
            test_loss = criterion(test_outputs, dataset['test_label'])
            test_pred = (torch.sigmoid(test_outputs) > 0.5).float()
            test_acc = (test_pred == dataset['test_label']).float().mean()
        
        history['train_loss'].append(loss.item())
        history['train_acc'].append(train_acc.item())
        history['test_loss'].append(test_loss.item())
        history['test_acc'].append(test_acc.item())
        history['epochs'].append(epoch + 1)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            no_improve = 0
        else:
            no_improve += 1
        
        #if no_improve >= patience:
        #    print(f"\nEarly stopping at epoch {epoch+1}")
        #    break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'Train Loss: {loss.item():.4f}, Train Acc: {train_acc.item():.4f}')
            print(f'Test Loss: {test_loss.item():.4f}, Test Acc: {test_acc.item():.4f}')
    
    return model, history

def plot_training_curves(history, save_dir='experiment_data/figures'):
    """
    Plot and save enhanced training curves.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['epochs'], history['train_loss'], label='Train Loss')
    plt.plot(history['epochs'], history['test_loss'], label='Test Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['epochs'], history['train_acc'], label='Train Accuracy')
    plt.plot(history['epochs'], history['test_acc'], label='Test Accuracy')
    plt.title('Accuracy During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_experiment(dataset, scaler, features, model, history, save_dir='experiment_data'):
    """
    Enhanced experiment saving with metadata.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save dataset
    torch.save(dataset, save_dir / 'dataset.pt')
    
    # Save scaler
    with open(save_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save features list
    with open(save_dir / 'features.pkl', 'wb') as f:
        pickle.dump(features, f)
    
    # Save model and training history
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'architecture': {
            'width': model.width,
            'grid': model.grid,
            'k': model.k
        },
        'timestamp': datetime.now().isoformat()
    }, save_dir / 'trained_model.pt')
    
    # Plot training curves
    plot_training_curves(history)
    
    print(f"Experiment saved to {save_dir}")

if __name__ == "__main__":
    data_path = Path('data/Wednesday-workingHours.pcap_ISCX.csv')
    
    dataset, scaler, features = prepare_dos_data(
        data_path, 
        attack_type='DoS Hulk',
        max_samples_per_class=231073
    )
    
    model, history = train_kan_model(
        dataset,
        input_dim=len(features),
        epochs=200
    )
    
    save_experiment(dataset, scaler, features, model, history)