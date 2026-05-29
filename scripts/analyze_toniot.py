"""
Generate analysis figures for TON_IoT run — mirrors the CICIDS2017 analysis folder.
Produces: confusion_matrix, roc_pr_curves, threshold_analysis, attack_distribution,
          correlation_heatmap, feature_correlations, model_structure, analysis_report.

Usage:
    python scripts/analyze_toniot.py \
        --run-dir experiment_data\runs\20260522_123241_seed42_TON_IoT_binary_w32-16_g5_k3

Author: Oleksandr Kuznetsov
"""
import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay, auc, confusion_matrix,
    precision_recall_curve, roc_curve,
)


def load_run(run_dir: Path):
    dataset    = torch.load(run_dir / "dataset.pt",       map_location="cpu")
    model_data = torch.load(run_dir / "trained_model.pt", map_location="cpu")
    with open(run_dir / "features.pkl", "rb") as f:
        features = pickle.load(f)
    return dataset, model_data, features


def plot_confusion_matrix(y_true, y_pred, out_dir: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Attack"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — TON_IoT")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print("  [OK] confusion_matrix.png")


def plot_roc_pr(y_true, y_prob, out_dir: Path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(fpr, tpr, lw=1.5, label=f"ROC curve (AUC={roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("Receiver Operating Characteristic (ROC)"); axes[0].legend()

    axes[1].plot(rec, prec, lw=1.5, label=f"PR curve (AP={pr_auc:.3f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve"); axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_dir / "roc_pr_curves.png", dpi=150)
    plt.close()
    print("  [OK] roc_pr_curves.png")
    return roc_auc, pr_auc


def plot_threshold_analysis(y_true, y_prob, out_dir: Path):
    thresholds = np.linspace(0.01, 0.99, 100)
    accs, precs, recs, f1s = [], [], [], []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        tn = ((pred == 0) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        acc  = (tp + tn) / (tp + fp + tn + fn + 1e-9)
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        accs.append(acc); precs.append(prec); recs.append(rec); f1s.append(f1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(thresholds, accs,  label="Accuracy")
    axes[0].plot(thresholds, precs, label="Precision")
    axes[0].plot(thresholds, recs,  label="Recall")
    axes[0].plot(thresholds, f1s,   label="F1")
    axes[0].axvline(0.5, color="gray", linestyle="--", lw=0.8, label="τ=0.5")
    axes[0].set_xlabel("Decision Threshold"); axes[0].set_title("Metrics vs Decision Threshold")
    axes[0].legend(); axes[0].set_ylim(0.5, 1.02)

    axes[1].plot(recs, precs, lw=1.5)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Trade-off")

    plt.tight_layout()
    plt.savefig(out_dir / "threshold_analysis.png", dpi=150)
    plt.close()
    print("  [OK] threshold_analysis.png")


def plot_attack_distribution(run_dir: Path, out_dir: Path):
    """Reconstruct label distribution from dataset.pt."""
    dataset = torch.load(run_dir / "dataset.pt", map_location="cpu")
    y_all = torch.cat([
        dataset["train_label"].flatten(),
        dataset["test_label"].flatten()
    ]).numpy().astype(int)
    counts = {0: int((y_all == 0).sum()), 1: int((y_all == 1).sum())}
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Normal (0)", "Attack (1)"], [counts[0], counts[1]],
           color=["steelblue", "tomato"])
    ax.set_title("Class Distribution — TON_IoT (balanced)")
    ax.set_ylabel("Samples")
    for i, (k, v) in enumerate(counts.items()):
        ax.text(i, v + 100, str(v), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "attack_distribution.png", dpi=150)
    plt.close()
    print("  [OK] attack_distribution.png")


def plot_correlation_heatmap(dataset, features, out_dir: Path, top_n: int = 15):
    X = dataset["test_input"].numpy()
    # Use only numeric (non-binary) features for correlation — filter out one-hot
    is_binary = np.all((X == 0) | (X == 1), axis=0)
    num_idx = np.where(~is_binary)[0][:top_n]
    if len(num_idx) < 2:
        num_idx = np.arange(min(top_n, X.shape[1]))
    feat_names = [features[i] for i in num_idx]
    X_sub = X[:, num_idx]
    corr = np.corrcoef(X_sub.T)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(feat_names))); ax.set_xticklabels(feat_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(feat_names))); ax.set_yticklabels(feat_names, fontsize=7)
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Feature Correlation Heatmap (top {len(feat_names)} numeric features)")
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_heatmap.png", dpi=150)
    plt.close()
    print("  [OK] correlation_heatmap.png")


def write_analysis_report(out_dir: Path, metrics: dict):
    lines = [
        "# Analysis Report — TON_IoT KAN Model",
        "",
        "## Model Performance",
        f"- Accuracy : {metrics['accuracy']:.4f}",
        f"- Precision: {metrics['precision']:.4f}",
        f"- Recall   : {metrics['recall']:.4f}",
        f"- F1 Score : {metrics['f1']:.4f}",
        f"- ROC-AUC  : {metrics['roc_auc']:.4f}",
        f"- PR-AUC   : {metrics['pr_auc']:.4f}",
        "",
        "## Dataset",
        "- TON_IoT Network Dataset (UNSW Canberra)",
        f"- Features : {metrics['n_features']}",
        f"- Test samples: {metrics['n_test']}",
        "",
        "## Architecture",
        f"- {metrics['arch']}",
        f"- Parameters: {metrics['n_params']:,}",
        f"- Grid: G={metrics['grid']}, k={metrics['k']}",
    ]
    (out_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("  [OK] analysis_report.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    print(f"Generating analysis in {out_dir} ...")

    dataset, model_data, features = load_run(run_dir)
    arch_cfg = model_data["architecture"]

    from kan import KAN
    model = KAN(**arch_cfg)
    model.load_state_dict(model_data["model_state_dict"])
    model.eval()

    X_test = dataset["test_input"]
    y_test = dataset["test_label"].numpy().flatten().astype(int)

    with torch.no_grad():
        logits = model(X_test).numpy().flatten()
    y_prob = 1 / (1 + np.exp(-logits))
    y_pred = (y_prob >= 0.5).astype(int)

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "n_features": X_test.shape[1],
        "n_test":     len(y_test),
        "arch":  str(arch_cfg["width"]),
        "n_params": sum(p.numel() for p in model.parameters()),
        "grid": arch_cfg["grid"],
        "k":    arch_cfg["k"],
    }

    roc_auc, pr_auc = plot_roc_pr(y_test, y_prob, out_dir)
    metrics["roc_auc"] = roc_auc
    metrics["pr_auc"]  = pr_auc

    plot_confusion_matrix(y_test, y_pred, out_dir)
    plot_threshold_analysis(y_test, y_prob, out_dir)
    plot_attack_distribution(run_dir, out_dir)
    plot_correlation_heatmap(dataset, features, out_dir)
    write_analysis_report(out_dir, metrics)

    # Save metrics JSON for reference
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print()
    print("Final metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
