"""Analysis for a saved CICIDS2017 KAN run.

Reported metrics use a fixed threshold (default 0.5). Threshold sweeps are
produced only as diagnostics and never replace the reported decision rule.
"""
from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from kan import KAN
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def load_experiment(load_dir: str | Path):
    load_dir = Path(load_dir)
    dataset = torch.load(load_dir / "dataset.pt", map_location="cpu")
    model_data = torch.load(load_dir / "trained_model.pt", map_location="cpu")
    with (load_dir / "scaler.pkl").open("rb") as f:
        scaler = pickle.load(f)
    with (load_dir / "features.pkl").open("rb") as f:
        features = pickle.load(f)
    return dataset, model_data, scaler, features


def _predictions(model, dataset):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(dataset["test_input"])).cpu().numpy().reshape(-1)
    y_true = dataset["test_label"].cpu().numpy().astype(int).reshape(-1)
    return y_true, probs


def fixed_threshold_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "metrics": {
            "accuracy": float(accuracy_score(y_true, pred)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, probs)),
            "pr_auc": float(average_precision_score(y_true, probs)),
        },
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "class_counts": {
            "test_benign": int((y_true == 0).sum()),
            "test_attack": int((y_true == 1).sum()),
        },
    }


def plot_confusion(y_true, probs, threshold: float, save_dir: Path) -> None:
    pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set(title=f"Confusion matrix (threshold={threshold:g})", xlabel="Predicted", ylabel="Actual")
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100.0
    sns.heatmap(cm_pct, annot=True, fmt=".2f", cmap="Blues", ax=axes[1])
    axes[1].set(title="Row-normalized (%)", xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    fig.savefig(save_dir / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr(y_true, probs, save_dir: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, probs)
    precision, recall, _ = precision_recall_curve(y_true, probs)
    roc_auc = roc_auc_score(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(fpr, tpr, label=f"ROC-AUC={roc_auc:.5f}")
    axes[0].plot([0, 1], [0, 1], "--")
    axes[0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
    axes[0].legend(); axes[0].grid(True)
    axes[1].plot(recall, precision, label=f"PR-AUC={pr_auc:.5f}")
    axes[1].set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision")
    axes[1].legend(); axes[1].grid(True)
    fig.tight_layout()
    fig.savefig(save_dir / "roc_pr_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_diagnostic(y_true, probs, save_dir: Path) -> None:
    thresholds = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(y_true, (probs >= t).astype(int), zero_division=0) for t in thresholds]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(thresholds, f1s)
    ax.axvline(0.5, linestyle="--", label="reported threshold = 0.5")
    ax.set(title="Diagnostic threshold sweep", xlabel="Threshold", ylabel="F1")
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_dir / "threshold_diagnostic.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze a saved CICIDS2017 KAN run")
    ap.add_argument("--load-dir", default="experiment_data")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    load_dir = Path(args.load_dir)
    analysis_dir = load_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    dataset, model_data, scaler, features = load_experiment(load_dir)
    model = KAN(**model_data["architecture"])
    model.load_state_dict(model_data["model_state_dict"])

    y_true, probs = _predictions(model, dataset)
    metrics = fixed_threshold_metrics(y_true, probs, args.threshold)
    metrics.update({"run_dir": str(load_dir), "timestamp": datetime.now().isoformat()})

    pred = (probs >= args.threshold).astype(int)
    print(classification_report(y_true, pred, digits=6))
    print(json.dumps(metrics, indent=2))

    plot_confusion(y_true, probs, args.threshold, analysis_dir)
    plot_roc_pr(y_true, probs, analysis_dir)
    plot_threshold_diagnostic(y_true, probs, analysis_dir)

    (load_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Analysis saved under: {analysis_dir}")


if __name__ == "__main__":
    main()
