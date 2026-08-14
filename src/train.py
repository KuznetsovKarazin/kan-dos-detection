"""Leakage-free KAN training for the CICIDS2017 DoS case study.

This revision preserves the original NCA case-study architecture and training
hyperparameters while correcting preprocessing leakage and recording a complete
run provenance bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import random
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from kan import KAN
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from src.cicids_preprocessing import prepare_cicids_dos_data
except ModuleNotFoundError:  # supports legacy: python src/train.py
    from cicids_preprocessing import prepare_cicids_dos_data


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _capture_git_state() -> dict[str, Any]:
    """Capture source-tree provenance before the run writes any artifacts."""
    status = _git_value("status", "--porcelain")
    return {
        "commit": _git_value("rev-parse", "HEAD"),
        "branch": _git_value("branch", "--show-current"),
        "dirty": bool(status and status != "unknown"),
        "status_porcelain": [] if not status or status == "unknown" else status.splitlines(),
    }


def _dist_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _pykan_provenance() -> dict[str, Any]:
    out: dict[str, Any] = {"version": _dist_version("pykan")}
    try:
        dist = metadata.distribution("pykan")
        raw = dist.read_text("direct_url.json")
        if raw:
            direct = json.loads(raw)
            out["direct_url"] = direct.get("url")
            out["vcs_info"] = direct.get("vcs_info")
    except Exception:
        pass
    try:
        import kan

        out["module_file"] = str(Path(kan.__file__).resolve())
    except Exception:
        pass
    return out


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _run_name(seed: int, attack_type: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_attack = attack_type.replace(" ", "_")
    return f"{ts}_seed{seed}_{safe_attack}_w32-16_g5_k3_NCA_R1"


def _save_prepared(run_dir: Path, prepared) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(prepared.dataset, run_dir / "dataset.pt")
    with (run_dir / "scaler.pkl").open("wb") as f:
        pickle.dump(prepared.scaler, f)
    with (run_dir / "features.pkl").open("wb") as f:
        pickle.dump(prepared.features, f)
    with (run_dir / "preprocessing.pkl").open("wb") as f:
        pickle.dump(prepared.preprocessing_state, f)
    prepared.feature_stats.to_csv(run_dir / "feature_stats.csv", index=False)
    np.savez_compressed(run_dir / "split_indices.npz", **prepared.split_indices)
    _write_json(run_dir / "preprocessing_manifest.json", prepared.preprocessing_manifest)


def _load_prepared(run_dir: Path):
    dataset = torch.load(run_dir / "dataset.pt", map_location="cpu")
    with (run_dir / "scaler.pkl").open("rb") as f:
        scaler = pickle.load(f)
    with (run_dir / "features.pkl").open("rb") as f:
        features = pickle.load(f)
    with (run_dir / "preprocessing.pkl").open("rb") as f:
        preprocessing_state = pickle.load(f)
    manifest = json.loads((run_dir / "preprocessing_manifest.json").read_text(encoding="utf-8"))
    return dataset, scaler, features, preprocessing_state, manifest


def train_kan_model(
    dataset: dict[str, torch.Tensor],
    input_dim: int,
    *,
    epochs: int = 200,
    seed: int = 42,
    lr: float = 1e-3,
) -> tuple[KAN, dict[str, list[float]]]:
    """Train the original [78,32,16,1], grid=5, k=3 KAN for fixed epochs.

    The held-out test split is intentionally not evaluated inside the training
    loop. It is used only once after the fixed training schedule is complete.
    """
    model = KAN(width=[input_dim, 32, 16, 1], grid=5, k=3, seed=seed, auto_save=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    history: dict[str, list[float]] = {"epochs": [], "train_loss": [], "train_acc": []}

    print(f"\nModel: [{input_dim} -> 32 -> 16 -> 1], grid=5, k=3, seed={seed}")
    print("Training for a fixed schedule; test split is not inspected per epoch.")

    for epoch in range(epochs):
        model.train()
        logits = model(dataset["train_input"])
        loss = criterion(logits, dataset["train_label"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_pred = (torch.sigmoid(logits) >= 0.5).float()
            train_acc = (train_pred == dataset["train_label"]).float().mean().item()

        history["epochs"].append(epoch + 1)
        history["train_loss"].append(float(loss.item()))
        history["train_acc"].append(float(train_acc))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"train_loss={loss.item():.6f} train_acc={train_acc:.6f}"
            )

    return model, history


def _fixed_threshold_metrics(model: KAN, dataset: dict[str, torch.Tensor], threshold: float = 0.5) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(dataset["test_input"])).cpu().numpy().reshape(-1)
    y_true = dataset["test_label"].cpu().numpy().astype(int).reshape(-1)
    y_pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "metrics": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, probs)),
            "pr_auc": float(average_precision_score(y_true, probs)),
        },
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        },
        "class_counts": {
            "test_benign": int((y_true == 0).sum()),
            "test_attack": int((y_true == 1).sum()),
        },
    }


def _plot_training_curves(history: dict[str, list[float]], run_dir: Path) -> None:
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(history["epochs"], history["train_loss"])
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="BCEWithLogits loss")
    axes[0].grid(True)
    axes[1].plot(history["epochs"], history["train_acc"])
    axes[1].set(title="Training accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].grid(True)
    fig.tight_layout()
    fig.savefig(fig_dir / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_model(run_dir: Path, model: KAN, history: dict[str, list[float]], seed: int) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
            # Store constructor arguments, not PyKAN's internal model.width representation.
            "architecture": {"width": [78, 32, 16, 1], "grid": 5, "k": 3, "seed": seed, "auto_save": False},
            "timestamp": datetime.now().astimezone().isoformat(),
            "revision": "NCA_R1_leakage_free",
        },
        run_dir / "trained_model.pt",
    )


def _build_run_meta(
    args,
    run_dir: Path,
    data_path: Path,
    manifest: dict[str, Any],
    git_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "created_at": datetime.now().astimezone().isoformat(),
        "git": dict(git_state),
        "config": {
            "data_path": str(data_path),
            "attack_type": args.attack_type,
            "max_samples_per_class": args.max_samples_per_class,
            "test_size": args.test_size,
            "seed": args.seed,
            "epochs": args.epochs,
            "lr": args.lr,
            "width": [78, 32, 16, 1],
            "grid": 5,
            "k": 3,
            "threshold": 0.5,
            "preprocessing": "train-fit median -> 3xIQR clipping -> StandardScaler",
        },
        "dataset": {
            "sha256": _sha256_file(data_path),
            "selected_total": manifest["selected_total"],
            "train_total": manifest["train_total"],
            "test_total": manifest["test_total"],
            "feature_count": manifest["feature_count"],
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "pykan": _pykan_provenance(),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Leakage-free CICIDS2017 DoS KAN training for NCA R1")
    ap.add_argument("--data", default="data/Wednesday-workingHours.pcap_ISCX.csv")
    ap.add_argument("--attack-type", default="DoS Hulk")
    ap.add_argument("--max-samples-per-class", type=int, default=231_073)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-root", default="experiment_data/runs")
    ap.add_argument("--run-dir", default=None, help="Exact run directory; otherwise a timestamped directory is created")
    ap.add_argument("--prepare-only", action="store_true", help="Prepare and audit data, save bundle, then stop before training")
    ap.add_argument("--reuse-prepared", action="store_true", help="Reuse dataset/preprocessing already saved in --run-dir")
    args = ap.parse_args()

    _set_seeds(args.seed)
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    # Record repository state before creating/updating any run artifacts.
    git_state = _capture_git_state()

    run_dir = Path(args.run_dir) if args.run_dir else Path(args.out_root) / _run_name(args.seed, args.attack_type)

    if args.reuse_prepared:
        if args.run_dir is None:
            raise ValueError("--reuse-prepared requires --run-dir")
        dataset, scaler, features, preprocessing_state, manifest = _load_prepared(run_dir)
        expected = {
            "attack_type": args.attack_type,
            "selected_samples_per_class": int(args.max_samples_per_class),
            "test_size": float(args.test_size),
            "seed": int(args.seed),
            "feature_count": 78,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(
                    f"Prepared-run mismatch for {key}: saved={manifest.get(key)!r}, requested={value!r}"
                )
        print(f"Reusing prepared data from: {run_dir}")
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"Run directory already contains files: {run_dir}. "
                "Use --reuse-prepared with the same --run-dir, or choose a new run directory."
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        prepared = prepare_cicids_dos_data(
            data_path,
            attack_type=args.attack_type,
            max_samples_per_class=args.max_samples_per_class,
            test_size=args.test_size,
            seed=args.seed,
            expected_feature_count=78,
        )
        _save_prepared(run_dir, prepared)
        dataset = prepared.dataset
        scaler = prepared.scaler
        features = prepared.features
        preprocessing_state = prepared.preprocessing_state
        manifest = prepared.preprocessing_manifest

    run_meta = _build_run_meta(args, run_dir, data_path, manifest, git_state)
    _write_json(run_dir / "run_meta.json", run_meta)

    print("\n=== Preprocessing audit ===")
    print(f"Selected/class : {manifest['selected_samples_per_class']}")
    print(f"Selected total : {manifest['selected_total']}")
    print(f"Train total    : {manifest['train_total']}")
    print(f"Test total     : {manifest['test_total']}")
    print(f"Features       : {manifest['feature_count']}")
    print(f"Train classes  : {manifest['train_class_counts']}")
    print(f"Test classes   : {manifest['test_class_counts']}")
    print(f"Split disjoint : {manifest['split_disjoint']}")
    print(f"Run directory  : {run_dir}")

    if args.prepare_only:
        print("\nPREPARE-ONLY complete. No model was trained.")
        return

    model, history = train_kan_model(
        dataset, input_dim=len(features), epochs=args.epochs, seed=args.seed, lr=args.lr
    )
    _save_model(run_dir, model, history, args.seed)
    _plot_training_curves(history, run_dir)

    metrics = _fixed_threshold_metrics(model, dataset, threshold=0.5)
    metrics.update({"run_dir": str(run_dir), "timestamp": datetime.now().isoformat()})
    _write_json(run_dir / "metrics.json", metrics)

    print("\n=== Final held-out test metrics (threshold=0.5) ===")
    for key, value in metrics["metrics"].items():
        print(f"{key:>10s}: {value:.8f}")
    print(f"confusion : {metrics['confusion_matrix']}")
    print(f"\nSaved reproducible R1 run to: {run_dir}")


if __name__ == "__main__":
    main()
