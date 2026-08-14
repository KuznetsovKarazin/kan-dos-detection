"""Leakage-free CICIDS2017 preprocessing for the NCA revision.

The key contract is that all feature-dependent preprocessing statistics are
fitted on the training split only and then frozen for application to the test
split. Class balancing is label-only and therefore performed before the split.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class PreparedCICIDS:
    dataset: dict[str, torch.Tensor]
    scaler: StandardScaler
    features: list[str]
    preprocessing_state: dict[str, Any]
    preprocessing_manifest: dict[str, Any]
    feature_stats: pd.DataFrame
    split_indices: dict[str, np.ndarray]


def _label_column(df: pd.DataFrame) -> str:
    if " Label" in df.columns:
        return " Label"
    matches = [c for c in df.columns if c.strip() == "Label"]
    if len(matches) == 1:
        return matches[0]
    raise ValueError("Could not identify CICIDS2017 label column")


def prepare_cicids_dos_data(
    filepath: str | Path,
    *,
    attack_type: str = "DoS Hulk",
    max_samples_per_class: int = 231_073,
    test_size: float = 0.2,
    seed: int = 42,
    expected_feature_count: int | None = 78,
) -> PreparedCICIDS:
    """Prepare the balanced CICIDS2017 DoS cohort without preprocessing leakage.

    Order of operations:
      1. Select BENIGN and the requested attack class (label-only balancing).
      2. Stratified train/test split.
      3. Replace +/-inf by NaN.
      4. Fit medians on train; apply to train/test.
      5. Fit 3*IQR clipping bounds on train; apply to train/test.
      6. Fit StandardScaler on train; transform train/test.

    The function also preserves original CSV row indices so that the split can
    be audited and reproduced independently of the serialized tensors.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(filepath)
    if not (0.0 < float(test_size) < 1.0):
        raise ValueError("test_size must be between 0 and 1")
    if int(max_samples_per_class) <= 0:
        raise ValueError("max_samples_per_class must be positive")

    df = pd.read_csv(filepath)
    label_col = _label_column(df)
    raw_class_counts = df[label_col].value_counts(dropna=False).to_dict()

    benign_all = df[df[label_col] == "BENIGN"]
    attack_all = df[df[label_col] == attack_type]
    n_per_class = min(
        int(max_samples_per_class), len(benign_all), len(attack_all)
    )
    if n_per_class <= 0:
        raise ValueError(
            f"Requested classes are unavailable: BENIGN={len(benign_all)}, "
            f"{attack_type}={len(attack_all)}"
        )

    # Match the historical cohort-selection method while keeping source rows.
    benign = benign_all.sample(n=n_per_class, random_state=seed).copy()
    attack = attack_all.sample(n=n_per_class, random_state=seed).copy()
    cohort = pd.concat([benign, attack], axis=0)
    source_rows = cohort.index.to_numpy(dtype=np.int64)
    y = (cohort[label_col] != "BENIGN").to_numpy(dtype=np.int64)

    numeric_cols = cohort.select_dtypes(include=[np.number]).columns.tolist()
    if expected_feature_count is not None and len(numeric_cols) != int(expected_feature_count):
        raise ValueError(
            f"Expected {expected_feature_count} numeric features, found {len(numeric_cols)}"
        )

    X_raw = cohort[numeric_cols].copy()
    (
        X_train_raw,
        X_test_raw,
        y_train,
        y_test,
        idx_train,
        idx_test,
    ) = train_test_split(
        X_raw,
        y,
        source_rows,
        test_size=float(test_size),
        random_state=int(seed),
        stratify=y,
    )

    # Work on independent copies and remove infinities before fitting statistics.
    X_train = X_train_raw.replace([np.inf, -np.inf], np.nan).copy()
    X_test = X_test_raw.replace([np.inf, -np.inf], np.nan).copy()

    medians = X_train.median(axis=0)
    if medians.isna().any():
        bad = medians[medians.isna()].index.tolist()
        raise ValueError(f"Training split has all-NaN features after inf replacement: {bad}")
    train_missing_before = X_train.isna().sum()
    test_missing_before = X_test.isna().sum()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    q1 = X_train.quantile(0.25)
    q3 = X_train.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 3.0 * iqr
    upper = q3 + 3.0 * iqr

    train_clip_count = ((X_train < lower) | (X_train > upper)).sum()
    test_clip_count = ((X_test < lower) | (X_test > upper)).sum()

    X_train = X_train.clip(lower=lower, upper=upper, axis=1)
    X_test = X_test.clip(lower=lower, upper=upper, axis=1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32, copy=False)
    X_test_scaled = scaler.transform(X_test).astype(np.float32, copy=False)

    dataset = {
        "train_input": torch.from_numpy(X_train_scaled),
        "train_label": torch.from_numpy(y_train.astype(np.float32)).reshape(-1, 1),
        "test_input": torch.from_numpy(X_test_scaled),
        "test_label": torch.from_numpy(y_test.astype(np.float32)).reshape(-1, 1),
    }

    feature_stats = pd.DataFrame(
        {
            "feature": numeric_cols,
            "median_train": medians.values,
            "q1_train": q1.values,
            "q3_train": q3.values,
            "iqr_train": iqr.values,
            "lower_bound_train": lower.values,
            "upper_bound_train": upper.values,
            "train_missing_imputed": train_missing_before.values,
            "test_missing_imputed": test_missing_before.values,
            "train_values_clipped": train_clip_count.values,
            "test_values_clipped": test_clip_count.values,
        }
    )

    preprocessing_state = {
        "features": list(numeric_cols),
        "median": medians.to_dict(),
        "q1": q1.to_dict(),
        "q3": q3.to_dict(),
        "iqr": iqr.to_dict(),
        "lower_bound": lower.to_dict(),
        "upper_bound": upper.to_dict(),
        "scaler": scaler,
        "contract": "split_then_train_fit_impute_3xIQR_clip_standardize",
    }

    train_counts = np.bincount(y_train, minlength=2)
    test_counts = np.bincount(y_test, minlength=2)
    preprocessing_manifest = {
        "dataset_path": str(filepath),
        "label_column": label_col,
        "attack_type": attack_type,
        "raw_rows": int(len(df)),
        "raw_columns": int(df.shape[1]),
        "raw_class_counts": {str(k): int(v) for k, v in raw_class_counts.items()},
        "selected_samples_per_class": int(n_per_class),
        "selected_total": int(2 * n_per_class),
        "feature_count": int(len(numeric_cols)),
        "features": list(numeric_cols),
        "test_size": float(test_size),
        "seed": int(seed),
        "train_total": int(len(y_train)),
        "test_total": int(len(y_test)),
        "train_class_counts": {"BENIGN": int(train_counts[0]), attack_type: int(train_counts[1])},
        "test_class_counts": {"BENIGN": int(test_counts[0]), attack_type: int(test_counts[1])},
        "preprocessing_order": [
            "label-only class balancing",
            "stratified train/test split",
            "replace +/-inf with NaN",
            "median imputation fitted on train only",
            "3xIQR clipping bounds fitted on train only",
            "StandardScaler fitted on train only",
        ],
        "outlier_handling": "clip/winsorize; rows are not removed",
        "split_disjoint": bool(set(idx_train.tolist()).isdisjoint(set(idx_test.tolist()))),
    }

    return PreparedCICIDS(
        dataset=dataset,
        scaler=scaler,
        features=list(numeric_cols),
        preprocessing_state=preprocessing_state,
        preprocessing_manifest=preprocessing_manifest,
        feature_stats=feature_stats,
        split_indices={
            "selected_source_rows": source_rows,
            "train_source_rows": np.asarray(idx_train, dtype=np.int64),
            "test_source_rows": np.asarray(idx_test, dtype=np.int64),
        },
    )
