from pathlib import Path

import numpy as np
import pandas as pd

from src.cicids_preprocessing import prepare_cicids_dos_data


def _synthetic_csv(path: Path) -> pd.DataFrame:
    n = 30
    df = pd.DataFrame(
        {
            "f1": np.arange(2 * n, dtype=float),
            "f2": np.concatenate([np.linspace(-2, 2, n), np.linspace(10, 20, n)]),
            " Label": ["BENIGN"] * n + ["DoS Hulk"] * n,
        }
    )
    # Include values whose treatment depends on train-fitted statistics.
    df.loc[3, "f1"] = np.inf
    df.loc[40, "f2"] = np.nan
    df.to_csv(path, index=False)
    return df


def test_preprocessing_is_train_fit_and_split_is_disjoint(tmp_path):
    csv_path = tmp_path / "toy.csv"
    raw = _synthetic_csv(csv_path)
    prepared = prepare_cicids_dos_data(
        csv_path,
        max_samples_per_class=30,
        test_size=0.25,
        seed=42,
        expected_feature_count=2,
    )

    train_rows = prepared.split_indices["train_source_rows"]
    test_rows = prepared.split_indices["test_source_rows"]
    assert set(train_rows.tolist()).isdisjoint(set(test_rows.tolist()))

    train_raw = raw.loc[train_rows, prepared.features].replace([np.inf, -np.inf], np.nan)
    expected_median = train_raw.median()
    for feature in prepared.features:
        assert np.isclose(
            prepared.preprocessing_state["median"][feature], expected_median[feature]
        )

    manifest = prepared.preprocessing_manifest
    assert manifest["train_total"] == 45
    assert manifest["test_total"] == 15
    assert manifest["feature_count"] == 2
    assert manifest["split_disjoint"] is True


def test_preprocessing_is_deterministic_for_fixed_seed(tmp_path):
    csv_path = tmp_path / "toy.csv"
    _synthetic_csv(csv_path)
    a = prepare_cicids_dos_data(
        csv_path, max_samples_per_class=30, test_size=0.2, seed=7, expected_feature_count=2
    )
    b = prepare_cicids_dos_data(
        csv_path, max_samples_per_class=30, test_size=0.2, seed=7, expected_feature_count=2
    )
    assert np.array_equal(a.split_indices["train_source_rows"], b.split_indices["train_source_rows"])
    assert np.array_equal(a.split_indices["test_source_rows"], b.split_indices["test_source_rows"])
    assert np.array_equal(a.dataset["train_input"].numpy(), b.dataset["train_input"].numpy())
    assert np.array_equal(a.dataset["test_input"].numpy(), b.dataset["test_input"].numpy())
