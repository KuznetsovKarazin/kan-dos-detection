"""
TON_IoT Binary IDS — KAN Training
Trains the same KAN architecture as CICIDS2017 but on the TON_IoT network dataset.
Preprocesses consistently with the iot-audit baseline project (same 95-feature pipeline).

Usage:
    python scripts/train_toniot.py --csv data/train_test_network.csv [--epochs 200] [--seed 42]

Author: Oleksandr Kuznetsov
"""

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from kan import KAN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Column configuration (matches iot-audit preprocessing pipeline)
# ---------------------------------------------------------------------------

# Columns to drop entirely: IPs, free-text, or perfectly-leaky columns
DROP_COLS = [
    "src_ip", "dst_ip",               # high-cardinality IPs
    "dns_query",                       # free text
    "ssl_subject", "ssl_issuer",       # free text
    "http_uri", "http_user_agent",     # free text
    "http_orig_mime_types", "http_resp_mime_types",  # free text
    "weird_name", "weird_addl",        # free text
    "type",                            # multiclass label — leaks binary label
]

# Target column
TARGET_COL = "label"

# Categorical columns to one-hot encode (low cardinality, informative)
CATEGORICAL_COLS = [
    "proto", "service", "conn_state",
    "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
    "ssl_version", "ssl_cipher", "ssl_resumed", "ssl_established",
    "http_trans_depth", "http_method", "http_version", "http_status_code",
    "weird_notice",
]


def load_and_preprocess(csv_path: Path, max_samples_per_class: int = 50000, seed: int = 42):
    """
    Load TON_IoT CSV, preprocess, and return (X_train, X_test, y_train, y_test,
    scaler, feature_list).

    Preprocessing order (leakage-safe):
      1. Drop irrelevant / leaky columns
      2. Separate into numeric and categorical
      3. Impute / encode on train split only
      4. StandardScale numerics on train, apply to test
      5. Balance classes (both classes limited to max_samples_per_class)
    """
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Raw shape: {df.shape}")

    # --- 1. Drop useless columns -----------------------------------------
    drop_existing = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop_existing)

    # --- 2. Extract target -------------------------------------------------
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Available: {df.columns.tolist()}")
    y_raw = df[TARGET_COL].values.astype(int)
    df = df.drop(columns=[TARGET_COL])

    # Print class distribution
    unique, counts = np.unique(y_raw, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c} samples ({c/len(y_raw)*100:.1f}%)")

    # --- 3. Balance classes (before split to keep it deterministic) -------
    rng = np.random.default_rng(seed)
    idx_0 = np.where(y_raw == 0)[0]
    idx_1 = np.where(y_raw == 1)[0]
    n_per_class = min(max_samples_per_class, len(idx_0), len(idx_1))
    print(f"\nBalancing: {n_per_class} samples per class")

    sel_0 = rng.choice(idx_0, n_per_class, replace=False)
    sel_1 = rng.choice(idx_1, n_per_class, replace=False)
    idx_balanced = np.concatenate([sel_0, sel_1])
    rng.shuffle(idx_balanced)

    df = df.iloc[idx_balanced].reset_index(drop=True)
    y = y_raw[idx_balanced]

    # --- 4. Separate numeric / categorical ---------------------------------
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    num_cols = [c for c in df.columns if c not in cat_cols and pd.api.types.is_numeric_dtype(df[c])]

    print(f"  Numeric cols: {len(num_cols)}")
    print(f"  Categorical cols to encode: {len(cat_cols)}")

    # --- 5. Train/test split FIRST (prevents data leakage) ----------------
    (df_tr, df_te,
     y_tr, y_te) = train_test_split(df, y, test_size=0.2, random_state=seed, stratify=y)

    df_tr = df_tr.reset_index(drop=True)
    df_te = df_te.reset_index(drop=True)

    # --- 6. One-hot encode categoricals (fit on train) --------------------
    # Replace inf/nan in categoricals with "unknown"
    for c in cat_cols:
        df_tr[c] = df_tr[c].astype(str).str.strip().replace({"nan": "unknown", "inf": "unknown"})
        df_te[c] = df_te[c].astype(str).str.strip().replace({"nan": "unknown", "inf": "unknown"})

    ohe_frames_tr, ohe_frames_te = [], []
    ohe_feature_names = []
    for c in cat_cols:
        categories = sorted(df_tr[c].unique())
        for cat in categories:
            col_name = f"{c}_{cat}"
            ohe_frames_tr.append((df_tr[c] == cat).astype(float).rename(col_name))
            ohe_frames_te.append((df_te[c] == cat).astype(float).rename(col_name))
            ohe_feature_names.append(col_name)

    # --- 7. Numeric preprocessing: impute → outlier clip → scale (train only) ---
    X_num_tr = df_tr[num_cols].copy().replace([np.inf, -np.inf], np.nan)
    X_num_te = df_te[num_cols].copy().replace([np.inf, -np.inf], np.nan)

    # Median imputation (fit on train)
    medians = X_num_tr.median()
    X_num_tr = X_num_tr.fillna(medians)
    X_num_te = X_num_te.fillna(medians)

    # Outlier clip: 3 × IQR rule (fit on train)
    for col in num_cols:
        q1, q3 = X_num_tr[col].quantile(0.25), X_num_tr[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        X_num_tr[col] = X_num_tr[col].clip(lo, hi)
        X_num_te[col] = X_num_te[col].clip(lo, hi)

    scaler = StandardScaler()
    X_num_tr_s = scaler.fit_transform(X_num_tr.values)
    X_num_te_s = scaler.transform(X_num_te.values)

    # --- 8. Assemble final feature matrices --------------------------------
    ohe_tr = np.column_stack([s.values for s in ohe_frames_tr]) if ohe_frames_tr else np.zeros((len(df_tr), 0))
    ohe_te = np.column_stack([s.values for s in ohe_frames_te]) if ohe_frames_te else np.zeros((len(df_te), 0))

    X_tr = np.hstack([X_num_tr_s, ohe_tr]).astype(np.float32)
    X_te = np.hstack([X_num_te_s, ohe_te]).astype(np.float32)

    feature_list = num_cols + ohe_feature_names
    print(f"\nFinal feature count: {len(feature_list)}")
    print(f"Train shape: {X_tr.shape},  Test shape: {X_te.shape}")
    print(f"Train class balance: {np.bincount(y_tr)}")

    return X_tr, X_te, y_tr.astype(np.float32), y_te.astype(np.float32), scaler, feature_list, num_cols


def train_kan_model(dataset: dict, input_dim: int, epochs: int = 200) -> tuple:
    """Same KAN architecture as CICIDS2017: [input_dim → 32 → 16 → 1]."""
    model = KAN(width=[input_dim, 32, 16, 1], grid=5, k=3, seed=42)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], "epochs": []}

    print(f"\nModel: [{input_dim} → 32 → 16 → 1]  G=5  k=3")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    for epoch in range(epochs):
        model.train()
        outputs = model(dataset["train_input"])
        loss = criterion(outputs, dataset["train_label"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_pred = (torch.sigmoid(outputs) > 0.5).float()
            train_acc = (train_pred == dataset["train_label"]).float().mean()

            test_out = model(dataset["test_input"])
            test_loss = criterion(test_out, dataset["test_label"])
            test_pred = (torch.sigmoid(test_out) > 0.5).float()
            test_acc = (test_pred == dataset["test_label"]).float().mean()

        history["train_loss"].append(loss.item())
        history["train_acc"].append(train_acc.item())
        history["test_loss"].append(test_loss.item())
        history["test_acc"].append(test_acc.item())
        history["epochs"].append(epoch + 1)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}]  "
                  f"Train: loss={loss.item():.4f} acc={train_acc.item():.4f}  "
                  f"Test:  loss={test_loss.item():.4f} acc={test_acc.item():.4f}")

    return model, history


def save_experiment(run_dir: Path, dataset: dict, scaler, features: list,
                    num_cols: list, model, history: dict) -> None:
    """Save everything in the same format as the CICIDS2017 run."""
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(dataset, run_dir / "dataset.pt")

    with open(run_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(run_dir / "features.pkl", "wb") as f:
        pickle.dump(features, f)

    with open(run_dir / "num_cols.pkl", "wb") as f:
        pickle.dump(num_cols, f)

    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history,
        "architecture": {"width": model.width, "grid": model.grid, "k": model.k},
        "timestamp": datetime.now().isoformat(),
        "dataset": "TON_IoT_binary",
    }, run_dir / "trained_model.pt")

    # Training curves
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["epochs"], history["train_loss"], label="Train Loss")
    plt.plot(history["epochs"], history["test_loss"],  label="Test Loss")
    plt.title("Loss During Training"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history["epochs"], history["train_acc"], label="Train Accuracy")
    plt.plot(history["epochs"], history["test_acc"],  label="Test Accuracy")
    plt.title("Accuracy During Training"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_dir / "training_curves.png", dpi=150)
    plt.close()

    print(f"\nSaved to: {run_dir}")


def main():
    ap = argparse.ArgumentParser(description="Train KAN on TON_IoT dataset")
    ap.add_argument("--csv",     required=True, help="Path to train_test_network.csv")
    ap.add_argument("--epochs",  type=int, default=200)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--max-samples", type=int, default=50000,
                    help="Max samples per class after balancing")
    ap.add_argument("--out-root", default="experiment_data/runs",
                    help="Root directory for run output")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Preprocess
    (X_tr, X_te, y_tr, y_te,
     scaler, features, num_cols) = load_and_preprocess(
        csv_path, max_samples_per_class=args.max_samples, seed=args.seed)

    input_dim = X_tr.shape[1]

    dataset = {
        "train_input": torch.FloatTensor(X_tr),
        "train_label": torch.FloatTensor(y_tr).reshape(-1, 1),
        "test_input":  torch.FloatTensor(X_te),
        "test_label":  torch.FloatTensor(y_te).reshape(-1, 1),
    }

    # Train
    model, history = train_kan_model(dataset, input_dim=input_dim, epochs=args.epochs)

    # Quick evaluation
    model.eval()
    with torch.no_grad():
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        probs = torch.sigmoid(model(dataset["test_input"])).numpy().flatten()
        preds = (probs > 0.5).astype(int)
        y_te_np = y_te.astype(int)
        acc = accuracy_score(y_te_np, preds)
        f1  = f1_score(y_te_np, preds)
        auc = roc_auc_score(y_te_np, probs)
        print(f"\n=== Final Test Metrics ===")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1       : {f1:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{ts}_seed{args.seed}_TON_IoT_binary_w32-16_g5_k3"
    run_dir = Path(args.out_root) / tag
    save_experiment(run_dir, dataset, scaler, features, num_cols, model, history)

    print(f"\n[NEXT] Run LUT sweep:")
    print(f"  python -m src.lut_v2.sweep_all \\")
    print(f"    --run-dir {run_dir} \\")
    print(f"    --Ls 2,4,8,16,32,64,128,256 \\")
    print(f"    --seeds 0,1,2,3,4 \\")
    print(f"    --batch-sizes 1,256")


if __name__ == "__main__":
    main()
