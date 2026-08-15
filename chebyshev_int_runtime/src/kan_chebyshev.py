#!/usr/bin/env python3
"""
KAN-IDS — Tappa 0 + Tappa 1  (v2: usa il preprocessing della tesi)
========================================================================
Classificatore KAN (base Chebyshev) per intrusion detection binaria su
TON_IoT, agganciato allo SPAZIO UNIFICATO a 10 feature numeriche della
tesi (section_310_unified_feature_engineering.py).

Perche' lo spazio unificato a 10 feature e non le 95:
  - 10 input -> KAN deployabile su MCU (95 farebbero esplodere gli edge)
  - feature gia' log-trasformate e su scale gestibili, Chebyshev-friendly
  - aggancio al filone domain-shift gia' presente nella tesi
  - confronto EQUO: la KAN gareggia con baseline sullo STESSO spazio a 10

CONFRONTO ONESTO
----------------
NON confrontiamo la KAN col LightGBM a 95 feature (F1=0.9992): sarebbe
ingiusto, spazi diversi. Confrontiamo KAN vs LogReg/MLP-piccolo sulle
STESSE 10 feature unificate.

DATI
----
  --csv path/to/train_test_network.csv   (Kaggle TON_IoT network)

Richiede nel path: section_310_unified_feature_engineering.py
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import section_310_unified_feature_engineering as fe


# =============================================================================
# TAPPA 0 — Dati -> spazio unificato a 10 feature (preprocessing della tesi)
# =============================================================================

def load_ton_unified(csv_path, label_col="label"):
    """Carica TON_IoT grezzo e applica build_unified_features_ton della tesi.
    Ritorna X (10 feature numeriche, z-scored) e y binario."""
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        label_col = "label" if "label" in df.columns else df.columns[-1]
    y = df[label_col].astype(int).to_numpy()

    Xfe = fe.build_unified_features_ton(df)            # 13 col (10 num + 3 cat)
    Xnum = Xfe[fe.UNIFIED_NUMERIC_FEATURES].to_numpy(dtype=np.float64)
    return Xnum, y, list(fe.UNIFIED_NUMERIC_FEATURES)


def make_fake_raw_ton(n=20000, seed=42):
    """Mini-TON_IoT GREZZO finto (colonne src_bytes, dst_bytes, ...) con un
    minimo di segnale separabile, per testare la meccanica end-to-end."""
    rng = np.random.RandomState(seed)
    y = (rng.rand(n) > 0.22).astype(int)
    # attacchi: piu' asimmetria e burst, durate piu' corte
    src = np.where(y == 1, rng.gamma(2, 8000, n), rng.gamma(2, 3000, n))
    dst = np.where(y == 1, rng.gamma(1.2, 1500, n), rng.gamma(2, 3000, n))
    spk = np.where(y == 1, rng.randint(1, 300, n), rng.randint(1, 80, n))
    dpk = np.where(y == 1, rng.randint(1, 40, n), rng.randint(1, 80, n))
    dur = np.where(y == 1, rng.rand(n) * 2 + 0.01, rng.rand(n) * 12 + 0.01)
    df = pd.DataFrame({
        "src_bytes": src.astype(int), "dst_bytes": dst.astype(int),
        "src_pkts": spk, "dst_pkts": dpk, "duration": dur,
        "proto": rng.choice(["tcp", "udp", "icmp"], n),
        "label": y,
    })
    return df


# =============================================================================
# TAPPA 1 — KAN Chebyshev classifier (NumPy puro, training BCE)
#   (single-layer: gradiente esatto; e' la versione che proponiamo per il
#    primo deployment. L'hidden layer resta lavoro successivo.)
# =============================================================================

def chebyshev_basis(x, degree):
    N = x.shape[0]
    T = np.empty((N, degree + 1), dtype=np.float64)
    T[:, 0] = 1.0
    if degree >= 1:
        T[:, 1] = x
    for n in range(2, degree + 1):
        T[:, n] = 2.0 * x * T[:, n - 1] - T[:, n - 2]
    return T


class ChebyshevKANBinary:
    """KAN single-layer [in_dim -> 1] con base Chebyshev + sigmoid.
    out(x) = sum_i  sum_d  coeffs[i,d] * T_d( norm(x_i) )
    Training: gradient descent su BCE pesata. Gradiente ESATTO."""

    def __init__(self, in_dim, degree=8, x_min=-3.5, x_max=3.5, seed=0):
        self.in_dim, self.degree = in_dim, degree
        self.x_min, self.x_max = x_min, x_max
        rng = np.random.RandomState(seed)
        self.coeffs = (rng.randn(in_dim, degree + 1) * 0.05)   # (in_dim, deg+1)

    def _norm(self, X):
        xn = 2.0 * (X - self.x_min) / (self.x_max - self.x_min) - 1.0
        return np.clip(xn, -1.0, 1.0)

    def _logits(self, X):
        Xn = self._norm(X)
        # bases[i] -> (N, deg+1); logit = sum_i bases[i] @ coeffs[i]
        self._bases = [chebyshev_basis(Xn[:, i], self.degree)
                       for i in range(self.in_dim)]
        z = np.zeros(X.shape[0])
        for i in range(self.in_dim):
            z += self._bases[i] @ self.coeffs[i]
        return z

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y, epochs=400, lr=0.3, l2=1e-4, verbose=True):
        y = y.astype(np.float64)
        pos = y.mean()
        w_pos, w_neg = 0.5 / max(pos, 1e-6), 0.5 / max(1 - pos, 1e-6)
        sw = np.where(y == 1, w_pos, w_neg)
        N = X.shape[0]
        for ep in range(epochs):
            z = self._logits(X)
            p = self._sigmoid(z)
            g = sw * (p - y)                       # dL/dz  (N,)
            for i in range(self.in_dim):
                grad_i = (self._bases[i] * g[:, None]).mean(0) + l2 * self.coeffs[i]
                self.coeffs[i] -= lr * grad_i
            if verbose and (ep % 80 == 0 or ep == epochs - 1):
                loss = -np.mean(sw * (y * np.log(p + 1e-9) +
                                      (1 - y) * np.log(1 - p + 1e-9)))
                print(f"   epoch {ep:4d}/{epochs}  BCE={loss:.4f}")

    def predict_proba(self, X):
        return self._sigmoid(self._logits(X))

    def predict(self, X, thr=0.5):
        return (self.predict_proba(X) >= thr).astype(int)


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--degree", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--clip", type=float, default=3.5,
                    help="dominio Chebyshev [-clip,clip] dopo z-score")
    args = ap.parse_args()

    print("=" * 66)
    print("KAN-IDS v2  —  spazio unificato 10-feature (preprocessing tesi)")
    print("=" * 66)

    # ---- TAPPA 0 ----
    if args.csv:
        print(f"\n[Tappa 0] TON_IoT reale: {args.csv}")
        Xraw, y, cols = load_ton_unified(args.csv)
        mode = "REALE"
    else:
        print("\n[Tappa 0] Nessun CSV -> mini-TON_IoT GREZZO finto (test meccanica)")
        df = make_fake_raw_ton()
        Xraw, y, cols = (df.pipe(lambda d: fe.build_unified_features_ton(d))
                         [fe.UNIFIED_NUMERIC_FEATURES].to_numpy(np.float64),
                         df["label"].to_numpy(), list(fe.UNIFIED_NUMERIC_FEATURES))
        mode = "SINTETICA"

    print(f"   modalita': {mode}  |  feature: {len(cols)}  |  campioni: {len(y)}")
    print(f"   positivi (attacco): {y.mean()*100:.1f}%")

    # z-score sul train (parametri solo train, come la tesi)
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        Xraw, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(Xtr_raw)
    Xtr = np.clip(scaler.transform(Xtr_raw), -args.clip, args.clip)
    Xte = np.clip(scaler.transform(Xte_raw), -args.clip, args.clip)

    # ---- BASELINE sulle stesse 10 feature (confronto equo) ----
    print("\n[Baseline] sulle stesse 10 feature unificate:")
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    f1_lr = f1_score(yte, lr.predict(Xte))
    auc_lr = roc_auc_score(yte, lr.predict_proba(Xte)[:, 1])
    print(f"   LogReg     F1={f1_lr:.4f}  AUC={auc_lr:.4f}")

    mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=300,
                        random_state=0).fit(Xtr, ytr)
    f1_mlp = f1_score(yte, mlp.predict(Xte))
    auc_mlp = roc_auc_score(yte, mlp.predict_proba(Xte)[:, 1])
    print(f"   MLP(16)    F1={f1_mlp:.4f}  AUC={auc_mlp:.4f}")

    # ---- TAPPA 1: KAN ----
    print(f"\n[Tappa 1] KAN Chebyshev [{len(cols)}->1], degree={args.degree}")
    kan = ChebyshevKANBinary(in_dim=len(cols), degree=args.degree,
                             x_min=-args.clip, x_max=args.clip)
    kan.fit(Xtr, ytr, epochs=args.epochs, lr=0.3)
    pk = kan.predict(Xte)
    f1_k = f1_score(yte, pk)
    auc_k = roc_auc_score(yte, kan.predict_proba(Xte))
    prec_k = precision_score(yte, pk, zero_division=0)
    rec_k = recall_score(yte, pk, zero_division=0)

    # ---- STOP ONESTO ----
    print("\n" + "=" * 66)
    print("STOP ONESTO")
    print("=" * 66)
    print(f"  {'modello (10 feature unificate)':<34}{'F1':>9}{'ROC-AUC':>11}")
    print("  " + "-" * 54)
    print(f"  {'LogReg':<34}{f1_lr:>9.4f}{auc_lr:>11.4f}")
    print(f"  {'MLP(16)':<34}{f1_mlp:>9.4f}{auc_mlp:>11.4f}")
    print(f"  {'KAN Chebyshev deg'+str(args.degree):<34}{f1_k:>9.4f}{auc_k:>11.4f}")
    print(f"  {'  KAN precision / recall':<34}{prec_k:>9.4f}{rec_k:>11.4f}")
    print()
    n_edges = len(cols)               # single-layer: in_dim edge
    print(f"  edge KAN da quantizzare in LUT: {n_edges}")
    print(f"  (single-layer -> deployment MCU molto leggero)")
    print()
    if mode == "REALE":
        ref = max(f1_lr, f1_mlp)
        verdict = ("PROSEGUIRE -> tappa 2 (export LUT)" if f1_k >= ref - 0.02
                   else "RIVEDERE: prova degree piu' alto o hidden layer")
        print(f"  KAN vs miglior baseline a 10 feat: ΔF1 = {f1_k - ref:+.4f}")
        print(f"  VERDETTO: {verdict}")
    else:
        print("  [!] SINTETICO: numeri non confrontabili col paper.")
    print("=" * 66)


if __name__ == "__main__":
    main()
