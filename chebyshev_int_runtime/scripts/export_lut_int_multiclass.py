#!/usr/bin/env python3
"""
export_lut_int_multiclass.py — export LUT INTEGER-ONLY multiclass
=================================================================
Estende export_lut_int.py (binario) al caso multiclass a C classi.

Struttura: KAN single-layer [in_dim -> C]. Ogni coppia (input i, classe c)
ha un edge univariato phi_{i,c}(x_i). Il logit della classe c e':
    z_c = sum_i phi_{i,c}(x_i)
La decisione e' argmax_c z_c (niente softmax: l'argmax dei logit basta).

Tutto INTERO (come il fully-integer binario):
  - tabella int16 pre-scalata: val = round(S * phi)
  - forward: lookup int16 -> interp intera -> accumulo int32 per classe
  - decisione: argmax dei C accumulatori int32. Zero float nel loop.

E_tot = in_dim * C edge (es. 10*10 = 100). ESP32-only (non entra nel Mega).
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _p in [_REPO, _REPO / "src", _REPO / "preprocessing"]:
    sys.path.insert(0, str(_p))


def _find_lut_kan():
    import os
    cands = []
    if os.environ.get("LUT_KAN_PATH"):
        cands.append(Path(os.environ["LUT_KAN_PATH"]))
    here = Path(__file__).resolve().parent
    cands += [_REPO / "lut-kan", here / "lut-kan", Path.cwd() / "lut-kan", _REPO]
    for c in cands:
        if (c / "src" / "quant" / "lut_builder.py").exists():
            return c
    raise FileNotFoundError("lut-kan non trovata; clonala nella root del repo.")

LUT_KAN = _find_lut_kan()
sys.path.insert(0, str(LUT_KAN))
from src.quant.lut_builder import build_lut_for_edges          # noqa: E402
from export_lut import ChebyshevEdge                            # riuso adapter

FP_BITS = 9
S = 1 << FP_BITS


def kan_mc_to_edges(kan, num_knots=9):
    """Da ChebyshevKANMulticlass a lista di edge in ordine (input, classe):
    edge index = i*C + c. coeffs[i, c] e' il vettore 1D dei coefficienti."""
    edges = []
    for i in range(kan.in_dim):
        for c in range(kan.C):
            edges.append(ChebyshevEdge(kan.coeffs[i, c], kan.x_min, kan.x_max, num_knots))
    return edges


def build_int_table_mc(kan, L=64, num_knots=9):
    """LUT int16 pre-scalata per tutti gli in_dim*C edge."""
    edges = kan_mc_to_edges(kan, num_knots=num_knots)
    art = build_lut_for_edges(
        edges=edges, L=L, interp="linear", oob_behavior="clip",
        boundary_mode="half_open", y_range_method="minmax",
        lower_pct=0.1, upper_pct=99.9, dtype="uint8", scheme="asymmetric",
        qmin=0, qmax=255, meta_dtype="float16", value_representation="phi")
    q = art.q_table.astype(np.float64)
    scale = art.scale.astype(np.float64)[:, :, None]
    ymin = art.y_min.astype(np.float64)[:, :, None]
    phi = ymin + scale * q
    table_int = np.round(S * phi).astype(np.int16)
    return art, table_int


def int_forward_argmax(art, table_int, X, in_dim, C, L):
    """Forward INTERO multiclass: per ogni classe accumula i suoi in_dim edge,
    poi argmax. Replica esatta del runtime C. Ritorna le predizioni."""
    knots = art.knots
    x_min, x_max = float(knots[0]), float(knots[-1])
    K = int(knots.size - 1)
    dx = (x_max - x_min) / K
    N = X.shape[0]
    Z = np.zeros((N, C), dtype=np.int64)
    for i in range(in_dim):
        x = X[:, i].astype(np.float64)
        xc = np.clip(x, x_min, x_max - 1e-6)
        t = (xc - x_min) / dx
        k = np.floor(t).astype(int); k = np.clip(k, 0, K - 1)
        u = np.clip((xc - (x_min + k * dx)) / dx, 0.0, 1.0)
        pos = u * (L - 1)
        r0 = np.clip(np.floor(pos).astype(int), 0, L - 1)
        r1 = np.clip(r0 + 1, 0, L - 1)
        frac256 = np.clip(((pos - r0) * 256).astype(np.int64), 0, 256)
        for c in range(C):
            e = i * C + c
            q0 = table_int[e][k, r0].astype(np.int64)
            q1 = table_int[e][k, r1].astype(np.int64)
            v = q0 + ((frac256 * (q1 - q0)) >> 8)
            Z[:, c] += v
    return np.argmax(Z, axis=1), Z


def write_int_c_header_mc(art, table_int, path, in_dim, C, L,
                          name="kan_ids_mc_int"):
    E = table_int.shape[0]      # = in_dim * C
    K = table_int.shape[1]
    knots = art.knots
    lines = [
        f"// {name}.h — KAN-IDS multiclass LUT INTEGER-ONLY",
        f"// {datetime.now().isoformat(timespec='seconds')}",
        f"// in_dim={in_dim}, C={C}, E={E} edge (=in_dim*C), K={K}, L={L}, FP_BITS={FP_BITS}",
        "// edge index = i*C + c. logit classe c = sum_i phi_{i,c}(x_i).",
        "// decisione = argmax_c logit_c. Inferenza interamente intera.",
        "#pragma once",
        "#include <stdint.h>",
        "#ifdef __AVR__\n#include <avr/pgmspace.h>\n#else\n#define PROGMEM\n#endif",
        f"#define KANMC_INDIM {in_dim}",
        f"#define KANMC_C {C}",
        f"#define KANMC_E {E}",
        f"#define KANMC_K {K}",
        f"#define KANMC_L {L}",
        f"#define KANMC_FP_BITS {FP_BITS}",
        f"static const float KANMC_XMIN = {float(knots[0]):.8f}f;",
        f"static const float KANMC_XMAX = {float(knots[-1]):.8f}f;",
    ]
    flat = table_int.reshape(E, K * L)
    lines.append(f"static const int16_t KANMC_TABLE[{E}][{K*L}] PROGMEM = {{")
    for e in range(E):
        lines.append("  {" + ",".join(str(int(v)) for v in flat[e]) + "},")
    lines.append("};")
    Path(path).write_text("\n".join(lines))
    return path


def main():
    import argparse
    import pandas as pd
    from sklearn.preprocessing import QuantileTransformer, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.metrics import f1_score
    from kan_chebyshev_multiclass import ChebyshevKANMulticlass

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="train_test_network.csv")
    ap.add_argument("--out", default="kan_ids_mc_int.h")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--degree", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--L", type=int, default=64)
    args = ap.parse_args()
    CLIP = 3.5; K = 10
    NUM = ["src_port","dst_port","duration","src_bytes","dst_bytes","missed_bytes",
           "src_pkts","src_ip_bytes","dst_pkts","dst_ip_bytes","dns_qclass",
           "dns_qtype","dns_rcode","http_request_body_len","http_response_body_len",
           "http_status_code"]
    SKEW = {"duration","src_bytes","dst_bytes","missed_bytes","src_pkts","src_ip_bytes",
            "dst_pkts","dst_ip_bytes","http_request_body_len","http_response_body_len"}

    print("=" * 66)
    print("EXPORT LUT INTEGER-ONLY MULTICLASS (Chebyshev single-layer)")
    print("=" * 66)

    df = pd.read_csv(args.csv)
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)
    feats = [c for c in NUM if c in df.columns]
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float64)
    le = LabelEncoder().fit(df["type"]); y = le.transform(df["type"]); C = len(le.classes_)
    mitm = list(le.classes_).index("mitm")
    mi = mutual_info_classif(X, y, random_state=42)
    order = np.argsort(mi)[::-1][:K]; feats_k = [feats[i] for i in order]; Xk = X[:, order]
    Xtr, Xte, ytr, yte = train_test_split(Xk, y, test_size=0.2, random_state=42, stratify=y)

    # preprocessing log1p + quantile (fit su train)
    def prep(a, b):
        a = a.copy(); b = b.copy()
        for j, n in enumerate(feats_k):
            if n in SKEW:
                a[:, j] = np.log1p(np.clip(a[:, j], 0, None))
                b[:, j] = np.log1p(np.clip(b[:, j], 0, None))
        qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                                 random_state=42).fit(a)
        return np.clip(qt.transform(a), -CLIP, CLIP), np.clip(qt.transform(b), -CLIP, CLIP)
    Xtr_s, Xte_s = prep(Xtr, Xte)

    print(f"\n[1] Addestro KAN multiclass [{K}->{C}] deg={args.degree}...")
    kan = ChebyshevKANMulticlass(in_dim=K, n_classes=C, degree=args.degree,
                                 x_min=-CLIP, x_max=CLIP)
    kan.fit(Xtr_s, ytr, epochs=args.epochs, lr=0.3, verbose=False)
    pred_float = kan.predict(Xte_s)
    f1_float = f1_score(yte, pred_float, average="macro", zero_division=0)
    print(f"    macro-F1 float = {f1_float:.4f}")

    print(f"\n[2] Costruisco LUT INT16 (E={K*C} edge, S=2^{FP_BITS})...")
    art, table_int = build_int_table_mc(kan, L=args.L)
    bytes_int = table_int.nbytes
    print(f"    edge={table_int.shape[0]}  K={table_int.shape[1]}  L={args.L}")
    print(f"    memoria tabella INT16: {bytes_int} byte ({bytes_int/1024:.1f} KB) — ESP32-only")

    print(f"\n[3] Forward INTERO + argmax (zero float)...")
    pred_int, Z = int_forward_argmax(art, table_int, Xte_s, K, C, args.L)
    f1_int = f1_score(yte, pred_int, average="macro", zero_division=0)

    print(f"\n[4] VERIFICA: argmax intero vs argmax float del modello")
    agree = (pred_int == pred_float).mean()
    print(f"    predizioni identiche: {agree*100:.3f}%")
    print(f"    macro-F1 float={f1_float:.4f}  int={f1_int:.4f}  ΔF1={f1_int-f1_float:+.4f}")
    print(f"    MITM F1 int: {f1_score(yte,pred_int,average=None,zero_division=0)[mitm]:.4f}")

    print(f"\n[5] Genero header C multiclass...")
    out = write_int_c_header_mc(art, table_int, args.out, K, C, args.L)
    print(f"    scritto {out}  ({Path(out).stat().st_size/1024:.0f} KB)")

    print("\n" + "=" * 66)
    verdict = "OK" if agree >= 0.99 else "DA RIVEDERE"
    print(f"ESITO: {verdict} — multiclass intero, predizioni al {agree*100:.2f}%")
    print("=" * 66)


if __name__ == "__main__":
    main()
