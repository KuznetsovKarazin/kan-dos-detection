#!/usr/bin/env python3
"""
export_lut_int.py — export LUT INTEGER-ONLY (feedback Prof. Kuznetsov)
=====================================================================
Versione a interi puri della catena di export, secondo l'indicazione del
Prof. Kuznetsov: eliminare completamente il float dall'inferenza, non solo
la valutazione dei polinomi.

IDEA (resa possibile dalla CLASSIFICAZIONE, non dalla regressione):
  - per la decisione binaria serve solo  segno(logit), cioe' z >= 0 ?
  - quindi NON servono sigmoid, ne' i valori float esatti
  - si pre-scala in export: ogni valore in tabella e' gia' round(S * phi),
    dove phi = ymin + scale*q  e  S = 2^FP_BITS (fattore fisso)
  - a runtime: lookup int16 -> interpolazione intera -> accumulo int32
    su tutti gli edge -> confronto con soglia intera (= 0). Zero float.

Confronto con la versione float (export_lut.py):
  float:  y = ymin + scale*q ; sum ; sigmoid ; >= 0.5     (≥1 float/edge)
  int  :  v = LUT_int16 ; sum in int32 ; >= 0             (0 float)

Genera kan_ids_layer_int.h e verifica che la decisione intera coincida
con quella float del modello originale.
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
from src.quant.lut_builder import build_lut_for_edges, LUTArtifact   # noqa: E402

from export_lut import ChebyshevEdge, kan_to_edges   # riusa l'adapter Chebyshev

# fattore di pre-scaling: phi tipicamente in [-qualche unita']; S=2^9 da
# risoluzione ampia restando comodo in int16 anche dopo l'interpolazione.
FP_BITS = 9
S = 1 << FP_BITS


def build_int_table(kan, L=64, num_knots=9):
    """Costruisce la LUT float (riuso build_lut_for_edges di Kuznetsov) e poi
    la converte in tabella int16 PRE-SCALATA: val[e,k,r] = round(S * phi)."""
    edges = kan_to_edges(kan, num_knots=num_knots)
    art = build_lut_for_edges(
        edges=edges, L=L, interp="linear", oob_behavior="clip",
        boundary_mode="half_open", y_range_method="minmax",
        lower_pct=0.1, upper_pct=99.9, dtype="uint8", scheme="asymmetric",
        qmin=0, qmax=255, meta_dtype="float16", value_representation="phi")

    E, K = art.q_table.shape[0], art.q_table.shape[1]
    # phi[e,k,r] = ymin[e,k] + scale[e,k] * q[e,k,r]
    q = art.q_table.astype(np.float64)                  # (E,K,L)
    scale = art.scale.astype(np.float64)[:, :, None]    # (E,K,1)
    ymin = art.y_min.astype(np.float64)[:, :, None]     # (E,K,1)
    phi = ymin + scale * q                              # (E,K,L) float
    table_int = np.round(S * phi).astype(np.int16)      # PRE-SCALATA
    return art, table_int


def int_forward_decision(art, table_int, X, L, thr_int=0):
    """Forward INTERO puro (replica esatta del runtime C che scriveremo).
    Ritorna le decisioni binarie (z_int >= thr_int)."""
    knots = art.knots
    x_min, x_max = float(knots[0]), float(knots[-1])
    K = int(knots.size - 1)
    dx = (x_max - x_min) / K
    N, E = X.shape[0], table_int.shape[0]
    z = np.zeros(N, dtype=np.int64)
    for i in range(E):
        x = X[:, i].astype(np.float64)
        xc = np.clip(x, x_min, x_max - 1e-6)
        t = (xc - x_min) / dx
        k = np.floor(t).astype(int); k = np.clip(k, 0, K - 1)
        u = (xc - (x_min + k * dx)) / dx
        u = np.clip(u, 0.0, 1.0)
        pos = u * (L - 1)
        r0 = np.floor(pos).astype(int); r0 = np.clip(r0, 0, L - 1)
        r1 = np.clip(r0 + 1, 0, L - 1)
        frac256 = np.clip(((pos - r0) * 256).astype(np.int64), 0, 256)
        q0 = table_int[i][k, r0].astype(np.int64)
        q1 = table_int[i][k, r1].astype(np.int64)
        v = q0 + ((frac256 * (q1 - q0)) >> 8)            # interp intera
        z += v
    return (z >= thr_int).astype(int), z


def write_int_c_header(art, table_int, path, L, name="kan_ids_layer_int"):
    E, K = table_int.shape[0], table_int.shape[1]
    knots = art.knots
    lines = [
        f"// {name}.h — KAN-IDS LUT INTEGER-ONLY (feedback Prof. Kuznetsov)",
        f"// {datetime.now().isoformat(timespec='seconds')}",
        f"// E={E} edge, K={K} segmenti, L={L} campioni/segmento, FP_BITS={FP_BITS}",
        "// Inferenza interamente INTERA: lookup int16 + interp intera +",
        "// accumulo int32 + confronto con soglia. Zero float a runtime.",
        "// Decisione binaria: attacco se sum_edge(v) >= 0",
        "#pragma once",
        "#include <stdint.h>",
        "#ifdef __AVR__\n#include <avr/pgmspace.h>\n#else\n#define PROGMEM\n#endif",
        f"#define KANI_E {E}",
        f"#define KANI_K {K}",
        f"#define KANI_L {L}",
        f"#define KANI_FP_BITS {FP_BITS}",
        f"static const float KANI_XMIN = {float(knots[0]):.8f}f;",
        f"static const float KANI_XMAX = {float(knots[-1]):.8f}f;",
        "// indicizzazione: una sola conversione int per input (Q16.16), poi tutto intero",
    ]
    flat = table_int.reshape(E, K * L)
    lines.append(f"static const int16_t KANI_TABLE[{E}][{K*L}] PROGMEM = {{")
    for e in range(E):
        lines.append("  {" + ",".join(str(int(v)) for v in flat[e]) + "},")
    lines.append("};")
    Path(path).write_text("\n".join(lines))
    return path


def main():
    import argparse
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    import section_310_unified_feature_engineering as fe
    from kan_chebyshev import ChebyshevKANBinary

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="train_test_network.csv")
    ap.add_argument("--out", default="kan_ids_layer_int.h")
    ap.add_argument("--degree", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--L", type=int, default=64)
    args = ap.parse_args()
    CLIP = 3.5

    print("=" * 66)
    print("EXPORT LUT INTEGER-ONLY (feedback Prof. Kuznetsov)")
    print("=" * 66)

    df = pd.read_csv(args.csv)
    X = fe.build_unified_features_ton(df)[fe.UNIFIED_NUMERIC_FEATURES].to_numpy(np.float64)
    y = df["label"].astype(int).to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr = np.clip(sc.transform(Xtr), -CLIP, CLIP)
    Xte = np.clip(sc.transform(Xte), -CLIP, CLIP)

    print(f"\n[1] Addestro KAN [{X.shape[1]}->1] deg={args.degree}...")
    kan = ChebyshevKANBinary(in_dim=X.shape[1], degree=args.degree, x_min=-CLIP, x_max=CLIP)
    kan.fit(Xtr, ytr, epochs=args.epochs, lr=0.3, verbose=False)
    dec_float = kan.predict(Xte)
    f1_float = f1_score(yte, dec_float)
    print(f"    F1 float = {f1_float:.4f}")

    print(f"\n[2] Costruisco LUT INT16 pre-scalata (S=2^{FP_BITS}={S})...")
    art, table_int = build_int_table(kan, L=args.L)
    E, K = table_int.shape[0], table_int.shape[1]
    bytes_int = table_int.nbytes
    print(f"    edge={E} K={K} L={args.L}")
    print(f"    memoria tabella INT16: {bytes_int} byte ({bytes_int/1024:.2f} KB)")
    print(f"    (float era: q_table uint8 + scale + ymin separati)")

    print(f"\n[3] Forward INTERO puro + decisione (soglia intera = 0)...")
    dec_int, z_int = int_forward_decision(art, table_int, Xte, args.L, thr_int=0)
    f1_int = f1_score(yte, dec_int)

    print(f"\n[4] VERIFICA: decisione INTERA vs decisione FLOAT del modello")
    agree = (dec_int == dec_float).mean()
    n_diff = int((dec_int != dec_float).sum())
    print(f"    decisioni identiche : {agree*100:.3f}%  ({len(yte)-n_diff}/{len(yte)})")
    print(f"    F1 float={f1_float:.4f}  |  F1 int={f1_int:.4f}  |  ΔF1={f1_int-f1_float:+.4f}")

    print(f"\n[5] Genero header C integer-only...")
    out = write_int_c_header(art, table_int, args.out, args.L)
    print(f"    scritto {out}  ({Path(out).stat().st_size/1024:.1f} KB)")

    print("\n" + "=" * 66)
    verdict = "OK" if agree >= 0.99 else "DA RIVEDERE (perdita di precisione int)"
    print(f"ESITO: {verdict} — inferenza interamente intera, decisioni al {agree*100:.2f}%")
    print("=" * 66)


if __name__ == "__main__":
    main()
