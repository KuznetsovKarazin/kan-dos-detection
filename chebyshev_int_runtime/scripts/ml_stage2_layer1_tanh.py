#!/usr/bin/env python3
"""
ml_stage2_layer1_tanh.py — quantizzazione stadio 2: primo strato + tanh
=======================================================================
Quantizza il PRIMO strato del multi-layer (edge Chebyshev [in_dim->hidden])
in LUT int16 pre-scalate, e il tanh in una LUT intera. Verifica quanto i
valori hidden quantizzati (post-tanh) si discostano da quelli float.

Strategia tanh-as-LUT:
  - il pre-attivazione hidden h_pre = sum_i phi1_{i,j}(x_i) (in scala intera S)
  - tanh applicato via tabella: TANH_LUT[clip(h_pre_scaled)] -> valore in [-1,1]
    rappresentato come intero in scala S2 (per indicizzare il layer2)
  - cosi' il tanh e' un lookup, zero float

Questo e' lo stadio 2: si ferma all'hidden post-tanh. Lo stadio 3 quantizzera'
il secondo strato. Qui misuriamo solo l'errore introdotto fin qui.
"""

import sys
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for p in [_REPO, _REPO/"src", _REPO/"preprocessing"]:
    sys.path.insert(0, str(p))


def cheb_phi_1d(coef, x, degree):
    """phi(x) con x gia' in [-1,1] (dominio Chebyshev), coef (degree+1,)."""
    xn = np.clip(x, -1.0, 1.0)
    T = np.empty((len(xn), degree+1))
    T[:,0]=1.0
    if degree>=1: T[:,1]=xn
    for n in range(2,degree+1): T[:,n]=2*xn*T[:,n-1]-T[:,n-2]
    return T @ coef


# ---- quantizzazione edge in LUT (riuso lo schema integer del single-layer) ----
def build_edge_lut_int(phi_func, x_min, x_max, K, L, S):
    """Tabula phi su K segmenti x L campioni, valori int16 = round(S*phi).
    Ritorna table (K,L) int e i bordi segmento. Dominio [x_min,x_max]."""
    seg = np.linspace(x_min, x_max, K+1)
    tbl = np.zeros((K, L), dtype=np.int64)
    for s in range(K):
        xs = np.linspace(seg[s], seg[s+1], L)
        tbl[s] = np.round(S * phi_func(xs)).astype(np.int64)
    return tbl, seg


def eval_edge_int(tbl, seg, x, K, L, x_min, x_max):
    """Valuta l'edge quantizzato (interp intera) su array x. Ritorna int (scala S)."""
    dx = (x_max - x_min)/K
    xc = np.clip(x, x_min, x_max-1e-6)
    k = np.clip(np.floor((xc-x_min)/dx).astype(int),0,K-1)
    u = np.clip((xc-(x_min+k*dx))/dx,0,1)
    pos = u*(L-1); r0=np.clip(np.floor(pos).astype(int),0,L-1); r1=np.clip(r0+1,0,L-1)
    fr=np.clip(((pos-r0)*256).astype(np.int64),0,256)
    q0=tbl[k,r0]; q1=tbl[k,r1]
    return q0 + ((fr*(q1-q0))>>8)


def eval_edge_int_domain(tbl, x_scaled, x_scale, K, L):
    """Variante ZERO-FLOAT per il layer2: l'input arriva gia' come intero
    in scala x_scale, rappresentando un valore nel dominio [-1, 1].
    Cioe' x_scaled = round(x_norm * x_scale), con x_norm in [-1,1].
    Tutta l'indicizzazione (segmento, campione, frazione) e' intera.

    Mappa il dominio [-x_scale, +x_scale] (= [-1,1]) su K segmenti x L campioni.
    """
    full = 2 * x_scale                       # ampiezza totale del dominio, intera
    # posizione nel dominio: off = x_scaled + x_scale  in [0, 2*x_scale]
    off = np.clip(x_scaled + x_scale, 0, full - 1)
    # segmento k: off * K / full  (intero)
    k = np.clip((off * K) // full, 0, K - 1).astype(np.int64)
    # posizione dentro il segmento, in 256-esimi: resto riscalato
    seg_w = full // K if full % K == 0 else full / K   # larghezza segmento
    # uso aritmetica intera: pos_in_seg = off - k*seg_w, normalizzato a [0,256)
    # per restare interi usiamo full e K direttamente:
    #   frazione_in_segmento = (off*K - k*full) / full   in [0,1)
    num = off * K - k * full                  # in [0, full)
    # campione r0 e frazione a 256: u = num/full ; pos = u*(L-1)
    pos256 = (num * (L - 1) * 256) // full     # = u*(L-1)*256, intero
    r0 = np.clip((pos256 // 256), 0, L - 1).astype(np.int64)
    r1 = np.clip(r0 + 1, 0, L - 1)
    fr = np.clip(pos256 - r0 * 256, 0, 256)
    q0 = tbl[k, r0]; q1 = tbl[k, r1]
    return q0 + ((fr * (q1 - q0)) >> 8)


def main():
    import pandas as pd
    from sklearn.preprocessing import QuantileTransformer, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.feature_selection import mutual_info_classif
    from kan_torch import train_kan_torch
    from kan_multilayer_numpy import from_torch

    NUM=['src_port','dst_port','duration','src_bytes','dst_bytes','missed_bytes','src_pkts','src_ip_bytes','dst_pkts','dst_ip_bytes','dns_qclass','dns_qtype','dns_rcode','http_request_body_len','http_response_body_len','http_status_code']
    SKEW={'duration','src_bytes','dst_bytes','missed_bytes','src_pkts','src_ip_bytes','dst_pkts','dst_ip_bytes','http_request_body_len','http_response_body_len'}
    CLIP=3.5; K=10; degree=8; hidden=16
    KSEG=8; L=64; S=512   # scala intera per gli edge (come binario)

    df=pd.read_csv('train_test_network.csv').sample(50000,random_state=42).reset_index(drop=True)
    feats=[c for c in NUM if c in df.columns]
    X=df[feats].apply(pd.to_numeric,errors='coerce').fillna(0).to_numpy(np.float64)
    le=LabelEncoder().fit(df['type']);y=le.transform(df['type']);C=len(le.classes_)
    mi=mutual_info_classif(X,y,random_state=42);order=np.argsort(mi)[::-1][:K];feats_k=[feats[i] for i in order];Xk=X[:,order]
    Xtr,Xte,ytr,yte=train_test_split(Xk,y,test_size=0.2,random_state=42,stratify=y)
    def prep(a,b):
        a=a.copy();b=b.copy()
        for j,n in enumerate(feats_k):
            if n in SKEW: a[:,j]=np.log1p(np.clip(a[:,j],0,None));b[:,j]=np.log1p(np.clip(b[:,j],0,None))
        qt=QuantileTransformer(output_distribution='normal',n_quantiles=1000,random_state=42).fit(a)
        return np.clip(qt.transform(a),-CLIP,CLIP),np.clip(qt.transform(b),-CLIP,CLIP)
    Xtr_s,Xte_s=prep(Xtr,Xte)
    counts=np.bincount(ytr,minlength=C);cw=(len(ytr)/(C*np.maximum(counts,1)))

    print("="*64); print("STADIO 2 — quantizzazione layer1 + tanh"); print("="*64)
    m=train_kan_torch(Xtr_s.astype(np.float32),ytr,K,C,hidden=hidden,degree=degree,epochs=400,lr=0.02,class_weights=cw,verbose=False)
    rep=from_torch(m)
    c1=rep.coeffs1   # (in_dim, hidden, deg+1)

    # hidden FLOAT di riferimento (post-tanh): input clampato a [-1,1] come nel forward
    Xc = np.clip(Xte_s, -1.0, 1.0)
    h_pre_float = np.zeros((Xte_s.shape[0], hidden))
    for i in range(K):
        for j in range(hidden):
            h_pre_float[:,j] += cheb_phi_1d(c1[i,j], Xc[:,i], degree)
    h_post_float = np.tanh(h_pre_float)

    # --- quantizzazione layer1: LUT per ogni edge (i,j), dominio [-1,1] ---
    # input del layer1 vive in [-1,1] (clamp), quindi tabulo su [-1,1]
    luts=[]
    for i in range(K):
        for j in range(hidden):
            tbl,seg=build_edge_lut_int(lambda x,ci=c1[i,j]: cheb_phi_1d(ci,x,degree),-1.0,1.0,KSEG,L,S)
            luts.append((tbl,seg))
    # forward intero layer1
    h_pre_int=np.zeros((Xte_s.shape[0],hidden),dtype=np.int64)
    e=0
    for i in range(K):
        xi=np.clip(Xte_s[:,i],-1.0,1.0)
        for j in range(hidden):
            tbl,seg=luts[e]; e+=1
            h_pre_int[:,j]+=eval_edge_int(tbl,seg,xi,KSEG,L,-1.0,1.0)
    # h_pre_int e' in scala S. tanh-as-LUT: costruisco tabella tanh
    # dominio pre-attivazione: copro [-H,H] con H scelto sui dati
    Hmax=int(np.percentile(np.abs(h_pre_int),99.9))+1
    TL=2048  # campioni tabella tanh
    grid=np.linspace(-Hmax,Hmax,TL)
    S2=10000  # scala per rappresentare tanh in [-1,1] come intero
    tanh_tab=np.round(S2*np.tanh(grid/S)).astype(np.int64)
    def tanh_lut(v):
        idx=np.clip(((v+Hmax)/(2*Hmax)*(TL-1)).astype(int),0,TL-1)
        return tanh_tab[idx]
    h_post_int=tanh_lut(h_pre_int)   # in scala S2, ~ [-S2,S2]

    # confronto: hidden post-tanh float vs quantizzato (riportato a float)
    h_post_int_f = h_post_int/S2
    err=np.abs(h_post_int_f-h_post_float)
    print(f"hidden={hidden}, edge layer1={K*hidden}, KSEG={KSEG}, L={L}, S={S}")
    print(f"errore hidden post-tanh (quant vs float):")
    print(f"  max  = {err.max():.4f}")
    print(f"  mean = {err.mean():.4f}")
    print(f"  Hmax (range pre-attivazione) = {Hmax} (scala S), tanh tab {TL} punti, S2={S2}")
    # quanti hidden cambiano "segno/zona" significativamente
    print(f"  frazione hidden con errore >0.05: {(err>0.05).mean()*100:.2f}%")
    print("="*64)
    print("Se l'errore medio e' piccolo (<0.02), lo stadio 2 e' ok -> stadio 3 (layer2)")
    print("="*64)


if __name__=="__main__":
    main()
