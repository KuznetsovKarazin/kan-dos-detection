#!/usr/bin/env python3
"""
export_ml_int.py — export del multi-layer quantizzato per il firmware C
=======================================================================
Genera tutto il necessario per main_kan_ml_wokwi.cpp:
  - kan_ml_layer1.h : LUT int16 layer1 (in_dim*hidden edge), dominio [-1,1]
  - kan_ml_tanh.h   : tabella tanh intera (pre-attivazione scala S -> scala S2)
  - kan_ml_layer2.h : LUT int16 layer2 (hidden*C edge), dominio [-1,1] scala S2
  - test_vectors_ml_q16.h : 40 input pre-quantizzati Q16.16 + label
  - stampa i logit interi di riferimento dei primi vettori (verifica C)

Catena (zero float): input pre-quant Q16.16 -> LUT l1 -> accumulo int ->
tanh-LUT -> hidden in scala S2 -> LUT l2 (indicizzata in scala S2) ->
accumulo int per classe -> argmax.
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for p in [_REPO, _REPO/"src", _REPO/"preprocessing", _REPO/"scripts"]:
    sys.path.insert(0, str(p))

from ml_stage2_layer1_tanh import cheb_phi_1d, build_edge_lut_int, eval_edge_int, eval_edge_int_domain

CLIP=3.5; K=10; degree=8; hidden=16
KSEG=8; L=64; S=512; S2=10000
NUM=['src_port','dst_port','duration','src_bytes','dst_bytes','missed_bytes','src_pkts','src_ip_bytes','dst_pkts','dst_ip_bytes','dns_qclass','dns_qtype','dns_rcode','http_request_body_len','http_response_body_len','http_status_code']
SKEW={'duration','src_bytes','dst_bytes','missed_bytes','src_pkts','src_ip_bytes','dst_pkts','dst_ip_bytes','http_request_body_len','http_response_body_len'}


def write_header_lut(path, name, macro, tables, extra_lines=None):
    """tables: array (E, KSEG, L) int. Scrive header con LUT appiattita."""
    E = tables.shape[0]
    flat = tables.reshape(E, KSEG*L)
    lines = [f"// {name} — {datetime.now().isoformat(timespec='seconds')}",
             "#pragma once", "#include <stdint.h>",
             "#ifdef __AVR__\n#include <avr/pgmspace.h>\n#else\n#define PROGMEM\n#endif"]
    if extra_lines: lines += extra_lines
    lines.append(f"static const int16_t {macro}[{E}][{KSEG*L}] PROGMEM = {{")
    for e in range(E):
        lines.append("  {" + ",".join(str(int(v)) for v in flat[e]) + "},")
    lines.append("};")
    Path(path).write_text("\n".join(lines))


def main():
    import pandas as pd
    from sklearn.preprocessing import QuantileTransformer, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.metrics import f1_score
    from kan_torch import train_kan_torch
    from kan_multilayer_numpy import from_torch

    df=pd.read_csv('train_test_network.csv').sample(60000,random_state=42).reset_index(drop=True)
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

    print("="*64); print("EXPORT MULTI-LAYER per firmware C"); print("="*64)
    m=train_kan_torch(Xtr_s.astype(np.float32),ytr,K,C,hidden=hidden,degree=degree,epochs=400,lr=0.02,class_weights=cw,verbose=False)
    rep=from_torch(m); c1,c2=rep.coeffs1,rep.coeffs2

    # --- LUT layer1 (dominio [-1,1]) ---
    tab1=np.zeros((K*hidden,KSEG,L),dtype=np.int64); e=0
    for i in range(K):
        for j in range(hidden):
            tbl,_=build_edge_lut_int(lambda x,ci=c1[i,j]: cheb_phi_1d(ci,x,degree),-1.0,1.0,KSEG,L,S)
            tab1[e]=tbl; e+=1
    # --- LUT layer2 (dominio [-1,1]) ---
    tab2=np.zeros((hidden*C,KSEG,L),dtype=np.int64); e=0
    for j in range(hidden):
        for c in range(C):
            tbl,_=build_edge_lut_int(lambda x,cj=c2[j,c]: cheb_phi_1d(cj,x,degree),-1.0,1.0,KSEG,L,S)
            tab2[e]=tbl; e+=1

    # --- forward intero di riferimento per stimare Hmax e i logit ---
    N=Xte_s.shape[0]
    h_pre=np.zeros((N,hidden),dtype=np.int64); e=0
    for i in range(K):
        xi=np.clip(Xte_s[:,i],-1.0,1.0)
        for j in range(hidden):
            h_pre[:,j]+=eval_edge_int(tab1[e],None,xi,KSEG,L,-1.0,1.0); e+=1
    Hmax=int(np.percentile(np.abs(h_pre),99.9))+1; TL=2048
    grid=np.linspace(-Hmax,Hmax,TL); tanh_tab=np.round(S2*np.tanh(grid/S)).astype(np.int64)
    def tanh_lut(v):
        idx=np.clip(((v+Hmax)/(2*Hmax)*(TL-1)).astype(int),0,TL-1); return tanh_tab[idx]
    h_post=tanh_lut(h_pre)
    Z=np.zeros((N,C),dtype=np.int64); e=0
    for j in range(hidden):
        hj=np.clip(h_post[:,j],-S2,S2)
        for c in range(C):
            Z[:,c]+=eval_edge_int_domain(tab2[e],hj,S2,KSEG,L); e+=1
    pred=np.argmax(Z,axis=1)
    print(f"macro-F1 int (riferimento): {f1_score(yte,pred,average='macro',zero_division=0):.4f}")

    # --- header LUT ---
    knots_min,knots_max=-1.0,1.0
    write_header_lut('kan_ml_layer1.h','kan_ml_layer1','KANML_L1',tab1,
        extra_lines=[f"#define KANML_INDIM {K}",f"#define KANML_HIDDEN {hidden}",
                     f"#define KANML_C {C}",f"#define KANML_KSEG {KSEG}",
                     f"#define KANML_L {L}",f"#define KANML_S {S}",f"#define KANML_S2 {S2}",
                     f"#define KANML_HMAX {Hmax}",f"#define KANML_TL {TL}"])
    write_header_lut('kan_ml_layer2.h','kan_ml_layer2','KANML_L2',tab2)
    # tabella tanh
    with open('kan_ml_tanh.h','w') as f:
        f.write("// kan_ml_tanh.h — tabella tanh intera\n#pragma once\n#include <stdint.h>\n")
        f.write("#ifdef __AVR__\n#include <avr/pgmspace.h>\n#else\n#define PROGMEM\n#endif\n")
        f.write(f"static const int16_t KANML_TANH[{TL}] PROGMEM = {{")
        f.write(",".join(str(int(v)) for v in tanh_tab))
        f.write("};\n")

    # --- test vectors pre-quantizzati Q16.16 (dominio [-1,1] del layer1) ---
    # il layer1 lavora su input clampato a [-1,1]; il firmware riceve x in Q16.16
    # rappresentante valore in [-1,1] (come nel binario fully-int adattato a [-1,1])
    FP=1<<16
    rng=np.random.RandomState(42); sel=[]
    for c in range(C):
        idxs=np.where(yte==c)[0]; sel.extend(rng.choice(idxs,min(4,len(idxs)),replace=False))
    sel=np.array(sel); rng.shuffle(sel)
    V=np.clip(Xte_s[sel],-1.0,1.0); lab=yte[sel]
    Xq=np.round((V+1.0)*FP).astype(np.int64)   # off da -1: (x+1)*2^16, in [0, 2*2^16]
    with open('test_vectors_ml_q16.h','w') as f:
        f.write("// 40 input multi-layer pre-quantizzati: (x+1)*2^16, x in [-1,1]\n#pragma once\n")
        f.write(f"#define N_TEST {len(sel)}\nstatic const int32_t TEST_XQ[N_TEST][{K}] = {{\n")
        for r in Xq: f.write('  {'+','.join(str(int(v)) for v in r)+'},\n')
        f.write("};\n")
        f.write(f"static const int TEST_LABEL[N_TEST] = {{{','.join(str(int(l)) for l in lab)}}};\n")

    # logit di riferimento primi 3 vettori
    print("\nLogit interi di riferimento (primi 3 vettori):")
    for t in range(3):
        zi=Z[sel[t]]
        print(f"  vec{t}: argmax={int(np.argmax(zi))} label={int(lab[t])} logits={zi.tolist()}")
    print(f"\nAccuratezza sui 40 vettori: {(pred[sel]==lab).mean()*100:.1f}%")
    print(f"\nHeader generati: kan_ml_layer1.h, kan_ml_layer2.h, kan_ml_tanh.h, test_vectors_ml_q16.h")
    print(f"memoria: layer1 {K*hidden*KSEG*L*2/1024:.0f}KB + layer2 {hidden*C*KSEG*L*2/1024:.0f}KB + tanh {TL*2/1024:.0f}KB")


if __name__=="__main__":
    main()
