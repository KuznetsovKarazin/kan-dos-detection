#!/usr/bin/env python3
"""
ml_stage3_full.py — quantizzazione stadio 3: secondo strato + forward completo
==============================================================================
Completa la quantizzazione del multi-layer: dopo layer1+tanh (stadio 2),
quantizza il SECONDO strato [hidden->C] e fa l'argmax finale. Misura
l'accuratezza del forward INTERAMENTE quantizzato vs il float.

E' il momento della verita': qui si vede se i ~0.92 del multi-layer
sopravvivono alla quantizzazione a due strati o se la propagazione degli
errori li degrada.

Catena intera completa:
  x -> [LUT layer1] -> h_pre(int) -> [LUT tanh] -> h_post(int, scala S2)
    -> [LUT layer2, dominio [-1,1]] -> logit_c(int) -> argmax
"""

import sys
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for p in [_REPO, _REPO/"src", _REPO/"preprocessing"]:
    sys.path.insert(0, str(p))

from ml_stage2_layer1_tanh import cheb_phi_1d, build_edge_lut_int, eval_edge_int, eval_edge_int_domain


def main():
    import pandas as pd
    from sklearn.preprocessing import QuantileTransformer, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.metrics import f1_score
    from kan_torch import train_kan_torch
    from kan_multilayer_numpy import from_torch

    NUM=['src_port','dst_port','duration','src_bytes','dst_bytes','missed_bytes','src_pkts','src_ip_bytes','dst_pkts','dst_ip_bytes','dns_qclass','dns_qtype','dns_rcode','http_request_body_len','http_response_body_len','http_status_code']
    SKEW={'duration','src_bytes','dst_bytes','missed_bytes','src_pkts','src_ip_bytes','dst_pkts','dst_ip_bytes','http_request_body_len','http_response_body_len'}
    CLIP=3.5; K=10; degree=8; hidden=16
    KSEG=8; L=64; S=512; S2=10000

    df=pd.read_csv('train_test_network.csv').sample(50000,random_state=42).reset_index(drop=True)
    feats=[c for c in NUM if c in df.columns]
    X=df[feats].apply(pd.to_numeric,errors='coerce').fillna(0).to_numpy(np.float64)
    le=LabelEncoder().fit(df['type']);y=le.transform(df['type']);C=len(le.classes_)
    mitm=list(le.classes_).index('mitm')
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

    print("="*64); print("STADIO 3 — quantizzazione completa multi-layer"); print("="*64)
    m=train_kan_torch(Xtr_s.astype(np.float32),ytr,K,C,hidden=hidden,degree=degree,epochs=400,lr=0.02,class_weights=cw,verbose=False)
    rep=from_torch(m)
    c1,c2=rep.coeffs1,rep.coeffs2
    pred_float=rep.predict(Xte_s)
    f1_float=f1_score(yte,pred_float,average='macro',zero_division=0)

    N=Xte_s.shape[0]
    # ---- layer1 quantizzato (dominio [-1,1]) ----
    luts1=[]
    for i in range(K):
        for j in range(hidden):
            luts1.append(build_edge_lut_int(lambda x,ci=c1[i,j]: cheb_phi_1d(ci,x,degree),-1.0,1.0,KSEG,L,S))
    h_pre=np.zeros((N,hidden),dtype=np.int64); e=0
    for i in range(K):
        xi=np.clip(Xte_s[:,i],-1.0,1.0)
        for j in range(hidden):
            tbl,seg=luts1[e]; e+=1
            h_pre[:,j]+=eval_edge_int(tbl,seg,xi,KSEG,L,-1.0,1.0)
    # ---- tanh-as-LUT ----
    Hmax=int(np.percentile(np.abs(h_pre),99.9))+1; TL=2048
    grid=np.linspace(-Hmax,Hmax,TL); tanh_tab=np.round(S2*np.tanh(grid/S)).astype(np.int64)
    def tanh_lut(v):
        idx=np.clip(((v+Hmax)/(2*Hmax)*(TL-1)).astype(int),0,TL-1); return tanh_tab[idx]
    h_post=tanh_lut(h_pre)            # scala S2, ~[-S2,S2] = [-1,1]*S2

    # ---- layer2 quantizzato: input = h_post/S2 in [-1,1] ----
    # tabulo edge layer2 su [-1,1]; input intero -> normalizzo a [-1,1] per indicizzare
    luts2=[]
    for j in range(hidden):
        for c in range(C):
            luts2.append(build_edge_lut_int(lambda x,cj=c2[j,c]: cheb_phi_1d(cj,x,degree),-1.0,1.0,KSEG,L,S))
    Z=np.zeros((N,C),dtype=np.int64)
    e=0
    for j in range(hidden):
        hj_int=np.clip(h_post[:,j],-S2,S2)    # intero in scala S2 (zero-float)
        for c in range(C):
            tbl,seg=luts2[e]; e+=1
            Z[:,c]+=eval_edge_int_domain(tbl,hj_int,S2,KSEG,L)
    pred_int=np.argmax(Z,axis=1)
    f1_int=f1_score(yte,pred_int,average='macro',zero_division=0)

    agree=(pred_int==pred_float).mean()
    print(f"edge totali: layer1={K*hidden} + layer2={hidden*C} = {K*hidden+hidden*C}")
    print(f"  (vs single-layer: {K*C} edge)")
    print(f"predizioni intero vs float: {agree*100:.2f}%")
    print(f"macro-F1 float = {f1_float:.4f}")
    print(f"macro-F1 INT   = {f1_int:.4f}   (ΔF1 = {f1_int-f1_float:+.4f})")
    print(f"MITM F1 int    = {f1_score(yte,pred_int,average=None,zero_division=0)[mitm]:.4f}")
    print(f"memoria LUT: {(K*hidden+hidden*C)*KSEG*L*2/1024:.0f} KB int16")
    print("="*64)
    print(f"CONFRONTO: single-layer int era 0.858. multi-layer int qui = {f1_int:.4f}")
    print("="*64)


if __name__=="__main__":
    main()
