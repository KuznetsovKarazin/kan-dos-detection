#!/usr/bin/env python3
"""
kan_multilayer_numpy.py — replica NumPy del forward multi-layer (stadio 1)
==========================================================================
Replica esatta, in NumPy puro, del forward di KANTorch (kan_torch.py):
  layer1 [in_dim -> hidden] Chebyshev  ->  tanh  ->  layer2 [hidden -> C] Chebyshev
  predizione = argmax dei logit del layer2

Serve a verificare la composizione dei due strati PRIMA di quantizzare.
Estrae i pesi da un modello PyTorch addestrato e ricostruisce il forward
senza torch. Se le predizioni coincidono al 100%, la replica e' corretta e
si puo' procedere alla quantizzazione stadio per stadio.

Punti chiave per la futura quantizzazione (gia' evidenziati qui):
  - layer1 riceve input nel dominio del preprocessing (clip [-CLIP, CLIP])
  - tanh mappa in [-1, 1]
  - layer2 riceve quindi input gia' in [-1, 1] (dominio Chebyshev naturale)
"""

import numpy as np


def _cheb_basis(x, degree):
    """x: (N, D) in dominio -> T: (N, D, degree+1) con normalizzazione
    al dominio [-1,1] gestita FUORI (qui x e' gia' nel range giusto)."""
    x = np.clip(x, -1.0, 1.0)
    T = [np.ones_like(x), x]
    for n in range(2, degree + 1):
        T.append(2.0 * x * T[-1] - T[-2])
    return np.stack(T, axis=-1)   # (N, D, degree+1)


def _layer_forward(x_norm, coeffs, degree):
    """x_norm: (N, in_dim) gia' normalizzato in [-1,1].
    coeffs: (in_dim, out_dim, degree+1). Ritorna (N, out_dim)."""
    T = _cheb_basis(x_norm, degree)               # (N, in_dim, deg+1)
    return np.einsum("nid,ijd->nj", T, coeffs)    # (N, out_dim)


class MultiLayerKANNumpy:
    """Forward multi-layer in NumPy. I pesi vengono da un KANTorch addestrato.

    NOTA sul dominio: nel KANTorch il layer1 riceve x gia' clippato in
    [-CLIP,CLIP] ma la base Chebyshev di PyTorch lo richiede in [-1,1].
    Nel codice PyTorch x viene passato direttamente e clampato a [-1,1]
    dentro _cheb -> quindi il layer1 di fatto lavora su x clampato a [-1,1].
    Replichiamo ESATTAMENTE questo comportamento (clamp a [-1,1], nessuna
    rinormalizzazione da [-CLIP,CLIP])."""

    def __init__(self, coeffs1, coeffs2, degree):
        self.coeffs1 = np.asarray(coeffs1, dtype=np.float64)
        self.coeffs2 = np.asarray(coeffs2, dtype=np.float64)
        self.degree = degree

    def logits(self, X):
        # layer1: X clampato a [-1,1] (come fa torch in _cheb)
        h = _layer_forward(X, self.coeffs1, self.degree)   # (N, hidden)
        h = np.tanh(h)                                      # (N, hidden) in [-1,1]
        z = _layer_forward(h, self.coeffs2, self.degree)    # (N, C)
        return z

    def predict(self, X):
        return np.argmax(self.logits(X), axis=1)


def from_torch(model):
    """Estrae i coefficienti da un KANTorch addestrato."""
    c1 = model.l1.coeffs.detach().cpu().numpy()
    c2 = model.l2.coeffs.detach().cpu().numpy()
    deg = model.l1.degree
    return MultiLayerKANNumpy(c1, c2, deg)
