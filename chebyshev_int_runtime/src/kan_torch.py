#!/usr/bin/env python3
"""
kan_torch.py — KAN Chebyshev multi-layer in PyTorch
===================================================
KAN profonda (multi-strato) addestrata con autograd, per recuperare il
divario con i gradient ensemble nel task multiclass — divario che la
versione single-layer NumPy non poteva colmare (nessuna interazione
tra feature).

Architettura: [in_dim -> hidden -> n_classes], ogni STRATO composto da
edge Chebyshev. Tra gli strati una nonlinearita' tanh che riporta nel
dominio [-1,1] (richiesto dalla base Chebyshev del layer successivo).

ESPORTABILITA' LUT: gli edge restano funzioni univariate Chebyshev, le
STESSE che build_lut_for_edges (lut-kan) sa quantizzare. La differenza
col single-layer e' che ci sono due strati da comporre sull'MCU: il
forward quantizzato dovra' valutare strato 1 -> tanh -> strato 2. Questo
e' il pezzo che nessuno dei due repo originali fornisce.

Questo file copre solo training + valutazione PyTorch. L'export LUT
multi-strato e' il passo successivo, da costruire solo se i numeri
giustificano il deployment.
"""

import numpy as np
import torch
import torch.nn as nn


class ChebyshevKANLayer(nn.Module):
    """Strato KAN: per ogni (input i, output j) una funzione univariata
    = combinazione di polinomi di Chebyshev T_0..T_degree.
    Input atteso in [-1, 1]. Parametri: coeffs (in_dim, out_dim, degree+1)."""

    def __init__(self, in_dim, out_dim, degree=8):
        super().__init__()
        self.in_dim, self.out_dim, self.degree = in_dim, out_dim, degree
        # init piccola; (in_dim, out_dim, degree+1)
        self.coeffs = nn.Parameter(torch.randn(in_dim, out_dim, degree + 1) * 0.1)

    def _cheb(self, x):
        # x: (N, in_dim) in [-1,1] -> T: (N, in_dim, degree+1)
        x = torch.clamp(x, -1.0, 1.0)
        T = [torch.ones_like(x), x]
        for n in range(2, self.degree + 1):
            T.append(2.0 * x * T[-1] - T[-2])
        return torch.stack(T, dim=-1)            # (N, in_dim, degree+1)

    def forward(self, x):
        T = self._cheb(x)                        # (N, in_dim, deg+1)
        # out[n,j] = sum_i sum_d T[n,i,d] * coeffs[i,j,d]
        return torch.einsum("nid,ijd->nj", T, self.coeffs)


class KANTorch(nn.Module):
    """KAN multi-strato [in_dim -> hidden -> n_classes].
    tanh tra gli strati per restare nel dominio Chebyshev."""

    def __init__(self, in_dim, n_classes, hidden=16, degree=8):
        super().__init__()
        self.l1 = ChebyshevKANLayer(in_dim, hidden, degree)
        self.l2 = ChebyshevKANLayer(hidden, n_classes, degree)

    def forward(self, x):
        h = torch.tanh(self.l1(x))               # in [-1,1] per il layer 2
        return self.l2(h)                        # logits (N, n_classes)


def train_kan_torch(Xtr, ytr, in_dim, n_classes, hidden=16, degree=8,
                    epochs=300, lr=0.01, weight_decay=1e-4,
                    class_weights=None, device="cpu", verbose=True):
    """Addestra la KAN multi-strato con Adam + cross-entropy pesata."""
    model = KANTorch(in_dim, n_classes, hidden, degree).to(device)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.long, device=device)
    cw = (torch.tensor(class_weights, dtype=torch.float32, device=device)
          if class_weights is not None else None)
    crit = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        logits = model(Xt)
        loss = crit(logits, yt)
        loss.backward()
        opt.step()
        if verbose and (ep % 50 == 0 or ep == epochs - 1):
            print(f"   epoch {ep:4d}/{epochs}  CE={loss.item():.4f}")
    return model


@torch.no_grad()
def predict_kan_torch(model, X, device="cpu"):
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    return model(Xt).argmax(dim=1).cpu().numpy()
