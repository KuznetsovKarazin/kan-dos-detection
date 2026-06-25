#!/usr/bin/env python3
"""
kan_chebyshev_multiclass.py — KAN Chebyshev per classificazione multiclass
==========================================================================
Estensione di ChebyshevKANBinary al caso a C classi.

Differenze rispetto al binario:
  - C uscite invece di 1  ->  in_dim x C edge (es. 10x10 = 100 edge)
  - testa softmax + argmax invece di sigmoid + soglia
  - loss cross-entropy categorica pesata per classe (gestisce MITM)

Struttura single-layer [in_dim -> C]:
  logit_c(x) = sum_i sum_d  coeffs[i,c,d] * T_d( norm(x_i) )
  pred       = argmax_c softmax(logit)_c

Gradiente esatto (come il binario): la single-layer si addestra in forma
chiusa rispetto ai coefficienti, niente backprop multi-strato.
"""

import numpy as np


def chebyshev_basis(x, degree):
    N = x.shape[0]
    T = np.empty((N, degree + 1), dtype=np.float64)
    T[:, 0] = 1.0
    if degree >= 1:
        T[:, 1] = x
    for n in range(2, degree + 1):
        T[:, n] = 2.0 * x * T[:, n - 1] - T[:, n - 2]
    return T


class ChebyshevKANMulticlass:
    """KAN single-layer [in_dim -> n_classes], base Chebyshev, softmax."""

    def __init__(self, in_dim, n_classes, degree=8, x_min=-3.5, x_max=3.5, seed=0):
        self.in_dim = in_dim
        self.C = n_classes
        self.degree = degree
        self.x_min, self.x_max = x_min, x_max
        rng = np.random.RandomState(seed)
        # coeffs[i, c, d]
        self.coeffs = rng.randn(in_dim, n_classes, degree + 1) * 0.05

    def _norm(self, X):
        xn = 2.0 * (X - self.x_min) / (self.x_max - self.x_min) - 1.0
        return np.clip(xn, -1.0, 1.0)

    def _logits(self, X):
        Xn = self._norm(X)
        self._bases = [chebyshev_basis(Xn[:, i], self.degree)
                       for i in range(self.in_dim)]
        N = X.shape[0]
        Z = np.zeros((N, self.C))
        for i in range(self.in_dim):
            # bases[i] (N, deg+1) @ coeffs[i] (deg+1 ... no): coeffs[i] is (C, deg+1)
            Z += self._bases[i] @ self.coeffs[i].T   # (N, C)
        return Z

    @staticmethod
    def _softmax(Z):
        Z = Z - Z.max(axis=1, keepdims=True)
        e = np.exp(np.clip(Z, -30, 30))
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, X, y, epochs=300, lr=0.3, l2=1e-4, verbose=True):
        N = X.shape[0]
        # one-hot + pesi di classe (bilanciamento, fondamentale per MITM)
        Y = np.zeros((N, self.C))
        Y[np.arange(N), y] = 1.0
        counts = np.bincount(y, minlength=self.C).astype(np.float64)
        cw = (N / (self.C * np.maximum(counts, 1)))     # peso inverso freq
        sw = cw[y]                                       # (N,)
        sw_sum = sw.sum()

        for ep in range(epochs):
            Z = self._logits(X)
            P = self._softmax(Z)
            # gradiente cross-entropy pesata: dL/dZ = sw*(P - Y)/sum(sw)
            G = (sw[:, None] * (P - Y)) / sw_sum         # (N, C)
            for i in range(self.in_dim):
                # dL/dcoeffs[i] = bases[i].T @ G  -> (deg+1, C) -> .T (C, deg+1)
                grad_i = (self._bases[i].T @ G).T + l2 * self.coeffs[i]
                self.coeffs[i] -= lr * grad_i
            if verbose and (ep % 60 == 0 or ep == epochs - 1):
                loss = -(sw * np.log(P[np.arange(N), y] + 1e-9)).sum() / sw_sum
                print(f"   epoch {ep:4d}/{epochs}  CE={loss:.4f}")

    def predict_proba(self, X):
        return self._softmax(self._logits(X))

    def predict(self, X):
        return np.argmax(self._logits(X), axis=1)
