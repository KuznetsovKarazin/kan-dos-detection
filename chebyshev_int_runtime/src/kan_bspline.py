#!/usr/bin/env python3
"""
kan_bspline.py — KAN con edge B-spline (base locale, alla lut-kan)
=================================================================
Alternativa a kan_chebyshev. Gli edge usano B-spline (Cox-de Boor) invece
dei polinomi di Chebyshev. Differenza chiave per il deployment LUT:
  - Chebyshev: base GLOBALE (ogni punto dipende da tutti i coefficienti)
  - B-spline:  base LOCALE  (ogni punto dipende da pochi nodi vicini)
La base locale tende a quantizzarsi meglio in LUT -> e' l'ipotesi da testare.

Il forward e' lineare nei coefficienti (come Chebyshev), quindi si addestra
con lo stesso schema a gradiente. La base e' presa da lut-kan
(bspline_basis_all_numpy, Cox-de Boor).

phi_e(x) = sum_j coef[e,j] * N_{j,degree}(x)
"""

import numpy as np


def bspline_basis(x, knots, degree, boundary_mode="half_open"):
    """Cox-de Boor: ritorna B (N, coef_len). Replica bspline_basis_all_numpy
    di lut-kan, trasposta per comodita' (campioni sulle righe)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    t = np.asarray(knots, dtype=np.float64).reshape(-1)
    k = int(degree)
    M, N = t.shape[0], x.shape[0]
    if M < k + 2:
        raise ValueError(f"knots troppo corti: M={M}, degree={k}")
    B = np.zeros((M - 1, N), dtype=np.float64)
    for i in range(M - 1):
        left, right = t[i], t[i + 1]
        if boundary_mode == "closed" and i == (M - 2):
            B[i, :] = ((x >= left) & (x <= right)).astype(np.float64)
        else:
            B[i, :] = ((x >= left) & (x < right)).astype(np.float64)
    for d in range(1, k + 1):
        Bn = np.zeros((M - 1 - d, N), dtype=np.float64)
        for i in range(M - 1 - d):
            de1 = t[i + d] - t[i]
            de2 = t[i + d + 1] - t[i + 1]
            term1 = ((x - t[i]) / de1) * B[i, :] if de1 != 0 else 0.0
            term2 = ((t[i + d + 1] - x) / de2) * B[i + 1, :] if de2 != 0 else 0.0
            Bn[i, :] = term1 + term2
        B = Bn
    return B.T   # (N, coef_len)


def make_knots(x_min, x_max, n_intervals, degree):
    """Knot vector uniforme con padding ai bordi (clamped-ish)."""
    inner = np.linspace(x_min, x_max, n_intervals + 1)
    left = np.repeat(x_min, degree)
    right = np.repeat(x_max, degree)
    return np.concatenate([left, inner, right]).astype(np.float64)


class BSplineKANBinary:
    """KAN single-layer [in_dim->1], edge B-spline, training BCE."""

    def __init__(self, in_dim, n_intervals=8, degree=3, x_min=-3.5, x_max=3.5, seed=0):
        self.in_dim = in_dim
        self.degree = degree
        self.x_min, self.x_max = x_min, x_max
        self.knots = make_knots(x_min, x_max, n_intervals, degree)
        self.coef_len = self.knots.shape[0] - degree - 1
        rng = np.random.RandomState(seed)
        self.coef = rng.randn(in_dim, self.coef_len) * 0.05

    def _phi_sum(self, X):
        self._B = []
        N = X.shape[0]
        z = np.zeros(N)
        for i in range(self.in_dim):
            xi = np.clip(X[:, i], self.x_min, self.x_max - 1e-6)
            Bi = bspline_basis(xi, self.knots, self.degree)   # (N, coef_len)
            self._B.append(Bi)
            z += Bi @ self.coef[i]
        return z

    @staticmethod
    def _sig(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y, epochs=250, lr=0.3, l2=1e-4, verbose=False):
        y = y.astype(np.float64)
        for ep in range(epochs):
            z = self._phi_sum(X)
            p = self._sig(z)
            g = (p - y) / X.shape[0]
            for i in range(self.in_dim):
                self.coef[i] -= lr * (self._B[i].T @ g + l2 * self.coef[i])
            if verbose and (ep % 60 == 0 or ep == epochs - 1):
                loss = -np.mean(y*np.log(p+1e-9)+(1-y)*np.log(1-p+1e-9))
                print(f"   epoch {ep:4d}/{epochs}  BCE={loss:.4f}")

    def decision_logit(self, X):
        return self._phi_sum(X)

    def predict(self, X):
        return (self._sig(self._phi_sum(X)) >= 0.5).astype(int)


class BSplineKANMulticlass:
    """KAN single-layer [in_dim->C], edge B-spline, softmax+CE pesata."""

    def __init__(self, in_dim, n_classes, n_intervals=8, degree=3,
                 x_min=-3.5, x_max=3.5, seed=0):
        self.in_dim, self.C = in_dim, n_classes
        self.degree = degree
        self.x_min, self.x_max = x_min, x_max
        self.knots = make_knots(x_min, x_max, n_intervals, degree)
        self.coef_len = self.knots.shape[0] - degree - 1
        rng = np.random.RandomState(seed)
        self.coef = rng.randn(in_dim, n_classes, self.coef_len) * 0.05

    def _logits(self, X):
        self._B = []
        N = X.shape[0]
        Z = np.zeros((N, self.C))
        for i in range(self.in_dim):
            xi = np.clip(X[:, i], self.x_min, self.x_max - 1e-6)
            Bi = bspline_basis(xi, self.knots, self.degree)
            self._B.append(Bi)
            Z += Bi @ self.coef[i].T
        return Z

    @staticmethod
    def _softmax(Z):
        Z = Z - Z.max(axis=1, keepdims=True)
        e = np.exp(np.clip(Z, -30, 30))
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, X, y, epochs=300, lr=0.3, l2=1e-4, verbose=False):
        N = X.shape[0]
        Y = np.zeros((N, self.C)); Y[np.arange(N), y] = 1.0
        counts = np.bincount(y, minlength=self.C).astype(np.float64)
        sw = (N/(self.C*np.maximum(counts,1)))[y]; sws = sw.sum()
        for ep in range(epochs):
            P = self._softmax(self._logits(X))
            G = (sw[:, None]*(P-Y))/sws
            for i in range(self.in_dim):
                self.coef[i] -= lr*((self._B[i].T @ G).T + l2*self.coef[i])
            if verbose and (ep % 60 == 0 or ep == epochs-1):
                loss = -(sw*np.log(P[np.arange(N), y]+1e-9)).sum()/sws
                print(f"   epoch {ep:4d}/{epochs}  CE={loss:.4f}")

    def predict(self, X):
        return np.argmax(self._logits(X), axis=1)
