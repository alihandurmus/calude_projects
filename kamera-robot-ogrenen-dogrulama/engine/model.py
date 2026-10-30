"""p(olay gerçek | bağlam) için Bayesçi lojistik regresyon (Laplace yaklaşımı)."""

from __future__ import annotations

import numpy as np


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=float)))


class BayesianLogisticRegression:
    """Az veriyle çalışan ve ne kadar emin olmadığını bilen olasılık modeli.

    - Sıfırdan değil, ön bilgiyle (prior) başlar; ilk günden makul kararlar verir.
    - Ağırlıkların ortalamasıyla birlikte kovaryansını da tutar. "Burada henüz emin değilim"
      bilgisi Thompson örneklemesiyle keşfi yönlendirir.
    - Etiketler ağırlıklı kullanılabilir; eski etiketler böyle unutulur.
    """

    def __init__(self, prior_mean: np.ndarray, prior_var: np.ndarray):
        self.prior_mean = np.asarray(prior_mean, dtype=float)
        self.prior_prec = 1.0 / np.asarray(prior_var, dtype=float)
        self.reset()

    def reset(self) -> None:
        self.mean = self.prior_mean.copy()
        self.cov = np.diag(1.0 / self.prior_prec)

    def fit(self, X, y, w=None, max_iter: int = 30, tol: float = 1e-8) -> None:
        """MAP ağırlıklarını Newton yöntemiyle bulur; kovaryans Hessian'ın tersidir."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(y) == 0:
            self.reset()
            return
        w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
        theta = self.mean.copy()  # önceki çözümden başla: birkaç iterasyonda yakınsar
        f = self._objective(theta, X, y, w)
        for _ in range(max_iter):
            p = sigmoid(X @ theta)
            grad = X.T @ (w * (p - y)) + self.prior_prec * (theta - self.prior_mean)
            step = np.linalg.solve(self._hessian(X, w, p), grad)
            t = 1.0
            while t > 1e-4:  # adım küçültmeli Newton: amaç fonksiyonu her adımda azalmalı
                cand = theta - t * step
                f_cand = self._objective(cand, X, y, w)
                if f_cand <= f:
                    break
                t *= 0.5
            else:
                break  # iyileşme yok, zaten optimumdayız
            theta, f = cand, f_cand
            if np.max(np.abs(t * step)) < tol:
                break
        self.mean = theta
        cov = np.linalg.inv(self._hessian(X, w, sigmoid(X @ theta)))
        self.cov = 0.5 * (cov + cov.T)

    def logit_moments(self, x: np.ndarray) -> tuple[float, float]:
        """Bu olay için logit'in posterior ortalaması ve varyansı."""
        return float(x @ self.mean), max(float(x @ self.cov @ x), 0.0)

    def _objective(self, theta, X, y, w) -> float:
        z = X @ theta
        d = theta - self.prior_mean
        return float(np.sum(w * (np.logaddexp(0.0, z) - y * z)) + 0.5 * np.sum(self.prior_prec * d * d))

    def _hessian(self, X, w, p) -> np.ndarray:
        return (X * (w * p * (1.0 - p))[:, None]).T @ X + np.diag(self.prior_prec)
