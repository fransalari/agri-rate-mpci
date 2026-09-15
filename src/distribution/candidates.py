"""Candidatos de distribución para shocks agrícolas (spec §25-28, §57-58).

Todos exponen la misma interfaz:
    fit(x, weights) → self
    sample(n, rng) → draws (respetando soporte)
    logpdf_sum(x)  → para AIC/AICc (parametricos; NaN si no aplica)

KDE con corrección de borde por REFLEXIÓN en 0 para soporte positivo
(spec §58): densidad f(x) = g(x) + g(-x) en [0,∞) ⇒ sin masa negativa.
El bandwidth sale de una grilla × Silverman elegida por verosimilitud
leave-one-out ponderada — nunca es una perilla actuarial (spec §28).
"""

from __future__ import annotations

import numpy as np
import scipy.stats as stats

_EPS = 1e-9


class BaseCandidate:
    name = "base"
    n_params = 2
    positive_support = True

    def __init__(self, positive_support: bool = True):
        self.positive_support = positive_support

    def fit(self, x, weights=None):
        raise NotImplementedError

    def sample(self, n, rng):
        raise NotImplementedError

    def logpdf_sum(self, x) -> float:
        return float("nan")

    def information_criteria(self, x) -> dict:
        ll = self.logpdf_sum(x)
        if not np.isfinite(ll):
            return {"aic": float("nan"), "aicc": float("nan"), "bic": float("nan")}
        k, n = self.n_params, len(x)
        aic = 2 * k - 2 * ll
        aicc = aic + (2 * k * (k + 1)) / max(n - k - 1, 1)
        bic = k * np.log(n) - 2 * ll
        return {"aic": aic, "aicc": aicc, "bic": bic}


class Parametric(BaseCandidate):
    _DISTS = {"normal": (stats.norm, 2), "lognormal": (stats.lognorm, 3),
              "gamma": (stats.gamma, 3), "weibull": (stats.weibull_min, 3),
              "student_t": (stats.t, 3)}

    def __init__(self, name: str, positive_support: bool = True):
        super().__init__(positive_support)
        self.name = name
        self._dist, self.n_params = self._DISTS[name]

    def fit(self, x, weights=None):
        x = np.asarray(x, float)
        # scipy no soporta pesos: submuestreo ponderado determinista para
        # aproximar el fit ponderado (documentado; pesos suaves ⇒ efecto menor)
        if weights is not None and np.ptp(weights) > 1e-12:
            w = np.asarray(weights, float)
            reps = np.maximum(1, np.round(w / w.max() * 4).astype(int))
            x = np.repeat(x, reps)
        if self.name in ("lognormal", "gamma", "weibull"):
            x = x[x > 0]
            self._params = self._dist.fit(x, floc=0)
        else:
            self._params = self._dist.fit(x)
        return self

    def sample(self, n, rng):
        draws = self._dist.rvs(*self._params, size=n, random_state=rng)
        if self.positive_support:
            bad = draws < 0
            tries = 0
            while bad.any() and tries < 20:
                draws[bad] = self._dist.rvs(*self._params, size=int(bad.sum()),
                                            random_state=rng)
                bad = draws < 0
                tries += 1
            draws = np.maximum(draws, 0.0)   # residual: documentado en governance
        return draws

    def logpdf_sum(self, x) -> float:
        x = np.asarray(x, float)
        if self.name in ("lognormal", "gamma", "weibull"):
            x = x[x > 0]
        with np.errstate(all="ignore"):
            lp = self._dist.logpdf(x, *self._params)
        return float(np.sum(lp)) if np.all(np.isfinite(lp)) else float("-inf")


class Empirical(BaseCandidate):
    name = "empirical"
    n_params = 0

    def fit(self, x, weights=None):
        self._x = np.asarray(x, float)
        w = np.ones_like(self._x) if weights is None else np.asarray(weights, float)
        self._p = w / w.sum()
        return self

    def sample(self, n, rng):
        return rng.choice(self._x, size=n, replace=True, p=self._p)


class ReflectedKDE(BaseCandidate):
    """KDE gaussiano con pesos y reflexión en 0 (suavizado bootstrap).

    Muestreo: elegir x_i con prob. w_i, sumar ruido N(0, h²); si el
    soporte es positivo, reflejar |·| (equivale a la densidad reflejada).
    """
    n_params = 1

    def __init__(self, name="kde", weighted=False, bw_factor=1.0,
                 positive_support=True):
        super().__init__(positive_support)
        self.name = name
        self.weighted = weighted
        self.bw_factor = bw_factor

    @staticmethod
    def silverman(x, w):
        n_eff = w.sum() ** 2 / max((w ** 2).sum(), _EPS)
        mu = np.average(x, weights=w)
        sd = np.sqrt(np.average((x - mu) ** 2, weights=w))
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        a = min(sd, iqr / 1.349) if iqr > 0 else sd
        return 0.9 * max(a, _EPS) * n_eff ** (-0.2)

    def fit(self, x, weights=None):
        self._x = np.asarray(x, float)
        w = (np.asarray(weights, float) if (self.weighted and weights is not None)
             else np.ones_like(self._x))
        self._p = w / w.sum()
        self.bandwidth = self.silverman(self._x, w) * self.bw_factor
        return self

    def sample(self, n, rng):
        idx = rng.choice(len(self._x), size=n, replace=True, p=self._p)
        draws = self._x[idx] + rng.normal(0.0, self.bandwidth, n)
        if self.positive_support:
            draws = np.abs(draws)            # reflexión: masa negativa → positiva
        return draws

    def loo_loglik(self, x=None) -> float:
        """Verosimilitud leave-one-out ponderada para elegir bandwidth."""
        x = self._x if x is None else np.asarray(x, float)
        h = max(self.bandwidth, _EPS)
        d = x[:, None] - self._x[None, :]
        K = np.exp(-0.5 * (d / h) ** 2) / (h * np.sqrt(2 * np.pi))
        if self.positive_support:
            d2 = x[:, None] + self._x[None, :]
            K = K + np.exp(-0.5 * (d2 / h) ** 2) / (h * np.sqrt(2 * np.pi))
        P = np.tile(self._p, (len(x), 1))
        if len(x) == len(self._x):
            np.fill_diagonal(K, 0.0)
            np.fill_diagonal(P, 0.0)
        dens = (K * P).sum(axis=1) / np.maximum(P.sum(axis=1), _EPS)
        return float(np.sum(np.log(np.maximum(dens, 1e-300))))


def build_candidate(name: str, mode: str, config, weights=None,
                    x=None) -> BaseCandidate:
    positive = mode == "multiplicative"
    if name in Parametric._DISTS:
        return Parametric(name, positive_support=positive)
    if name == "empirical":
        return Empirical(positive)
    if name in ("log_kernel", "weighted_log_kernel"):
        from .kernel_engine import LogKernelCandidate
        return LogKernelCandidate(name, weighted=name.startswith("weighted")
                                  ).fit(x, weights)
    if name in ("kde", "weighted_kde"):
        weighted = name == "weighted_kde"
        # bandwidth por grilla × Silverman vía LOO-likelihood (spec §28)
        best, best_ll = None, -np.inf
        for f in config.bandwidth_factors:
            cand = ReflectedKDE(name, weighted, f, positive).fit(x, weights)
            ll = cand.loo_loglik()
            if ll > best_ll:
                best, best_ll = cand, ll
        return best
    raise ValueError(name)


class EnsembleCandidate(BaseCandidate):
    """Mezcla de 2 candidatos (spec §49)."""
    name = "ensemble"
    n_params = 0

    def __init__(self, members: list, weights: list):
        super().__init__(members[0].positive_support)
        self.members = members
        self.mix = np.asarray(weights, float) / np.sum(weights)
        self.label = " + ".join(f"{w:.0%} {m.name}"
                                for m, w in zip(members, self.mix))

    def fit(self, x, weights=None):
        return self

    def sample(self, n, rng):
        counts = rng.multinomial(n, self.mix)
        parts = [m.sample(int(k), rng) for m, k in zip(self.members, counts) if k]
        return rng.permutation(np.concatenate(parts))
