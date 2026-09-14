"""Ajuste de distribuciones al rinde detrendeado y ranking por AIC."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as stats

DISTRIBUTIONS = {
    "normal": stats.norm,
    "lognormal": stats.lognorm,
    "gamma": stats.gamma,
    "weibull": stats.weibull_min,
}


def fit_all(y: np.ndarray | pd.Series) -> pd.DataFrame:
    """Ajusta todas las distribuciones y devuelve tabla ordenada por AIC."""
    y = np.asarray(pd.Series(y).dropna(), dtype=float)
    y = y[y > 0]  # lognorm/gamma/weibull requieren soporte positivo
    rows = []
    for name, dist in DISTRIBUTIONS.items():
        try:
            params = dist.fit(y)
            ll = np.sum(dist.logpdf(y, *params))
            aic = 2 * len(params) - 2 * ll
            rows.append({"distribution": name, "aic": aic, "params": params})
        except Exception:
            continue
    return (pd.DataFrame(rows).sort_values("aic").reset_index(drop=True))


def sampler(name: str, y: np.ndarray | pd.Series):
    """Devuelve una función rvs(n, rng) para el método elegido.

    name ∈ {'empirical bootstrap', 'kde', 'normal', 'lognormal', 'gamma',
    'weibull'}.
    """
    y = np.asarray(pd.Series(y).dropna(), dtype=float)

    if name == "empirical bootstrap":
        return lambda n, rng: rng.choice(y, size=n, replace=True)

    if name == "kde":
        kde = stats.gaussian_kde(y[y > 0])
        return lambda n, rng: kde.resample(n, seed=rng).ravel()

    dist = DISTRIBUTIONS[name]
    params = dist.fit(y[y > 0])
    return lambda n, rng: dist.rvs(*params, size=n, random_state=rng)
