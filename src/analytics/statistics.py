"""Estadística descriptiva y métricas de riesgo de rendimiento."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as stats

PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def describe(y: pd.Series | np.ndarray) -> pd.Series:
    y = pd.Series(y).dropna().astype(float)
    out = {
        "n": int(len(y)),
        "mean": y.mean(),
        "median": y.median(),
        "std": y.std(ddof=1),
        "cv": y.std(ddof=1) / y.mean() if y.mean() else np.nan,
        "min": y.min(),
        "max": y.max(),
        "skewness": stats.skew(y),
        "kurtosis": stats.kurtosis(y),
    }
    for p in PERCENTILES:
        out[f"p{p}"] = np.percentile(y, p)
    return pd.Series(out)


def worst_years(df: pd.DataFrame, n: int = 5,
                yield_col: str = "detrended",
                year_col: str = "campania") -> pd.DataFrame:
    """Peores campañas según rinde (detrended si está disponible)."""
    col = yield_col if yield_col in df.columns else "rendimiento_kgxha"
    return (df.nsmallest(n, col)[[year_col, col]]
            .rename(columns={col: "rinde_kgxha"})
            .reset_index(drop=True))


def normality_test(y: pd.Series | np.ndarray) -> dict:
    y = pd.Series(y).dropna().astype(float)
    stat, p = stats.shapiro(y)
    return {"shapiro_stat": float(stat), "p_value": float(p),
            "normal_5pct": bool(p > 0.05)}
