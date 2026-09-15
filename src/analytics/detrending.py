"""Motor de detrending para series de rendimiento.

Métodos: linear | quadratic | loess | moving_average.

Salidas por año:
    trend            → tendencia estimada
    detrended        → Y_t - Trend_t + Trend_ref  (aditivo, en kg/ha de la
                       campaña de referencia = última campaña por defecto)
    yield_index      → Y_t / Trend_t              (relativo)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

METHODS = ("linear", "quadratic", "loess", "kernel_regression", "moving_average", "none")


def detrend(df: pd.DataFrame,
            method: str = "linear",
            yield_col: str = "rendimiento_kgxha",
            year_col: str = "anio",
            reference_year: int | None = None,
            loess_frac: float = 0.5,
            ma_window: int = 5) -> pd.DataFrame:
    """Devuelve el DataFrame con columnas trend / detrended / yield_index."""
    if method not in METHODS:
        raise ValueError(f"method debe ser uno de {METHODS}")

    out = df.sort_values(year_col).reset_index(drop=True).copy()
    y = out[yield_col].astype(float).to_numpy()
    t = out[year_col].astype(float).to_numpy()

    if method == "none":
        trend = np.full_like(y, y.mean())
    elif method == "linear":
        trend = _polyfit(t, y, 1)
    elif method == "quadratic":
        trend = _polyfit(t, y, 2)
    elif method == "loess":
        trend = sm.nonparametric.lowess(y, t, frac=loess_frac, return_sorted=False)
    elif method == "kernel_regression":
        from src.normalization.trend_models import KernelRegressionModel
        trend = KernelRegressionModel().fit(t.astype(float), y).predict(t.astype(float))
    else:  # moving_average
        trend = (pd.Series(y).rolling(ma_window, center=True, min_periods=1)
                 .mean().to_numpy())

    # evitar tendencia <= 0 (posible en ajustes polinómicos con series cortas)
    trend = np.maximum(trend, 1e-6)

    ref_year = reference_year if reference_year is not None else int(out[year_col].max())
    ref_trend = float(trend[out[year_col] == ref_year][0]) if (out[year_col] == ref_year).any() else float(trend[-1])

    out["trend"] = trend
    out["detrended"] = y - trend + ref_trend
    out["detrended"] = out["detrended"].clip(lower=0.0)
    out["yield_index"] = y / trend
    out.attrs["reference_year"] = ref_year
    out.attrs["reference_trend"] = ref_trend
    out.attrs["method"] = method
    return out


def _polyfit(t: np.ndarray, y: np.ndarray, deg: int) -> np.ndarray:
    # centrar t para estabilidad numérica
    tc = t - t.mean()
    coefs = np.polyfit(tc, y, deg)
    return np.polyval(coefs, tc)


def _kernel_regression_trend(df):
    from src.normalization.trend_models import KernelRegressionModel
    m = KernelRegressionModel().fit(df["anio"].to_numpy(float),
                                    df["rendimiento_kgxha"].to_numpy(float))
    return m.predict(df["anio"].to_numpy(float))


def trend_slope(df: pd.DataFrame,
                yield_col: str = "rendimiento_kgxha",
                year_col: str = "anio") -> float:
    """Pendiente lineal (kg/ha por año) — para el dashboard."""
    t = df[year_col].astype(float).to_numpy()
    y = df[yield_col].astype(float).to_numpy()
    if len(t) < 2:
        return float("nan")
    return float(np.polyfit(t - t.mean(), y, 1)[0])
