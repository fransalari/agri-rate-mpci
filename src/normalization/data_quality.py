"""DataQualityReport (spec §25) y detección de structural breaks (§23–24).

Principio central: DATA ERROR ≠ CLIMATE EXTREME. Un rinde bajísimo puede
ser un evento climático real y NO se elimina automáticamente.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as stats

from .schemas import StructuralBreakReport

ERROR, WARNING, EXTREME = ("DATA_QUALITY_ERROR", "DATA_QUALITY_WARNING",
                           "POTENTIAL_CLIMATE_EXTREME")


def data_quality_report(years: np.ndarray, yields: np.ndarray) -> list[dict]:
    issues: list[dict] = []
    years = np.asarray(years)
    y = np.asarray(yields, dtype=float)

    def add(level, check, detail):
        issues.append({"level": level, "check": check, "detail": detail})

    dup = pd.Series(years).duplicated()
    if dup.any():
        add(ERROR, "duplicate_years", f"años duplicados: {sorted(set(years[dup]))}")
    if np.any(y < 0):
        add(ERROR, "negative_yields", f"{int((y < 0).sum())} valores negativos")
    if np.any(~np.isfinite(y)):
        add(ERROR, "non_finite", "valores no finitos en la serie")

    gaps = np.diff(np.sort(years))
    if len(gaps) and gaps.max() >= 3:
        add(WARNING, "large_gaps", f"hueco máximo de {int(gaps.max())} campañas")
    missing = int((gaps - 1).clip(min=0).sum()) if len(gaps) else 0
    if missing:
        add(WARNING, "missing_years", f"{missing} campañas faltantes")
    if len(y) < 15:
        add(WARNING, "short_series", f"solo {len(y)} observaciones")

    # valores repetidos sospechosos (redondeo/relleno)
    vc = pd.Series(y[y > 0]).value_counts()
    if len(vc) and vc.iloc[0] >= max(4, len(y) // 5):
        add(WARNING, "repeated_values",
            f"el valor {vc.index[0]:.0f} se repite {int(vc.iloc[0])} veces")

    # salto de media/varianza entre mitades → posible cambio de unidad/fuente
    if len(y) >= 16:
        a, b = y[: len(y) // 2], y[len(y) // 2:]
        if a.mean() > 0 and (b.mean() / a.mean() > 4 or a.mean() / max(b.mean(), 1e-9) > 4):
            add(WARNING, "mean_shift",
                "salto de media >4× entre mitades (¿cambio de unidad/fuente?)")

    # extremos: clasificarlos como potencial evento climático, no error.
    # Se evalúan sobre residuos de un trend lineal rápido para que la
    # evolución tecnológica no enmascare eventos antiguos.
    if len(y) >= 10 and np.nanstd(y) > 0:
        t0 = years.astype(float) - years.mean()
        beta = np.polyfit(t0, y, 1)
        resid = y - np.polyval(beta, t0)
        med, mad = np.median(resid), stats.median_abs_deviation(resid)
        if mad > 0:
            z = (resid - med) / (1.4826 * mad)
            for yr, val, zi in zip(years, y, z):
                if zi < -3 and val >= 0:
                    add(EXTREME, "low_yield_extreme",
                        f"{int(yr)}: rinde {val:.0f} (z robusto {zi:.1f}) — "
                        "posible evento climático; se conserva")
    return issues


# ---------------------------------------------------------------- breaks
def pettitt_test(x: np.ndarray) -> tuple[int, float, float]:
    """Pettitt (cambio de mediana). Devuelve (índice, K, p aprox)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    r = stats.rankdata(x)
    U = 2 * np.cumsum(r) - np.arange(1, n + 1) * (n + 1)
    k_idx = int(np.argmax(np.abs(U[:-1])))
    K = float(np.abs(U[:-1]).max())
    p = 2.0 * np.exp((-6.0 * K ** 2) / (n ** 3 + n ** 2))
    return k_idx, K, min(1.0, p)


def chow_test(years, y, break_year) -> tuple[float, float]:
    """Chow clásico en el año candidato (lineal en ambos segmentos)."""
    t = np.asarray(years, dtype=float)
    y = np.asarray(y, dtype=float)
    m = t <= break_year
    n1, n2, k = int(m.sum()), int((~m).sum()), 2
    if n1 < k + 2 or n2 < k + 2:
        return np.nan, np.nan

    def sse(tt, yy):
        X = np.column_stack([np.ones_like(tt), tt - tt.mean()])
        beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
        return float(((yy - X @ beta) ** 2).sum())

    s_pool = sse(t, y)
    s1, s2 = sse(t[m], y[m]), sse(t[~m], y[~m])
    num = (s_pool - s1 - s2) / k
    den = (s1 + s2) / (n1 + n2 - 2 * k)
    if den <= 0:
        return np.nan, np.nan
    F = num / den
    p = 1 - stats.f.cdf(F, k, n1 + n2 - 2 * k)
    return float(F), float(p)


def detect_structural_break(years, shocks, alpha: float = 0.05
                            ) -> StructuralBreakReport:
    """Pettitt sobre los shocks + Chow de corroboración, con la
    clasificación conservadora del spec §24: un break estadístico NO
    implica cortar la serie."""
    years = np.asarray(years)
    x = np.asarray(shocks, dtype=float)
    if len(x) < 12:
        return StructuralBreakReport(note="serie corta para test de break")

    idx, K, p = pettitt_test(x)
    year = int(years[idx])
    before, after = x[: idx + 1], x[idx + 1:]
    magnitude = float(abs(after.mean() - before.mean()) /
                      max(np.std(x, ddof=1), 1e-9))
    _, chow_p = chow_test(years, x, year)

    if p >= alpha:
        return StructuralBreakReport(detected=False, year=None, test="pettitt",
                                     statistic=K, p_value=p,
                                     confidence="NONE", suggested_action="KEEP")

    corroborated = np.isfinite(chow_p) and chow_p < alpha
    if magnitude >= 1.0 and corroborated:
        conf, action = "STRONG", "REVIEW"
        note = ("Break fuerte: revisar causa (¿riego, cambio metodológico, "
                "unidad?). Solo esas causas justifican HARD_CUTOFF.")
    elif magnitude >= 0.6:
        conf, action = "MODERATE", "DOWNWEIGHT"
        note = "Break moderado: el half-life ya reduce el peso de la historia previa."
    else:
        conf, action = "WEAK", "KEEP"
        note = "Señal débil: compatible con variabilidad climática."
    return StructuralBreakReport(detected=True, year=year, test="pettitt+chow",
                                 statistic=K, p_value=p, magnitude=magnitude,
                                 confidence=conf, suggested_action=action,
                                 note=note)
