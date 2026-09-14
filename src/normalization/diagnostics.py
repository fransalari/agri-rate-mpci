"""Diagnósticos de residuos (spec §30) y guardrails agronómicos (§56).

Cada diagnóstico devuelve PASS / WARNING / FAIL con valor, umbral e
interpretación.
"""

from __future__ import annotations

import numpy as np
import scipy.stats as stats

_EPS = 1e-9


def _verdict(value, warn, fail, higher_is_worse=True):
    v = abs(value)
    if higher_is_worse:
        return "FAIL" if v >= fail else ("WARNING" if v >= warn else "PASS")
    return "FAIL" if v <= fail else ("WARNING" if v <= warn else "PASS")


def residual_diagnostics(years: np.ndarray, shocks: np.ndarray,
                         mode: str) -> dict:
    years = np.asarray(years, dtype=float)
    x = np.asarray(shocks, dtype=float)
    center = 1.0 if mode == "multiplicative" else 0.0
    r = x - center
    out = {}

    # 1. tendencia residual: no debería quedar tecnología en los shocks
    rho, p = stats.spearmanr(years, x)
    out["residual_trend"] = {
        "value": float(rho), "p_value": float(p), "threshold": 0.35,
        "status": _verdict(rho, 0.35, 0.55),
        "interpretation": "correlación shock–año; alta ⇒ el trend no capturó la tecnología",
    }

    # 2. autocorrelación lag-1
    if len(r) > 3 and np.std(r) > 0:
        r1 = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    else:
        r1 = 0.0
    out["autocorrelation"] = {
        "value": r1, "threshold": 0.45,
        "status": _verdict(r1, 0.45, 0.65),
        "interpretation": "autocorrelación lag-1 de los shocks",
    }

    # 3. estabilidad de varianza entre mitades
    h = len(r) // 2
    if h >= 4:
        s1, s2 = np.std(r[:h], ddof=1), np.std(r[h:], ddof=1)
        ratio = float(max(s1, s2) / max(min(s1, s2), _EPS))
    else:
        ratio = 1.0
    out["variance_stability"] = {
        "value": ratio, "threshold": 2.0,
        "status": _verdict(ratio, 2.0, 3.0),
        "interpretation": "ratio de desvíos entre mitades de la serie",
    }
    return out


def agronomic_guardrails(model, years, config, mode: str) -> list[str]:
    """Warnings/rechazos agronómicos; devuelve lista de motivos de rechazo."""
    reasons = []
    t = np.asarray(years, dtype=float)
    horizon = np.arange(t.min(), t.max() + config.max_extrapolation_years + 1)
    trend = model.predict(horizon)

    if np.any(~np.isfinite(trend)) or np.any(trend <= 0):
        reasons.append("trend no positivo/finito en el horizonte")
        return reasons

    # crecimiento anual implícito extremo o colapso absurdo
    growth = np.diff(trend) / trend[:-1]
    if np.nanmax(growth) > 3 * config.max_annual_tech_growth:
        reasons.append("crecimiento tecnológico anual extremo (explosión)")
    if np.nanmin(growth) < -3 * config.max_annual_tech_growth:
        reasons.append("caída tecnológica sostenida absurda")

    # oscilación: un trend tecnológico no debería cambiar de dirección seguido
    signs = np.sign(np.diff(trend))
    flips = int(np.sum(signs[:-1] * signs[1:] < 0))
    if flips > max(3, len(horizon) // 6):
        reasons.append(f"trend oscilante ({flips} cambios de dirección)")
    return reasons


def diagnostics_summary(diags: dict, n_eff: float, config) -> dict:
    """Resumen estilo output §41: incluye sample_size."""
    out = {k: v["status"] for k, v in diags.items()}
    if n_eff >= config.n_eff_favorable:
        out["sample_size"] = "PASS"
    elif n_eff >= config.minimum_effective_sample_size:
        out["sample_size"] = "WARNING"
    else:
        out["sample_size"] = "FAIL"
    return out
