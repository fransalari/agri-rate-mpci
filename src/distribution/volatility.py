"""Capa de volatilidad (spec §16-20).

Y_t = μ_t × R_t  (o aditivo). La volatilidad de R_t puede variar en el
tiempo y ser asimétrica (downside ≠ upside). Selección basada en
evidencia con preferencia por el modelo más simple (filosofía 1-SE):
solo se agrega complejidad si hay señal estadística.

La des-volatilización estandariza los shocks para el ajuste de
distribución; la simulación re-aplica la volatilidad target (con el
multiplicador del nivel de agregación).
"""

from __future__ import annotations

import numpy as np
import scipy.stats as stats

_EPS = 1e-9


def _center(mode: str) -> float:
    return 1.0 if mode == "multiplicative" else 0.0


def downside_upside_sd(shocks: np.ndarray, center: float,
                       weights: np.ndarray | None = None) -> tuple[float, float]:
    """SD separada de desvíos bajo/sobre el centro (spec §18)."""
    x = np.asarray(shocks, float) - center
    w = np.ones_like(x) if weights is None else np.asarray(weights, float)
    down, up = x < 0, x >= 0
    def _sd(mask):
        if mask.sum() < 2:
            return float("nan")
        return float(np.sqrt(np.average(x[mask] ** 2, weights=w[mask])))
    return _sd(down), _sd(up)


def ewma_sigma(shocks: np.ndarray, center: float, halflife: float) -> np.ndarray:
    """σ_t EWMA causal: usa solo información hasta t-1 (sin leakage)."""
    x = np.asarray(shocks, float) - center
    lam = 0.5 ** (1.0 / halflife)
    n = len(x)
    var0 = float(np.var(x[: max(5, n // 4)], ddof=1)) if n >= 6 else float(np.var(x, ddof=1))
    sig2 = np.empty(n)
    prev = max(var0, _EPS)
    for i in range(n):
        sig2[i] = prev                       # σ_t conocido ANTES de observar x_t
        prev = lam * prev + (1 - lam) * x[i] ** 2
    return np.sqrt(np.maximum(sig2, _EPS))


def select_volatility_model(years: np.ndarray, shocks: np.ndarray,
                            mode: str, config) -> dict:
    """Evidencia → modelo. Constante salvo señal (spec §17 + §32).

    - deriva temporal de |desvío| (Spearman p < α) ⇒ dinámica (EWMA);
    - ratio downside/upside SD fuera de banda ⇒ asimétrica.
    """
    c = _center(mode)
    x = np.asarray(shocks, float)
    dev = np.abs(x - c)
    n = len(x)

    drift_p = 1.0
    if n >= 12 and np.std(dev) > 0:
        _, drift_p = stats.spearmanr(np.asarray(years, float), dev)
        drift_p = float(drift_p) if np.isfinite(drift_p) else 1.0
    dynamic = drift_p < config.vol_drift_alpha

    sd_down, sd_up = downside_upside_sd(x, c)
    n_down = int((x < c).sum())
    n_up = n - n_down
    ratio = (sd_down / sd_up) if (np.isfinite(sd_down) and np.isfinite(sd_up)
                                  and sd_up > 0) else 1.0
    asym = (n_down >= config.min_side_obs and n_up >= config.min_side_obs
            and not (config.vol_asym_band[0] <= ratio <= config.vol_asym_band[1]))

    model = ("asymmetric_ewma" if (dynamic and asym) else
             "ewma" if dynamic else
             "asymmetric" if asym else "constant")

    sigma_t = (ewma_sigma(x, c, config.ewma_halflife) if dynamic
               else np.full(n, max(float(np.std(x - c, ddof=1)), _EPS)))
    return {
        "model": model, "dynamic": dynamic, "asymmetric": asym,
        "drift_p_value": drift_p, "downside_sd": sd_down, "upside_sd": sd_up,
        "down_up_ratio": ratio, "sigma_t": sigma_t,
        "sigma_target": float(sigma_t[-1]),
        "evidence": {
            "variance_drift": "DYNAMIC" if dynamic else "STABLE",
            "asymmetry": "ASYMMETRIC" if asym else "SYMMETRIC",
        },
    }


def devolatilize(shocks: np.ndarray, vol: dict, mode: str) -> np.ndarray:
    """Shock estandarizado a σ de referencia (el target), preservando el
    centro. Con σ constante es la identidad."""
    c = _center(mode)
    x = np.asarray(shocks, float)
    scale = vol["sigma_target"] / np.maximum(vol["sigma_t"], _EPS)
    return c + (x - c) * scale


def apply_target_volatility(std_shocks: np.ndarray, vol: dict, mode: str,
                            level_multiplier: float = 1.0,
                            floor: float | None = None) -> np.ndarray:
    """Re-escala shocks simulados a la volatilidad target × nivel de
    agregación, con asimetría down/up si corresponde (spec §38, §56)."""
    c = _center(mode)
    x = np.asarray(std_shocks, float) - c
    if vol["asymmetric"] and np.isfinite(vol["downside_sd"]) and np.isfinite(vol["upside_sd"]):
        base_down, base_up = vol["downside_sd"], vol["upside_sd"]
        base = np.sqrt(0.5 * (base_down ** 2 + base_up ** 2))
        k_down = base_down / max(base, _EPS)
        k_up = base_up / max(base, _EPS)
        x = np.where(x < 0, x * k_down, x * k_up)
        # renormalizar para no duplicar la asimetría ya presente en la muestra
        x = x / max(np.sqrt(0.5 * (k_down ** 2 + k_up ** 2)), _EPS)
    out = c + x * level_multiplier
    if floor is not None:
        out = np.maximum(out, floor)
    return out
