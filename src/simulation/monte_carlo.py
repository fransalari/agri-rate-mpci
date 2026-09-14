"""Simulación Monte Carlo de la cobertura de rendimiento.

Genera rindes desde la distribución elegida (ajustada al detrended),
aplica la cobertura y devuelve pérdida esperada, probabilidad de
siniestro, VaR/TVaR y prima bruta con gastos y margen — la misma
estructura del workbook MPCI (Deductions / MR Margin).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..pricing.yield_insurance import Coverage, payout
from .distributions import sampler


def simulate(detrended_yields: np.ndarray | pd.Series,
             cov: Coverage,
             distribution: str = "empirical bootstrap",
             n_sim: int = 10_000,
             yield_bounds: tuple[float, float] | None = None,
             expense_ratio: float = 0.25,
             risk_margin: float = 0.10,
             seed: int | None = 42) -> dict:
    rng = np.random.default_rng(seed)
    rvs = sampler(distribution, detrended_yields)
    sims = np.asarray(rvs(n_sim, rng), dtype=float)

    if yield_bounds is not None:
        sims = np.clip(sims, *yield_bounds)
    sims = np.maximum(sims, 0.0)

    ind = payout(sims, cov)
    var95, var99 = np.percentile(ind, [95, 99])
    tail95 = ind[ind >= var95]
    tail99 = ind[ind >= var99]

    pure = float(ind.mean())
    return {
        "distribution": distribution,
        "n_sim": int(n_sim),
        "simulated_yields": sims,
        "indemnities": ind,
        "expected_loss_usd_ha": pure,
        "pure_premium_rate": pure / cov.sum_insured,
        "probability_of_loss": float((ind > 0).mean()),
        "var95": float(var95),
        "var99": float(var99),
        "tvar95": float(tail95.mean()) if len(tail95) else 0.0,
        "tvar99": float(tail99.mean()) if len(tail99) else 0.0,
        "gross_premium_usd_ha": pure * (1 + expense_ratio + risk_margin),
        "expense_ratio": expense_ratio,
        "risk_margin": risk_margin,
    }


def summary_table(result: dict) -> pd.DataFrame:
    rows = [
        ("Expected Loss (Pure Premium)", result["expected_loss_usd_ha"]),
        ("Pure Premium Rate", result["pure_premium_rate"]),
        ("Probability of Loss", result["probability_of_loss"]),
        ("VaR 95%", result["var95"]),
        ("VaR 99%", result["var99"]),
        ("TVaR 95%", result["tvar95"]),
        ("TVaR 99%", result["tvar99"]),
        ("Gross Premium", result["gross_premium_usd_ha"]),
    ]
    return pd.DataFrame(rows, columns=["Métrica", "Valor"])
