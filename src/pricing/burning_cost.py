"""Burning cost histórico sobre la serie detrendeada.

    BurningCost = (1/n) Σ LossCost_t
    Frequency   = P(Indemnity > 0)
    Severity    = E[LossCost | Indemnity > 0]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .yield_insurance import Coverage, loss_cost, payout


def burning_cost(detrended_yields: np.ndarray | pd.Series,
                 cov: Coverage) -> dict:
    y = np.asarray(pd.Series(detrended_yields).dropna(), dtype=float)
    lc = loss_cost(y, cov)
    ind = payout(y, cov)
    hit = ind > 0
    return {
        "n_years": int(len(y)),
        "burning_cost": float(lc.mean()),                    # tasa pura
        "pure_premium_usd_ha": float(ind.mean()),
        "frequency": float(hit.mean()),
        "severity": float(lc[hit].mean()) if hit.any() else 0.0,
        "worst_loss_cost": float(lc.max()) if len(lc) else np.nan,
        "sum_insured_usd_ha": cov.sum_insured,
    }


def guarantee_curve(detrended_yields: np.ndarray | pd.Series,
                    expected_yield: float,
                    price: float = 0.40,
                    guarantees: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9)
                    ) -> pd.DataFrame:
    """Curva de tarifas por nivel de garantía (V0.2 del plan)."""
    rows = []
    for g in guarantees:
        cov = Coverage(expected_yield=expected_yield, guarantee=g, price=price)
        r = burning_cost(detrended_yields, cov)
        rows.append({
            "guarantee": g,
            "guaranteed_yield": cov.guaranteed_yield,
            "pure_premium_rate": r["burning_cost"],
            "frequency": r["frequency"],
            "severity": r["severity"],
        })
    return pd.DataFrame(rows)


def yearly_table(detrended: pd.DataFrame, cov: Coverage,
                 yield_col: str = "detrended",
                 year_col: str = "campania") -> pd.DataFrame:
    """Tabla año a año: rinde, indemnización y loss cost (estilo 'For Agro Cat')."""
    y = detrended[yield_col].to_numpy(dtype=float)
    out = detrended[[year_col, yield_col]].copy()
    out["indemnity_usd_ha"] = payout(y, cov)
    out["loss_cost"] = loss_cost(y, cov)
    return out
