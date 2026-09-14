"""Cobertura de rendimiento (yield shortfall / MPCI simplificado).

    GuaranteedYield = ExpectedYield × Guarantee
    LossYield       = max(0, GuaranteedYield − Yield)
    Indemnity       = min(LossYield × Price − Deducible, Límite) ≥ 0

Deducible y límite son opcionales (V0.3 los expone en la UI; el motor
ya los soporta).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Coverage:
    expected_yield: float          # kg/ha (rinde esperado en tecnología actual)
    guarantee: float = 0.70        # fracción del rinde esperado
    price: float = 0.40            # USD/kg
    deductible_pct: float = 0.0    # fracción de la suma asegurada, tipo franquicia deducible
    max_payout: float | None = None  # USD/ha (límite indemnizatorio)

    @property
    def guaranteed_yield(self) -> float:
        return self.expected_yield * self.guarantee

    @property
    def sum_insured(self) -> float:
        """Suma asegurada por ha = rinde garantizado × precio."""
        return self.guaranteed_yield * self.price


def payout(yields: np.ndarray | pd.Series, cov: Coverage) -> np.ndarray:
    """Indemnización USD/ha para un vector de rindes."""
    y = np.asarray(yields, dtype=float)
    loss_yield = np.maximum(0.0, cov.guaranteed_yield - y)
    indemnity = loss_yield * cov.price
    if cov.deductible_pct:
        indemnity = np.maximum(0.0, indemnity - cov.deductible_pct * cov.sum_insured)
    if cov.max_payout is not None:
        indemnity = np.minimum(indemnity, cov.max_payout)
    return indemnity


def loss_cost(yields: np.ndarray | pd.Series, cov: Coverage) -> np.ndarray:
    """Loss cost anual = indemnización / suma asegurada (tasa por año)."""
    return payout(yields, cov) / cov.sum_insured
