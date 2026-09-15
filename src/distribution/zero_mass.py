"""Modelo de masa en cero (spec §33-39).

P(Y=0) > 0 es real (granizo total, sequía extrema, abandono). Una
distribución continua pura no puede producirlo ⇒ modelo hurdle:

    Y = 0                    con prob. p0
    Y ~ dist. positiva       con prob. 1 - p0

p0 se estima con shrinkage beta-binomial hacia un prior por nivel de
agregación: series cortas sin ceros observados NO implican p0 = 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = ("TRUE_TOTAL_LOSS", "PLANTED_NOT_HARVESTED",
              "MISSING_CODED_AS_ZERO", "NO_CROP_PLANTED", "SURVEY_ERROR",
              "UNKNOWN_ZERO")


def classify_zeros(raw_serie: pd.DataFrame) -> pd.DataFrame:
    """Clasifica campañas con rinde 0 usando superficie sembrada/cosechada
    (spec §36). Devuelve DataFrame year|category|detail."""
    rows = []
    cols = raw_serie.columns
    has_sup = ("superficie_sembrada_ha" in cols
               and "superficie_cosechada_ha" in cols)
    zero_rows = raw_serie[raw_serie["rendimiento_kgxha"].fillna(-1) == 0]
    for _, r in zero_rows.iterrows():
        year = int(r.get("anio", r.get("year", 0)))
        if not has_sup:
            cat, det = "UNKNOWN_ZERO", "sin datos de superficie para clasificar"
        else:
            semb = r.get("superficie_sembrada_ha")
            cos = r.get("superficie_cosechada_ha")
            if pd.notna(semb) and semb > 0 and (pd.isna(cos) or cos == 0):
                cat = "PLANTED_NOT_HARVESTED"
                det = f"sembradas {semb:,.0f} ha, cosechadas 0 → pérdida/abandono total"
            elif pd.notna(semb) and semb == 0:
                cat, det = "NO_CROP_PLANTED", "sin superficie sembrada"
            elif pd.notna(cos) and cos > 0:
                cat, det = "SURVEY_ERROR", f"rinde 0 con {cos:,.0f} ha cosechadas (inconsistente)"
            else:
                cat, det = "UNKNOWN_ZERO", "superficie no informada"
        rows.append({"year": year, "category": cat, "detail": det})
    return pd.DataFrame(rows, columns=["year", "category", "detail"])


def estimate_p_zero(zero_report: pd.DataFrame, n_years: int,
                    aggregation_level: str, config) -> dict:
    """Shrinkage beta-binomial (spec §35, §39):

        p0 = (k + m·prior) / (n + m)

    k = ceros contables, prior = prior del nivel, m = fuerza del prior.
    """
    level = aggregation_level.upper()
    prior = config.zero_prior_by_level.get(level, 0.001)
    m = config.zero_prior_strength
    countable = (zero_report[zero_report["category"].isin(config.zero_countable)]
                 if len(zero_report) else zero_report)
    k = int(len(countable))
    excluded = int(len(zero_report) - k)
    p0 = (k + m * prior) / (max(n_years, 1) + m)
    method = "HIERARCHICAL_SHRINKAGE" if k == 0 else "EMPIRICAL_SHRUNK"
    return {
        "enabled": True, "p_zero": float(p0), "observed_zeros": k,
        "excluded_zeros": excluded, "n_years": int(n_years),
        "aggregation_level": level, "prior": prior, "prior_strength": m,
        "estimation_method": method,
        "note": ("sin ceros observados: p0 > 0 por prior jerárquico del "
                 f"nivel {level}" if k == 0 else
                 f"{k} cero(s) contable(s) + shrinkage hacia prior {prior:.2%}"),
    }
