"""Consolidado de cartera (pantalla RESULTADOS).

Tarificación técnica area-yield multi-departamento sobre la serie
normalizada del motor, con:
  - trigger por % del rinde esperado o rinde garantizado fijo;
  - recargos explícitos (deducciones + margen);
  - LOSS CAP opcional (tope de pérdida como % de la suma asegurada);
  - escenario CAT (período de retorno configurable, default 100 años)
    vía Monte Carlo sobre la distribución ajustada (AIC) de la serie
    normalizada;
  - peor año histórico de la cartera.

Todas las pérdidas se expresan como Loss Cost (LC) = fracción de la
suma asegurada: LC_t = clip((T − y_t)/T, 0, 1) con T = rinde gatillo
(area-yield puro). El cap trunca LC_t en cap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.simulation import distributions

_EPS = 1e-9


@dataclass
class PortfolioParams:
    trigger_mode: str = "pct"        # "pct" (% de E[y]) | "fixed" (kg/ha)
    trigger_value: float = 65.0      # % o kg/ha según modo
    sum_insured_ha: float = 200.0    # USD/ha
    deductions: float = 0.25
    mr_margin: float = 0.10
    loss_cap: float | None = None    # fracción de SA (0.5 = 50%); None = sin cap
    return_period: int = 100
    n_sims: int = 20_000
    seed: int = 42

    @property
    def loading_divisor(self) -> float:
        d = 1.0 - self.deductions - self.mr_margin
        if d <= 0.05:
            raise ValueError("deducciones + margen dejan divisor <= 5%")
        return d


def loss_cost(norm_yields: np.ndarray, trigger_yield: float,
              cap: float | None = None) -> np.ndarray:
    """LC area-yield como fracción de la suma asegurada."""
    lc = np.clip((trigger_yield - np.asarray(norm_yields, float))
                 / max(trigger_yield, _EPS), 0.0, 1.0)
    if cap is not None:
        lc = np.minimum(lc, cap)
    return lc


def department_result(name: str, normalized_history: pd.DataFrame,
                      expected_yield: float, superficie_ha: float,
                      params: PortfolioParams) -> dict:
    """Tarifa técnica + escenario CAT para un departamento."""
    nh = normalized_history
    y = nh["normalized_yield"].to_numpy(float)
    years = nh["year"].to_numpy(int)

    if params.trigger_mode == "fixed":
        trigger_yield = float(params.trigger_value)
    else:
        trigger_yield = float(expected_yield * params.trigger_value / 100.0)
    trigger_pct = trigger_yield / max(expected_yield, _EPS)

    sa_total = superficie_ha * params.sum_insured_ha
    cap = params.loss_cap

    lc_hist = loss_cost(y, trigger_yield, cap=None)
    lc_hist_cap = loss_cost(y, trigger_yield, cap=cap)
    pure = float(lc_hist_cap.mean())
    pure_nocap = float(lc_hist.mean())
    tech = pure / params.loading_divisor
    tech_nocap = pure_nocap / params.loading_divisor

    # --- CAT via Monte Carlo sobre distribución ajustada (AIC) ----------
    rng = np.random.default_rng(params.seed + (abs(hash(name)) % 10_000))
    try:
        ranking = distributions.fit_all(y)
        best = ranking.iloc[0]["dist"]
        draw = distributions.sampler(best, y)
        sims = np.asarray(draw(params.n_sims, rng), float)
    except Exception:
        best = "empirical"
        sims = rng.choice(y, size=params.n_sims, replace=True)
    sims = np.clip(sims, 0.0, None)
    q = 1.0 - 1.0 / params.return_period
    lc_sim = loss_cost(sims, trigger_yield, cap=None)
    lc_sim_cap = loss_cost(sims, trigger_yield, cap=cap)
    lc_cat = float(np.quantile(lc_sim, q))
    lc_cat_cap = float(np.quantile(lc_sim_cap, q))

    return {
        "departamento": name,
        "expected_yield": expected_yield,
        "trigger_pct": trigger_pct,
        "trigger_yield": trigger_yield,
        "superficie_ha": superficie_ha,
        "sa_total": sa_total,
        "pure_rate": pure,
        "pure_rate_nocap": pure_nocap,
        "tech_rate": tech,
        "tech_rate_nocap": tech_nocap,
        "prima_tecnica": tech * sa_total,
        "prima_tecnica_nocap": tech_nocap * sa_total,
        "cat_dist": best,
        "lc_cat": lc_cat_cap,
        "lc_cat_nocap": lc_cat,
        "perdida_cat": lc_cat_cap * sa_total,
        "perdida_cat_nocap": lc_cat * sa_total,
        "lc_by_year": pd.DataFrame({"year": years, "lc": lc_hist_cap,
                                    "lc_nocap": lc_hist}),
        "n_years": len(y),
    }


def consolidate(results: list[dict], params: PortfolioParams) -> dict:
    """Totales de cartera + peor año histórico (suma comonótona, sin
    diversificación — criterio conservador estándar para PML)."""
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "lc_by_year"}
                       for r in results])
    prima = float(df["prima_tecnica"].sum())
    prima_nocap = float(df["prima_tecnica_nocap"].sum())
    sa = float(df["sa_total"].sum())
    cat = float(df["perdida_cat"].sum())
    cat_nocap = float(df["perdida_cat_nocap"].sum())

    # peor año histórico: pérdida agregada por campaña
    frames = []
    for r in results:
        t = r["lc_by_year"].copy()
        t["loss"] = t["lc"] * r["sa_total"]
        t["loss_nocap"] = t["lc_nocap"] * r["sa_total"]
        t["sa"] = r["sa_total"]
        t["departamento"] = r["departamento"]
        frames.append(t)
    ally = pd.concat(frames, ignore_index=True)
    per_year = (ally.groupby("year")
                .agg(loss=("loss", "sum"), loss_nocap=("loss_nocap", "sum"),
                     sa=("sa", "sum"), n=("departamento", "nunique"))
                .reset_index())
    # solo años donde reporta al menos la mitad de la SA de la cartera
    eligible = per_year[per_year["sa"] >= 0.5 * sa]
    if len(eligible):
        worst = eligible.loc[eligible["loss"].idxmax()]
        worst_year = int(worst["year"])
        worst_loss = float(worst["loss"])
        worst_loss_nocap = float(worst["loss_nocap"])
        worst_detail = (ally[ally["year"] == worst_year]
                        [["departamento", "lc", "loss"]]
                        .sort_values("loss", ascending=False))
    else:
        worst_year, worst_loss, worst_loss_nocap, worst_detail = None, np.nan, np.nan, None

    return {
        "table": df,
        "prima_tecnica": prima, "prima_tecnica_nocap": prima_nocap,
        "sa_total": sa,
        "superficie": float(df["superficie_ha"].sum()),
        "tasa_media": prima / max(sa, _EPS),
        "tasa_media_nocap": prima_nocap / max(sa, _EPS),
        "perdida_cat": cat, "perdida_cat_nocap": cat_nocap,
        "pml_x": cat / max(prima, _EPS),
        "pml_x_nocap": cat_nocap / max(prima_nocap, _EPS),
        "worst_year": worst_year, "worst_loss": worst_loss,
        "worst_loss_nocap": worst_loss_nocap,
        "worst_x": worst_loss / max(prima, _EPS),
        "worst_detail": worst_detail,
        "per_year": per_year.sort_values("year"),
    }


def superficie_estimada(df_cultivo: pd.DataFrame, departamento: str,
                        n_last: int = 3) -> float:
    """Superficie sembrada estimada: promedio de las últimas n campañas."""
    s = (df_cultivo[df_cultivo["departamento"] == departamento]
         .dropna(subset=["superficie_sembrada_ha"])
         .sort_values("anio").tail(n_last))
    return float(s["superficie_sembrada_ha"].mean()) if len(s) else 0.0
