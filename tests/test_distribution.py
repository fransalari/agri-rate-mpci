"""Tests del Yield Risk Distribution Engine (spec §73-84)."""

import numpy as np
import pandas as pd
import pytest

from src.distribution import RiskDistributionConfig, RiskDistributionEngine
from src.distribution.backtest import rolling_insurance_backtest
from src.distribution.candidates import ReflectedKDE, build_candidate
from src.distribution.volatility import select_volatility_model
from src.distribution.zero_mass import classify_zeros, estimate_p_zero

SEED = 11
CFG = RiskDistributionConfig(execution_mode="FAST", n_sims=6000,
                             random_seed=SEED)


def serie(years, y, semb=None, cos=None):
    d = pd.DataFrame({"anio": years, "campania": [f"{a}/{a+1}" for a in years],
                      "rendimiento_kgxha": y})
    if semb is not None:
        d["superficie_sembrada_ha"] = semb
        d["superficie_cosechada_ha"] = cos
    return d


def run(df, level="DEPARTMENT", cfg=None):
    eng = RiskDistributionEngine(cfg or CFG)
    return eng.run(df, aggregation_level=level, target_year=int(df["anio"].max()) + 1)


# ---------------------------------------------------------- §76 / §82
def test_underdispersed_parametric_rejected():
    """Cola pesada real: un paramétrico fino no puede ganar solo por AIC."""
    rng = np.random.default_rng(SEED)
    years = np.arange(1975, 2025)
    base = rng.normal(1.05, 0.10, len(years))
    crash = rng.random(len(years)) < 0.18            # años catastróficos
    shocks = np.where(crash, rng.uniform(0.25, 0.6, len(years)), base)
    y = 2000 * np.clip(shocks, 0.05, None)
    res = run(serie(years, y))
    assert res.status == "OK"
    r = res.ranking.set_index("distribution")
    sel = res.distribution["recommended"]
    # el seleccionado preserva la SD downside (gate §19)
    sel_row = next(c for c in res.candidates
                   if c.name == (sel if sel != "MODEL_ENSEMBLE"
                                 else res.distribution["label"].split()[-1]))
    assert sel_row.downside_sd_ratio >= CFG.gate_downside_sd_ratio
    # ningún candidato RECHAZADO puede ser el seleccionado
    rejected = {c.name for c in res.candidates if c.status.startswith("REJECTED")}
    assert sel not in rejected
    # y si normal/gamma comprimen la cola, deben quedar marcados
    thin = [c for c in res.candidates if c.name in ("normal", "gamma")
            and np.isfinite(c.downside_sd_ratio)
            and c.downside_sd_ratio < CFG.gate_downside_sd_ratio]
    for c in thin:
        assert c.status == "REJECTED_RISK_UNDERDISPERSION"


# ---------------------------------------------------------------- §77
def test_kde_competitive_on_skewed():
    rng = np.random.default_rng(SEED)
    years = np.arange(1978, 2025)
    comp = rng.random(len(years)) < 0.25
    shocks = np.where(comp, rng.normal(0.55, 0.08, len(years)),
                      rng.normal(1.12, 0.07, len(years)))     # bimodal
    y = 1800 * np.clip(shocks, 0.05, None)
    res = run(serie(years, y))
    r = res.ranking.set_index("distribution")
    kde_best = min(r.loc["kde", "risk_score"], r.loc["weighted_kde", "risk_score"])
    param = [n for n in ("normal", "lognormal", "gamma", "weibull") if n in r.index]
    ok_param = [n for n in param if "REJECTED" not in str(r.loc[n, "status"])]
    # KDE debe estar dentro del podio o superar a los paramétricos válidos
    if ok_param:
        assert kde_best <= min(r.loc[n, "risk_score"] for n in ok_param) * 1.15
    else:
        assert res.distribution["recommended"] in ("kde", "weighted_kde",
                                                   "empirical", "MODEL_ENSEMBLE")


# ---------------------------------------------------------- §78 / §68
def test_zero_mass_recovered_and_simulated():
    rng = np.random.default_rng(SEED)
    years = np.arange(1980, 2025)
    shocks = np.clip(rng.normal(1.0, 0.25, len(years)), 0.05, None)
    y = 1500 * shocks
    zero_years = rng.choice(len(years), size=2, replace=False)
    y[zero_years] = 0.0
    semb = np.full(len(years), 1000.0)
    cos = np.where(y == 0, 0.0, 950.0)
    res = run(serie(years, y, semb, cos), level="FARM")
    z = res.zero_mass
    assert z["observed_zeros"] == 2
    emp = 2 / len(years)
    assert 0.2 * emp < z["p_zero"] < 2.0 * emp        # shrinkage razonable
    frac0 = float((res.simulated_yields == 0).mean())
    assert abs(frac0 - z["p_zero"]) < 0.02            # el MC produce ceros


# ---------------------------------------------------------------- §79
def test_no_observed_zeros_p0_positive():
    rep = classify_zeros(serie(np.arange(2010, 2025),
                               np.full(15, 2000.0)))
    z = estimate_p_zero(rep, n_years=15, aggregation_level="FIELD", config=CFG)
    assert z["p_zero"] > 0
    assert z["estimation_method"] == "HIERARCHICAL_SHRINKAGE"


# ---------------------------------------------------------------- §80
def test_aggregation_level_monotonic():
    rng = np.random.default_rng(SEED)
    years = np.arange(1980, 2025)
    y = 1500 * np.clip(rng.normal(1.0, 0.22, len(years)), 0.05, None)
    df = serie(years, y)
    res = {lvl: run(df, level=lvl) for lvl in ("FIELD", "FARM", "DEPARTMENT")}
    p0 = {k: v.zero_mass["p_zero"] for k, v in res.items()}
    assert p0["FIELD"] > p0["FARM"] > p0["DEPARTMENT"]
    sd = {k: float(np.std(v.simulated_yields[v.simulated_yields > 0]))
          for k, v in res.items()}
    assert sd["FIELD"] > sd["FARM"] > sd["DEPARTMENT"]


# ---------------------------------------------------------------- §81
def test_kde_boundary_no_negatives():
    rng = np.random.default_rng(SEED)
    x = np.clip(rng.normal(0.35, 0.30, 60), 0.01, None)   # masa cerca de 0
    kde = ReflectedKDE("kde", positive_support=True).fit(x)
    draws = kde.sample(50_000, rng)
    assert (draws >= 0).all()
    # y la reflexión conserva la masa cercana al borde (no la corre lejos)
    assert np.quantile(draws, 0.05) < np.quantile(x, 0.25)


def test_engine_simulations_non_negative():
    rng = np.random.default_rng(SEED)
    years = np.arange(1985, 2025)
    y = 900 * np.clip(rng.normal(1.0, 0.45, len(years)), 0.02, None)
    res = run(serie(years, y), level="FIELD")
    assert res.status == "OK"
    assert (res.simulated_yields >= 0).all()


# ---------------------------------------------------------------- §75
def test_asymmetric_volatility_detected():
    rng = np.random.default_rng(SEED)
    years = np.arange(1975, 2025)
    down = -np.abs(rng.normal(0, 0.30, len(years)))
    up = np.abs(rng.normal(0, 0.10, len(years)))
    shocks = 1.0 + np.where(rng.random(len(years)) < 0.5, down, up)
    vol = select_volatility_model(years, np.clip(shocks, 0.02, None),
                                  "multiplicative", CFG)
    assert vol["asymmetric"]
    assert vol["downside_sd"] > vol["upside_sd"]


def test_constant_volatility_when_no_evidence():
    rng = np.random.default_rng(SEED)
    years = np.arange(1980, 2025)
    shocks = np.clip(rng.normal(1.0, 0.18, len(years)), 0.05, None)
    vol = select_volatility_model(years, shocks, "multiplicative", CFG)
    assert vol["model"] == "constant"                  # 1-SE: sin evidencia,
                                                       # sin complejidad (§32)


# ---------------------------------------------------------------- §45
def test_underpricing_penalized_asymmetrically():
    """El backtest castiga ~2× la prima subestimada vs sobreestimada (§45).

    Serie A: entrenamiento calmo, crisis en validación → modelo SUBPRECIA.
    Serie B (espejo): crisis en entrenamiento, validación calma → SOBREPRECIA.
    Con sesgos comparables, la penalidad de A debe superar a la de B.
    """
    rng = np.random.default_rng(SEED)
    n = 42
    years = np.arange(1983, 1983 + n)
    calm = np.clip(rng.normal(1.0, 0.08, n), 0.05, None)
    crash_idx = np.arange(n) % 3 == 0
    crisis = np.where(crash_idx, 0.55, calm)
    a = calm.copy(); a[-12:] = crisis[-12:]            # crisis solo al final
    b = calm.copy(); b[:-12] = crisis[:-12]            # crisis solo al inicio
    from src.normalization.validation import half_life_weights
    wfn = lambda ty, ref: half_life_weights(ty, ref_year=ref, half_life=float("inf"))
    vol_a = select_volatility_model(years, a, "multiplicative", CFG)
    vol_b = select_volatility_model(years, b, "multiplicative", CFG)
    bt_a = rolling_insurance_backtest(years, a, wfn, "empirical",
                                      "multiplicative", vol_a, CFG,
                                      np.random.default_rng(1))
    bt_b = rolling_insurance_backtest(years, b, wfn, "empirical",
                                      "multiplicative", vol_b, CFG,
                                      np.random.default_rng(1))
    assert bt_a["indemnity"] > bt_b["indemnity"]       # subestimar duele más


# ---------------------------------------------------------------- §84
def test_near_tie_not_high_confidence():
    rng = np.random.default_rng(3)
    years = np.arange(1998, 2025)                      # serie corta y ruidosa
    y = 1200 * np.clip(rng.normal(1.0, 0.35, len(years)), 0.05, None)
    res = run(serie(years, y))
    if res.status == "OK":
        stab = res.selection["stability_pct"]
        if stab < 60:
            assert (res.selection["confidence"] != "HIGH"
                    or res.distribution["recommended"] == "MODEL_ENSEMBLE")


# --------------------------------------------------------- contrato §54
def test_output_contract_and_governance():
    rng = np.random.default_rng(SEED)
    years = np.arange(1980, 2025)
    y = 1600 * np.clip(rng.normal(1.0, 0.22, len(years)), 0.05, None)
    res = run(serie(years, y))
    assert res.status == "OK"
    for key in ("model", "detrending_mode", "target_yield"):
        assert key in res.trend
    for key in ("p05", "p10", "p20", "p30", "es_05", "es_10"):
        assert np.isfinite(res.tail[key])
    assert res.tail["p05"] <= res.tail["p10"] <= res.tail["p20"]
    assert {"coverage", "claim_prob", "expected_indemnity",
            "hist_loss_cost"} <= set(res.claim_curve.columns)
    nh = res.normalized_history
    assert {"standardized_shock", "zero_event", "historical_weight"} <= set(nh.columns)
    for key in ("engine_version", "dataset_hash", "rejected_models",
                "aggregation_level"):
        assert key in res.governance
    # reproducibilidad
    res2 = run(serie(years, y))
    assert res2.distribution["recommended"] == res.distribution["recommended"]
    assert res2.tail["p10"] == pytest.approx(res.tail["p10"])


# ------------------------------------------------- extremos preservados §74
def test_extreme_years_remain_in_shocks():
    rng = np.random.default_rng(SEED)
    years = np.arange(1980, 2025)
    y = (1500 + 40 * (years - 1980)) * np.clip(rng.normal(1.0, 0.12, len(years)), 0.05, None)
    crash_year = int(years.max()) - 8                  # siempre dentro de la ventana
    y[years == crash_year] *= 0.30
    res = run(serie(years, y))
    nh = res.normalized_history
    row = nh[nh["year"] == crash_year]
    assert len(row) == 1 and float(row["relative_shock"].iloc[0]) < 0.6
    # el trend robusto NO absorbe el crash y la simulación lo hace posible
    mu = res.trend["target_yield"]
    assert float((res.simulated_yields < 0.6 * mu).mean()) > 0.004
