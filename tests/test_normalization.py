"""Tests sintéticos del Yield Risk Normalization Engine (spec §57 + §7 + §61)."""

import numpy as np
import pandas as pd
import pytest

from src.normalization import (ModelConfiguration, YieldNormalizationConfig,
                               YieldNormalizationEngine)
from src.normalization.data_quality import detect_structural_break
from src.normalization.engine import _parsimony_pick
from src.normalization.schemas import CandidateResult
from src.normalization.validation import run_cv, validation_years

SEED = 7


def make_series(years, yields):
    return pd.DataFrame({"year": years, "yield": yields})


def run(series, mode="FAST", **kw):
    cfg = YieldNormalizationConfig(execution_mode=mode, random_seed=SEED)
    return YieldNormalizationEngine(cfg).run(series, **kw)


def best_per_model(res):
    r = res.model_ranking
    return r.loc[r.groupby("model")["cv_score"].idxmin()].set_index("model")


# ---------------------------------------------------------------- CASE A
def test_case_a_linear_trend():
    rng = np.random.default_rng(SEED)
    years = np.arange(1980, 2026)
    y = 2000 + 55 * (years - 1980) + rng.normal(0, 220, len(years))
    res = run(make_series(years, y), target_year=2026)
    assert res.recommended.trend_model in ("linear", "robust_linear", "log_linear")
    # el rinde esperado recupera el nivel teórico
    assert res.expected_yield == pytest.approx(2000 + 55 * 46, rel=0.08)


# ---------------------------------------------------------------- CASE B
def test_case_b_droughts_robust_stable():
    rng = np.random.default_rng(SEED)
    years = np.arange(1980, 2026)
    y = 2500 + 50 * (years - 1980) + rng.normal(0, 200, len(years))
    for bad in (1988, 2009, 2018):
        y[years == bad] *= 0.35   # sequías severas
    res = run(make_series(years, y), target_year=2026)
    bpm = best_per_model(res)
    # robusto rinde al menos como OLS, o su trend es más estable
    assert (bpm.loc["robust_linear", "cv_score"] <= bpm.loc["linear", "cv_score"] * 1.02
            or bpm.loc["robust_linear", "instability"] <= bpm.loc["linear", "instability"])


# ---------------------------------------------------------------- CASE C
def test_case_c_exponential_growth():
    rng = np.random.default_rng(SEED)
    years = np.arange(1975, 2026)
    y = 1500 * 1.03 ** (years - 1975) * np.exp(rng.normal(0, 0.10, len(years)))
    res = run(make_series(years, y), target_year=2026)
    champ = res.numerical_champion_score
    bpm = best_per_model(res)
    # log-linear competitivo: dentro de 1 SE del campeón, o seleccionado
    assert (res.recommended.trend_model == "log_linear"
            or bpm.loc["log_linear", "cv_score"] <= champ + res.score_se * 1.5)


# ---------------------------------------------------------------- CASE D
def test_case_d_structural_break_detected():
    rng = np.random.default_rng(SEED)
    years = np.arange(1975, 2026)
    shocks = rng.normal(1.0, 0.12, len(years))
    shocks[years >= 1996] += 0.5     # salto real de nivel relativo
    rep = detect_structural_break(years, shocks)
    assert rep.detected and abs(rep.year - 1996) <= 3
    assert rep.confidence in ("MODERATE", "STRONG")


# ---------------------------------------------------------------- CASE E
def test_case_e_no_trend_constant_wins():
    rng = np.random.default_rng(SEED)
    years = np.arange(1985, 2026)
    y = 3000 + rng.normal(0, 350, len(years))
    res = run(make_series(years, y), target_year=2026)
    assert res.recommended.trend_model == "constant"
    assert res.expected_yield == pytest.approx(3000, rel=0.07)


# ---------------------------------------------------------------- CASE F
def test_case_f_old_history_irrelevant():
    rng = np.random.default_rng(SEED)
    years = np.arange(1970, 2026)
    y = np.where(years < 1995, 1800.0, 1800 + 90 * (years - 1995))
    y = y + rng.normal(0, 150, len(years))
    res = run(make_series(years, y), target_year=2026)
    # el motor descarta o descuenta la historia plana previa
    assert (res.recommended.start_year >= 1985
            or res.recommended.half_life_years <= 25)


# ---------------------------------------------------------------- CASE G
def test_case_g_old_extreme_retained():
    rng = np.random.default_rng(SEED)
    years = np.arange(1980, 2026)
    y = 2200 + 45 * (years - 1980) + rng.normal(0, 180, len(years))
    y[years == 1988] = (2200 + 45 * 8) * 0.40   # evento severo antiguo
    res = run(make_series(years, y), target_year=2026)
    nh = res.normalized_history
    if 1988 in set(nh["year"]):
        row = nh[nh["year"] == 1988].iloc[0]
        # el shock severo se conserva en la historia normalizada (no se borra)
        assert row["relative_shock"] < 0.65
        assert row["normalized_yield"] < 0.65 * res.expected_yield
    extremes = [i for i in res.data_quality
                if i["level"] == "POTENTIAL_CLIMATE_EXTREME"]
    assert extremes, "el extremo debe clasificarse como clima, no como error"


# ---------------------------------------------------------------- CASE H
def test_case_h_small_sample_low_confidence():
    rng = np.random.default_rng(SEED)
    years = np.arange(2012, 2025)
    y = 2800 + rng.normal(0, 300, len(years))
    res = run(make_series(years, y), target_year=2026)
    assert res.selection_confidence != "HIGH"
    assert res.warnings


def test_insufficient_data():
    res = run(make_series([2020, 2021, 2022, 2023], [1, 2, 3, 4]))
    assert res.status == "INSUFFICIENT_DATA"


# ---------------------------------------------------------------- CASE I
def test_case_i_spline_not_selected_on_noise():
    rng = np.random.default_rng(SEED)
    years = np.arange(1980, 2026)
    y = 3000 + 500 * np.sin((years - 1980) / 2.5) * 0 + rng.normal(0, 400, len(years))
    res = run(make_series(years, y), mode="FULL", target_year=2026)
    assert res.recommended.trend_model in ("constant", "linear",
                                           "robust_linear", "log_linear")


# ---------------------------------------------------------------- CASE J
def _cand(model, score, se, tail=0.15, n_eff=30.0, inst=0.02):
    c = CandidateResult(config=ModelConfiguration(model, "multiplicative",
                                                  1980, 20.0))
    c.score, c.score_se, c.tail_error = score, se, tail
    c.n_eff, c.instability_penalty, c.aicc = n_eff, inst, 200.0
    c.complexity_penalty = YieldNormalizationConfig().complexity_penalty(model)
    return c


def test_case_j_one_se_rule_prefers_simpler():
    cfg = YieldNormalizationConfig()
    gam = _cand("spline", 0.151, 0.006, tail=0.152, n_eff=30.1)
    robust = _cand("robust_linear", 0.154, 0.006, tail=0.149, n_eff=30.9)
    final_set = [c for c in (gam, robust)
                 if c.score <= gam.score + gam.score_se]
    assert robust in final_set
    assert _parsimony_pick(final_set, cfg).config.trend_model == "robust_linear"


def test_case_j_material_gap_keeps_champion():
    cfg = YieldNormalizationConfig()
    gam = _cand("spline", 0.120, 0.004)
    robust = _cand("robust_linear", 0.154, 0.004)
    final_set = [c for c in (gam, robust)
                 if c.score <= gam.score + gam.score_se]
    assert _parsimony_pick(final_set, cfg).config.trend_model == "spline"


# ------------------------------------------------------------- CASE K/L
def test_case_k_multiplicative_shocks():
    rng = np.random.default_rng(SEED)
    years = np.arange(1980, 2026)
    trend = 1500 + 80 * (years - 1980)
    y = trend * np.exp(rng.normal(0, 0.22, len(years)))
    res = run(make_series(years, y), target_year=2026)
    r = res.model_ranking
    best_mult = r[r["mode"] == "mult"]["cv_score"].min()
    best_add = r[r["mode"] == "addi"]["cv_score"].min()
    assert best_mult <= best_add


def test_case_l_additive_shocks_can_win():
    rng = np.random.default_rng(SEED)
    years = np.arange(1980, 2026)
    trend = 1500 + 80 * (years - 1980)
    y = trend + rng.normal(0, 250, len(years))
    res = run(make_series(years, y), target_year=2026)
    r = res.model_ranking
    best_mult = r[r["mode"] == "mult"]["cv_score"].min()
    best_add = r[r["mode"] == "addi"]["cv_score"].min()
    assert best_add <= best_mult * 1.03


# ------------------------------------------------------------- LEAKAGE
def test_no_future_leakage():
    """Modificar años futuros NO puede cambiar predicciones pasadas (spec §7)."""
    rng = np.random.default_rng(SEED)
    years = np.arange(1985, 2026)
    y = 2000 + 40 * (years - 1985) + rng.normal(0, 200, len(years))
    cfg = YieldNormalizationConfig()
    val = validation_years(years, cfg.minimum_training_years,
                           cfg.max_validation_years)
    base = run_cv(years, y, model_name="linear", mode="multiplicative",
                  start_year=1985, half_life=20.0, val_years=val, config=cfg)

    cut = val[len(val) // 2]         # alterar drásticamente desde `cut`
    y2 = y.copy()
    y2[years >= cut] *= 5.0
    alt = run_cv(years, y2, model_name="linear", mode="multiplicative",
                 start_year=1985, half_life=20.0, val_years=val, config=cfg)

    for f_base, f_alt in zip(base["folds"], alt["folds"]):
        if f_base["year"] < cut:
            assert f_alt["abs_err"] == pytest.approx(f_base["abs_err"])
            assert f_alt["pinball"] == pytest.approx(f_base["pinball"])


# ---------------------------------------------------------- GOVERNANCE
def test_reproducibility_and_governance():
    rng = np.random.default_rng(SEED)
    years = np.arange(1985, 2026)
    y = 2500 + 30 * (years - 1985) + rng.normal(0, 250, len(years))
    s = make_series(years, y)
    r1, r2 = run(s, target_year=2026), run(s, target_year=2026)
    assert r1.recommended == r2.recommended
    assert r1.validation_score == pytest.approx(r2.validation_score)
    gov = r1.governance
    for key in ("engine_version", "dataset_hash", "random_seed",
                "selected_model", "numerical_champion", "selection_reason"):
        assert key in gov
    assert r1.selection_reason and r1.model_ranking is not None


def test_output_contract():
    """Criterio de aceptación funcional (spec §66)."""
    rng = np.random.default_rng(SEED)
    years = np.arange(1980, 2026)
    y = 1800 + 50 * (years - 1980) + rng.normal(0, 200, len(years))
    res = run(make_series(years, y), crop="soja", location="test",
              target_year=2026)
    assert res.recommended is not None
    assert np.isfinite(res.expected_yield)
    nh = res.normalized_history
    assert set(nh.columns) >= {"year", "observed_yield", "historical_trend",
                               "relative_shock", "normalized_yield",
                               "historical_weight"}
    assert np.isfinite(res.effective_sample_size)
    assert res.selection_confidence in ("HIGH", "MEDIUM", "LOW")
    assert res.diagnostics and "sample_size" in res.diagnostics
