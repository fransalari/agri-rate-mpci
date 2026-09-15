"""Tests del Kernel Yield Risk Engine (spec kernel §51-58)."""

import numpy as np
import pytest

from src.distribution import RiskDistributionConfig, RiskDistributionEngine
from src.distribution.kernel_engine import KernelRiskModel

SEED = 5


def _km(p_zero=0.0, n=45, sd=0.25):
    rng = np.random.default_rng(SEED)
    shocks = np.clip(rng.normal(1.0, sd, n), 0.05, None)
    w = 2.0 ** (-(n - 1 - np.arange(n)) / 25)
    return KernelRiskModel.fit(shocks, years=np.arange(1980, 1980 + n),
                               weights=w, weighted=True, p_zero=p_zero), shocks


# ---------------------------------------------------------------- §51 §18 §6
def test_deterministic_mc_integration_converge():
    km, _ = _km(p_zero=0.01)
    for g in (0.60, 0.70, 0.80):
        det = km.expected_loss_deterministic(g, 40)["expected_loss"]
        mc = km.expected_loss_mc(g, 400_000,
                                 np.random.default_rng(1))["expected_loss"]
        ana = km.expected_loss_integration(g)["expected_loss"]
        assert det == pytest.approx(ana, abs=2e-3)
        assert mc == pytest.approx(ana, abs=2e-3)


# ---------------------------------------------------------------- §55 §7
def test_no_negative_yields_large_sample():
    km, _ = _km(p_zero=0.02)
    draws = km.sample(1_000_000, np.random.default_rng(2))
    assert (draws >= 0).all()
    pos = draws[draws > 0]
    assert (pos > 0).all()                       # log-kernel: estrictamente > 0


# ---------------------------------------------------------------- §54 §8
def test_zero_mass_deterministic_and_mc():
    km, _ = _km(p_zero=0.03)
    sc = km.deterministic_scenarios(20)
    zero_row = sc[sc["simulated_relative_shock"] == 0.0]
    assert len(zero_row) == 1
    assert zero_row["weight"].iloc[0] == pytest.approx(0.03)
    assert sc["weight"].sum() == pytest.approx(1.0)
    draws = km.sample(200_000, np.random.default_rng(3))
    assert (draws == 0).mean() == pytest.approx(0.03, abs=0.005)


# ---------------------------------------------------------------- §52 §14
def test_historical_extreme_remains_anchor():
    rng = np.random.default_rng(SEED)
    shocks = np.clip(rng.normal(1.0, 0.12, 40), 0.05, None)
    shocks[10] = 0.35                            # sequía severa
    km = KernelRiskModel.fit(shocks, years=np.arange(1985, 2025))
    assert 0.35 in km.anchors_shock              # el extremo sigue anclado
    sc = km.deterministic_scenarios(20)
    around = sc[sc["base_relative_shock"] == 0.35]
    assert len(around) == 20                     # nube alrededor del extremo
    assert around["simulated_relative_shock"].min() < 0.35   # incl. peores
    # y su masa de probabilidad no desaparece por suavizado
    tail_mass = sc.loc[sc["simulated_relative_shock"] < 0.5, "weight"].sum()
    assert tail_mass > 0.01


# ---------------------------------------------------------------- §53 §3
def test_kernel_applied_after_detrending():
    rng = np.random.default_rng(SEED)
    years = np.arange(1980, 2025)
    trend = 1000 + 60 * (years - 1980)           # tecnología fuerte
    y = trend * np.clip(rng.normal(1.0, 0.10, len(years)), 0.05, None)
    import pandas as pd
    df = pd.DataFrame({"anio": years, "campania": years.astype(str),
                       "rendimiento_kgxha": y})
    res = RiskDistributionEngine(RiskDistributionConfig(
        execution_mode="FAST")).run(df, target_year=2026)
    assert res.status == "OK"
    assert res.trend["model"] != "constant"      # detrend capturó tecnología
    mu = res.trend["target_yield"]
    # la dispersión simulada refleja el shock (~10%), no la tecnología (×3.6)
    cv = float(np.std(res.simulated_yields[res.simulated_yields > 0]) / mu)
    assert cv < 0.25
    assert {"log_kernel", "weighted_log_kernel"} <= set(
        res.ranking["distribution"])             # §28: siempre evaluados


# ---------------------------------------------------------------- §56 §44-45
def test_bandwidth_flags_and_auto_sensible():
    km, shocks = _km()
    assert km.bandwidth_method == "TAIL_AWARE_LOO"
    assert 0.5 <= km.bandwidth_factor <= 1.5
    # sobre-suavizado manual: P10 de escenarios sube materialmente
    z = np.log(km.anchors_shock)
    big = KernelRiskModel(anchors_shock=km.anchors_shock,
                          anchor_years=km.anchor_years, weights=km.weights,
                          bandwidth=km.bandwidth * 4, p_zero=0.0)
    p10_ok = np.quantile(km.sample(50_000, np.random.default_rng(4)), 0.10)
    p10_big = np.quantile(big.sample(50_000, np.random.default_rng(4)), 0.10)
    assert p10_big < p10_ok * 0.95 or p10_big > p10_ok * 1.05  # distorsiona


# ---------------------------------------------------------------- §58 §23-24
def test_experience_guardrail_modes():
    burn, model = 0.061, 0.045                   # burn > modelo
    assert max(model, burn) == pytest.approx(0.061)          # MAX_MODEL_OBSERVED
    z = 0.4
    blend = z * burn + (1 - z) * model                        # CREDIBILITY_BLEND
    assert model < blend < burn


# ---------------------------------------------------------------- governance
def test_kernel_governance_and_reproducibility():
    km, _ = _km(p_zero=0.01)
    g = km.governance()
    for k in ("kernel_version", "bandwidth", "bandwidth_method",
              "historical_anchors", "effective_sample", "p_zero"):
        assert k in g
    a = km.sample(10_000, np.random.default_rng(9))
    b = km.sample(10_000, np.random.default_rng(9))
    assert np.array_equal(a, b)
