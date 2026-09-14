"""Tests de los motores de agro-rate (pytest)."""

import numpy as np
import pandas as pd
import pytest

from src.analytics import detrending
from src.data import database
from src.pricing import burning_cost as bc
from src.pricing.yield_insurance import Coverage, payout
from src.simulation import monte_carlo as mc


@pytest.fixture(scope="module")
def serie():
    df = database.load()
    s = database.series(df, "girasol", "12 de Octubre")
    assert len(s) > 30
    return s


# ------------------------------------------------------------- detrending
@pytest.mark.parametrize("method", detrending.METHODS)
def test_detrend_methods(serie, method):
    det = detrending.detrend(serie, method=method)
    assert {"trend", "detrended", "yield_index"} <= set(det.columns)
    assert det["detrended"].notna().all()
    assert (det["detrended"] >= 0).all()
    # la media detrendeada queda en el orden de la tendencia de referencia
    assert det["detrended"].mean() == pytest.approx(
        det.attrs["reference_trend"], rel=0.35)


def test_detrend_reduces_cv_linear(serie):
    det = detrending.detrend(serie, method="linear")
    cv_raw = serie["rendimiento_kgxha"].std() / serie["rendimiento_kgxha"].mean()
    cv_det = det["detrended"].std() / det["detrended"].mean()
    assert cv_det <= cv_raw  # el trend explica parte de la varianza


# ---------------------------------------------------------------- pricing
def test_payout_shape():
    cov = Coverage(expected_yield=3000, guarantee=0.70, price=0.40)
    y = np.array([0, 1000, 2100, 2500, 4000])
    ind = payout(y, cov)
    assert ind[0] == pytest.approx(2100 * 0.40)  # pérdida total
    assert ind[2] == 0.0                          # justo en el trigger
    assert (ind[y >= cov.guaranteed_yield] == 0).all()
    assert np.all(np.diff(ind) <= 0)              # decreciente en el rinde


def test_payout_deductible_and_limit():
    cov = Coverage(expected_yield=3000, guarantee=0.70, price=0.40,
                   deductible_pct=0.10, max_payout=300.0)
    full = Coverage(expected_yield=3000, guarantee=0.70, price=0.40)
    y = np.array([0.0, 1500.0])
    assert (payout(y, cov) <= 300.0).all()
    assert (payout(y, cov) <= payout(y, full)).all()


def test_burning_cost_consistency(serie):
    det = detrending.detrend(serie, method="linear")
    cov = Coverage(expected_yield=float(det["trend"].iloc[-1]),
                   guarantee=0.70, price=0.40)
    res = bc.burning_cost(det["detrended"], cov)
    assert 0 <= res["burning_cost"] <= 1
    assert 0 <= res["frequency"] <= 1
    assert res["pure_premium_usd_ha"] == pytest.approx(
        res["burning_cost"] * cov.sum_insured)


def test_guarantee_curve_monotonic(serie):
    det = detrending.detrend(serie, method="linear")
    curve = bc.guarantee_curve(det["detrended"],
                               expected_yield=float(det["trend"].iloc[-1]))
    # más garantía → más prima y más frecuencia
    assert curve["pure_premium_rate"].is_monotonic_increasing
    assert curve["frequency"].is_monotonic_increasing


# ------------------------------------------------------------- simulation
@pytest.mark.parametrize("dist", ["empirical bootstrap", "normal",
                                  "lognormal", "gamma", "weibull", "kde"])
def test_simulation_runs(serie, dist):
    det = detrending.detrend(serie, method="linear")
    cov = Coverage(expected_yield=float(det["trend"].iloc[-1]),
                   guarantee=0.70, price=0.40)
    res = mc.simulate(det["detrended"], cov, distribution=dist, n_sim=5000)
    assert res["expected_loss_usd_ha"] >= 0
    assert 0 <= res["probability_of_loss"] <= 1
    assert res["tvar95"] >= res["var95"]
    assert res["gross_premium_usd_ha"] == pytest.approx(
        res["expected_loss_usd_ha"] * 1.35)


def test_simulation_close_to_burning_cost(serie):
    """Bootstrap con n grande ≈ burning cost histórico."""
    det = detrending.detrend(serie, method="linear")
    cov = Coverage(expected_yield=float(det["trend"].iloc[-1]),
                   guarantee=0.70, price=0.40)
    hist = bc.burning_cost(det["detrended"], cov)
    sim = mc.simulate(det["detrended"], cov,
                      distribution="empirical bootstrap", n_sim=200_000)
    assert sim["expected_loss_usd_ha"] == pytest.approx(
        hist["pure_premium_usd_ha"], rel=0.05)
