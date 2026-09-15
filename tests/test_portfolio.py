"""Tests del consolidado de cartera (pantalla Resultados) y loss cap."""

import numpy as np
import pandas as pd
import pytest

from src.pricing import portfolio as pf


def make_nh(years, yields):
    return pd.DataFrame({"year": years, "normalized_yield": yields})


BASE = pf.PortfolioParams(trigger_mode="pct", trigger_value=65.0,
                          sum_insured_ha=200.0, deductions=0.25,
                          mr_margin=0.10, n_sims=5000)


def test_loss_cost_area_yield():
    lc = pf.loss_cost(np.array([1300.0, 650.0, 0.0, 2000.0]), 1300.0)
    assert lc == pytest.approx([0.0, 0.5, 1.0, 0.0])
    capped = pf.loss_cost(np.array([1300.0, 650.0, 0.0]), 1300.0, cap=0.4)
    assert capped == pytest.approx([0.0, 0.4, 0.4])


def test_department_result_loading_and_trigger():
    rng = np.random.default_rng(1)
    years = np.arange(1990, 2025)
    y = np.clip(rng.normal(1500, 400, len(years)), 0, None)
    r = pf.department_result("X", make_nh(years, y), expected_yield=1500.0,
                             superficie_ha=1000.0, params=BASE)
    assert r["trigger_yield"] == pytest.approx(975.0)      # 65% de 1500
    assert r["trigger_pct"] == pytest.approx(0.65)
    assert r["sa_total"] == pytest.approx(200_000.0)
    # recargo: técnica = pura / (1 - 0.25 - 0.10)
    assert r["tech_rate"] == pytest.approx(r["pure_rate"] / 0.65)
    assert r["lc_cat"] >= r["pure_rate"]                   # cola > media
    assert r["perdida_cat"] == pytest.approx(r["lc_cat"] * r["sa_total"])


def test_loss_cap_reduces_tail_more_than_premium():
    rng = np.random.default_rng(2)
    years = np.arange(1980, 2025)
    y = np.clip(rng.normal(1500, 500, len(years)), 0, None)
    nh = make_nh(years, y)
    from dataclasses import replace
    capped = replace(BASE, loss_cap=0.30)
    r0 = pf.department_result("X", nh, 1500.0, 1000.0, BASE)
    r1 = pf.department_result("X", nh, 1500.0, 1000.0, capped)
    assert r1["pure_rate"] <= r0["pure_rate"]
    assert r1["lc_cat"] <= 0.30 + 1e-9                     # cap respetado en CAT
    assert r1["lc_cat"] <= r0["lc_cat"]
    # reducción relativa de la cola >= reducción relativa de la prima
    red_prima = 1 - r1["pure_rate"] / max(r0["pure_rate"], 1e-9)
    red_cat = 1 - r1["lc_cat"] / max(r0["lc_cat"], 1e-9)
    assert red_cat >= red_prima - 1e-9
    # y sin cap, los campos *_nocap coinciden con la corrida base
    assert r1["pure_rate_nocap"] == pytest.approx(r0["pure_rate"])


def test_consolidate_totals_and_worst_year():
    rng = np.random.default_rng(3)
    years = np.arange(1990, 2025)
    res = []
    for i, name in enumerate(["A", "B", "C"]):
        y = np.clip(rng.normal(1500, 350, len(years)), 0, None)
        y[years == 2009] *= 0.2                            # año catastrófico común
        res.append(pf.department_result(name, make_nh(years, y), 1500.0,
                                        1000.0 * (i + 1), BASE))
    cons = pf.consolidate(res, BASE)
    t = cons["table"]
    assert cons["prima_tecnica"] == pytest.approx(t["prima_tecnica"].sum())
    assert cons["sa_total"] == pytest.approx(t["sa_total"].sum())
    assert cons["worst_year"] == 2009
    assert cons["worst_loss"] > 0
    assert cons["pml_x"] == pytest.approx(cons["perdida_cat"] / cons["prima_tecnica"])
    assert set(cons["worst_detail"]["departamento"]) == {"A", "B", "C"}


def test_fixed_trigger_mode():
    from dataclasses import replace
    p = replace(BASE, trigger_mode="fixed", trigger_value=1000.0)
    years = np.arange(2000, 2025)
    y = np.full(len(years), 1200.0)
    r = pf.department_result("X", make_nh(years, y), 1600.0, 500.0, p)
    assert r["trigger_yield"] == pytest.approx(1000.0)
    assert r["pure_rate"] == pytest.approx(0.0)            # nunca gatilla
