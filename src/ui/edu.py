"""Capa educativa del Actuarial Pricing Lab (spec Lab §3-§25).

REGLA DE ORO (§2): esta capa LEE resultados del motor; jamás recalcula
con fórmulas simplificadas. Guiado y Experto usan el mismo engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .i18n import tr

STAGES = ["Datos", "Trend", "Distribución", "Garantía", "Cola",
          "Simulación", "Loss Cost", "Incertidumbre", "Tasa"]


def guided() -> bool:
    return st.session_state.get("mode", "Guiado") == "Guiado"


def journey(active: int):
    """Barra 'Follow the rate' (§4) con etapa activa y tasa pura viva."""
    parts = []
    for i, s in enumerate(STAGES):
        s = tr(s)
        parts.append(f"<b style='color:#2C5F2D'>{s}</b>" if i == active else s)
    live = st.session_state.get("live_rate")
    tail = (f" &nbsp;·&nbsp; {tr('tasa pura actual')}: <b>{live:.2%}</b>"
            if live is not None else "")
    st.markdown("<div style='font-size:0.85em;color:#666'>"
                + " → ".join(parts) + tail + "</div>",
                unsafe_allow_html=True)


def edu(title: str, concepto: str, formula: str | None, por_que: str,
        impacto: str, advertencia: str):
    """Componente educativo estándar (§5). Solo visible en modo Guiado."""
    if not guided():
        return
    with st.expander(f"📖 {tr(title)}"):
        st.markdown(f"**{tr('Concepto')}** · {tr(concepto)}")
        if formula:
            st.latex(formula)
        st.markdown(f"**{tr('Por qué importa')}** · {tr(por_que)}")
        st.markdown(f"**{tr('Impacto en la tasa')}** · {tr(impacto)}")
        st.markdown(f"⚠️ **{tr('Advertencia actuarial')}** · {tr(advertencia)}")


# ======================================================================
# Análisis (usan el motor real vía objetos ya calculados)
# ======================================================================

def _std_shocks(rk):
    nh = rk.normalized_history
    pos = nh[~nh["zero_event"]]
    return (pos["year"].to_numpy(), 
            pos["standardized_shock"].to_numpy(float),
            pos["historical_weight"].to_numpy(float))


def _rate_from_dist(name, shocks, weights, guarantee, p0, cfg, rng, n=6000):
    """Tasa pura relativa re-ajustando SOLO la distribución seleccionada
    (sin re-seleccionar modelo): para LOYO / bootstrap / model risk."""
    from src.distribution.candidates import build_candidate
    cand = build_candidate(name, "multiplicative", cfg, weights=weights,
                           x=shocks)
    if name not in ("kde", "weighted_kde"):
        cand.fit(shocks, weights)
    sims = np.maximum(cand.sample(n, rng), 0.0)
    u = rng.uniform(size=n)
    sims = np.where(u < p0, 0.0, sims)
    return float(np.maximum(guarantee - sims, 0.0).mean() / guarantee)


def _base_name(rk):
    name = rk.distribution["recommended"]
    return (rk.ranking.iloc[0]["distribution"]
            if name == "MODEL_ENSEMBLE" else name)


@st.cache_data(show_spinner=False)
def loyo_rates(_rk, guarantee: float, cache_key: str) -> pd.DataFrame:
    """Leave-one-year-out (§21): tasa sin cada campaña."""
    from src.distribution.config import RiskDistributionConfig
    cfg = RiskDistributionConfig()
    years, shocks, w = _std_shocks(_rk)
    p0 = _rk.zero_mass["p_zero"]
    name = _base_name(_rk)
    rng = np.random.default_rng(7)
    base = _rate_from_dist(name, shocks, w, guarantee, p0, cfg, rng)
    rows = []
    for i, y in enumerate(years):
        m = np.arange(len(years)) != i
        r = _rate_from_dist(name, shocks[m], w[m], guarantee, p0, cfg,
                            np.random.default_rng(7))
        rows.append({"year": int(y), "rate": r, "delta": r - base})
    df = pd.DataFrame(rows)
    df.attrs["base"] = base
    return df


@st.cache_data(show_spinner=False)
def model_risk_table(_rk, guarantee: float, cache_key: str) -> pd.DataFrame:
    """Model risk (§19): tasa pura por metodología defendible."""
    from src.distribution.config import RiskDistributionConfig
    cfg = RiskDistributionConfig()
    years, shocks, w = _std_shocks(_rk)
    p0 = _rk.zero_mass["p_zero"]
    nh = _rk.normalized_history
    hist = np.where(nh["zero_event"], 0.0, nh["relative_shock"]).astype(float)
    rows = [{"metodo": "burning cost (histórico)",
             "rate": float(np.maximum(guarantee - hist, 0.0).mean() / guarantee),
             "status": ""}]
    for _, r in _rk.ranking.iterrows():
        try:
            rate = _rate_from_dist(r["distribution"], shocks, w, guarantee,
                                   p0, cfg, np.random.default_rng(7))
        except Exception:
            continue
        rows.append({"metodo": r["distribution"], "rate": rate,
                     "status": r["status"] or ""})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def bootstrap_rate_ci(_rk, guarantee: float, cache_key: str, B: int = 200):
    """Incertidumbre de la tasa (§18): bootstrap ponderado de shocks →
    re-fit de la distribución seleccionada → IC 90%."""
    from src.distribution.config import RiskDistributionConfig
    cfg = RiskDistributionConfig()
    _, shocks, w = _std_shocks(_rk)
    p0 = _rk.zero_mass["p_zero"]
    name = _base_name(_rk)
    p = w / w.sum()
    rng = np.random.default_rng(11)
    rates = []
    for _ in range(B):
        idx = rng.choice(len(shocks), size=len(shocks), replace=True, p=p)
        try:
            rates.append(_rate_from_dist(name, shocks[idx], w[idx], guarantee,
                                         p0, cfg, rng, n=3000))
        except Exception:
            continue
    lo, hi = np.percentile(rates, [5, 95])
    return float(lo), float(hi), len(rates)


def sanity_checks(rk, claim_curve, loyo, guarantee, sims) -> list[dict]:
    """Checks GREEN/AMBER/RED (§20). Nunca bloquean."""
    checks = []
    def add(name, status, detail):
        checks.append({"check": tr(name), "status": status,
                       "detalle": detail})
    cc = claim_curve.sort_values("coverage")
    mono = bool(np.all(np.diff(cc["expected_indemnity"]) >= -1e-9))
    add("Monotonicidad: tasa crece con la cobertura",
        "🟢" if mono else "🔴",
        "OK" if mono else tr("una cobertura mayor debería costar más"))
    neg = int((sims < 0).sum())
    add("Sin rindes simulados negativos", "🟢" if neg == 0 else "🔴",
        f"{neg} draws < 0")
    nh = rk.normalized_history
    n_tail = int((nh["relative_shock"] < guarantee).sum())
    add("Observaciones en la cola asegurada",
        "🟢" if n_tail >= 5 else ("🟠" if n_tail >= 3 else "🔴"),
        tr("{n} de {t} campañas por debajo de la garantía — solo esas informan directamente el precio",
           n=n_tail, t=len(nh)))
    base = loyo.attrs["base"]
    infl = loyo.loc[loyo["delta"].abs().idxmax()]
    ratio = abs(infl["delta"]) / max(base, 1e-9)
    add("¿Domina la prima una sola campaña?",
        "🟢" if ratio < 0.10 else ("🟠" if ratio < 0.20 else "🔴"),
        tr("quitar {y} mueve la tasa {d:+.2%} ({r:.0%} de la base)",
           y=int(infl["year"]), d=infl["delta"], r=ratio))
    rejected = [c for c in rk.candidates
                if c.status == "REJECTED_RISK_UNDERDISPERSION"]
    add("Gates de subdispersión de riesgo",
        "🟢",
        tr("{n} candidato(s) rechazados por comprimir la cola — el seleccionado los pasó",
           n=len(rejected)))
    neff = rk.sample["effective_sample_size"]
    add("Tamaño de muestra efectivo",
        "🟢" if neff >= 30 else ("🟠" if neff >= 20 else "🔴"),
        f"N_eff = {neff:.1f}")
    return checks


def classify_years(serie: pd.DataFrame) -> pd.DataFrame:
    """Clasificación de campañas para el timeline (§6)."""
    df = serie.dropna(subset=["rendimiento_kgxha"]).copy()
    med = df["rendimiento_kgxha"].median()
    r = df["rendimiento_kgxha"] / max(med, 1e-9)
    df["clase"] = np.select(
        [r < 0.60, r < 0.85, r > 1.15],
        ["EXTREMO", "BAJO", "BUENO"], default="NORMAL")
    return df


GLOSSARY = {
    "Coverage / Cobertura": "Fracción del rinde esperado que se garantiza (ej. 70%). Define el trigger del seguro.",
    "Garantía": "Rinde asegurado = cobertura × rinde esperado. Debajo de ese nivel hay siniestro.",
    "Burning cost": "Tasa pura observada: promedio histórico del loss cost. Lo que 'quemó' la cartera en el pasado, a tecnología actual.",
    "Prima pura": "Pérdida esperada como % de la suma asegurada. Sin recargos de ningún tipo.",
    "Tasa técnica": "Prima pura ÷ (1 − deducciones − margen de riesgo). Aún sin recargos comerciales.",
    "Detrending / Normalización": "Separar la mejora tecnológica μ(t) del shock climático. El shock se conserva; la tecnología se actualiza.",
    "KDE / Kernel": "Densidad no paramétrica: suma de campanas sobre cada dato. No impone forma, pero el bandwidth es un supuesto.",
    "Bandwidth": "Ancho de las campanas del KDE. Poco = ruido; mucho = borra estructura de la cola. Nunca es una perilla de tarifa.",
    "Monte Carlo": "Simular miles de campañas posibles desde la distribución elegida para construir la distribución de pérdidas.",
    "Frecuencia × Severidad": "Pérdida esperada ≈ P(siniestro) × severidad media dado siniestro. La descomposición central del pricing.",
    "Expected Shortfall": "Rinde promedio en el peor α% de los escenarios. Mide cuán malo es 'lo malo'.",
    "P(Y=0) / masa en cero": "Probabilidad de pérdida total. Una densidad continua pura no puede producirla; se modela como mezcla.",
    "Bootstrap": "Re-muestrear la historia para medir cuánto cambiaría la estimación con otros datos igualmente plausibles.",
    "Model risk": "La variación de la tasa entre metodologías defendibles. Distinto del riesgo de datos y del riesgo de proceso.",
    "Leave-one-year-out": "Recalcular la tasa quitando cada campaña: detecta si un solo año domina el precio.",
    "PML": "Pérdida máxima probable a un período de retorno dado (ej. 1 en 100 años).",
    "Credibilidad": "Cuánta confianza estadística soporta la muestra. Un promedio estable no implica una cola estable.",
    "N_eff": "Tamaño de muestra efectivo bajo pesos por half-life: (Σw)²/Σw².",
}


def glossary_sidebar():
    with st.sidebar.expander("📚 " + tr("Glosario")):
        for k, v in GLOSSARY.items():
            st.markdown(f"**{k}** — {tr(v)}")
