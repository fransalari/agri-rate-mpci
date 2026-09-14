"""AGRO RATE — Agricultural Risk & Insurance Analytics (V0.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import detrending, statistics as ystats
from src.normalization import YieldNormalizationConfig, YieldNormalizationEngine
from src.data import database
from src.pricing import burning_cost as bc
from src.pricing.yield_insurance import Coverage
from src.simulation import monte_carlo as mc
from src.simulation.distributions import fit_all

st.set_page_config(page_title="AGRO RATE", page_icon="🌾", layout="wide")


@st.cache_data(show_spinner="Cargando datos...")
def get_data() -> pd.DataFrame:
    return database.load()


df = get_data()

# ---------------------------------------------------------------- sidebar
st.sidebar.title("🌾 AGRO RATE")
st.sidebar.caption("Agricultural Risk & Insurance Analytics")

page = st.sidebar.radio(
    "Módulo",
    ["🏠 Dashboard", "🌱 Análisis de rindes", "📈 Detrending",
     "🧠 Normalización", "💰 Pricing", "🎲 Monte Carlo", "📦 Datos"],
)

cultivos = sorted(df["cultivo"].unique())
_pref = next((i for i, c in enumerate(cultivos)
              if c.casefold() in ("soja total", "girasol", "maíz")), 0)
cultivo = st.sidebar.selectbox("Cultivo", cultivos, index=_pref)
_sub = df.loc[df["cultivo"] == cultivo]
deptos = sorted(_sub["departamento"].unique())
_counts = _sub.groupby("departamento")["anio"].nunique()
_best = _counts.idxmax() if len(_counts) else deptos[0]
depto = st.sidebar.selectbox("Departamento", deptos,
                             index=deptos.index(_best))

anios = df.loc[(df["cultivo"] == cultivo) & (df["departamento"] == depto), "anio"]
a_min, a_max = int(anios.min()), int(anios.max())
desde, hasta = st.sidebar.slider("Período", a_min, a_max, (a_min, a_max))

serie = database.series(df, cultivo, depto, desde, hasta)

method = st.sidebar.selectbox(
    "Detrending",
    ["auto (motor)", "linear", "quadratic", "loess", "moving_average", "none"],
    help="'auto' selecciona modelo, modo, start year y half-life con "
         "validación out-of-sample (Yield Risk Normalization Engine)")

TARGET_YEAR = int(a_max) + 1


@st.cache_resource(show_spinner="Ejecutando motor de normalización...")
def run_engine(cultivo: str, depto: str, desde: int, hasta: int,
               mode: str, n_rows: int):
    data = database.series(get_data(), cultivo, depto, desde, hasta)
    cfg = YieldNormalizationConfig(execution_mode=mode)
    return YieldNormalizationEngine(cfg).run(
        data, crop=cultivo, location=depto, target_year=hasta + 1)


def engine_to_det(res, serie_df):
    nh = res.normalized_history.rename(columns={
        "year": "anio", "observed_yield": "rendimiento_kgxha",
        "historical_trend": "trend", "normalized_yield": "detrended",
        "relative_shock": "yield_index"})
    out = nh.merge(serie_df[["anio", "campania"]], on="anio", how="left")
    out.attrs["reference_year"] = res.target_year
    out.attrs["reference_trend"] = res.expected_yield
    out.attrs["method"] = f"auto: {res.recommended.label()}"
    return out


auto_result = None
if method == "auto (motor)" and len(serie) >= 8:
    auto_result = run_engine(cultivo, depto, desde, hasta, "FAST", len(serie))
    if auto_result.status == "OK":
        det = engine_to_det(auto_result, serie)
        st.sidebar.success(f"AUTO · {auto_result.recommended.trend_model} · "
                           f"conf. {auto_result.selection_confidence}")
    else:
        st.sidebar.warning("Serie insuficiente para el motor; fallback linear")
        det = detrending.detrend(serie, method="linear")
elif len(serie) >= 3 and method != "auto (motor)":
    det = detrending.detrend(serie, method=method)
else:
    det = detrending.detrend(serie, method="linear") if len(serie) >= 3 else serie.assign(
        trend=np.nan, detrended=serie.get("rendimiento_kgxha"), yield_index=np.nan)

st.sidebar.caption(f"Fuente: {df.attrs.get('source', '—')}")

# ---------------------------------------------------------------- helpers

def serie_chart(data: pd.DataFrame, show_trend: bool = False) -> go.Figure:
    fig = px.scatter(data, x="anio", y="rendimiento_kgxha",
                     hover_data=["campania"],
                     labels={"anio": "Campaña", "rendimiento_kgxha": "kg/ha"})
    fig.update_traces(mode="lines+markers")
    if show_trend and "trend" in data.columns:
        fig.add_scatter(x=data["anio"], y=data["trend"],
                        mode="lines", name="Tendencia",
                        line=dict(dash="dash"))
    fig.update_layout(height=420, showlegend=show_trend)
    return fig


if serie.empty:
    st.warning("No hay datos para esa combinación de filtros.")
    st.stop()

# ---------------------------------------------------------------- pages
if page == "🏠 Dashboard":
    st.title("AGRO RATE")
    st.caption(f"{cultivo.title()} · {depto} · Chaco · {desde}–{hasta}")

    stats_ = ystats.describe(serie["rendimiento_kgxha"])
    slope = detrending.trend_slope(serie)
    worst = ystats.worst_years(det, n=1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rinde promedio", f"{stats_['mean']:,.0f} kg/ha")
    c2.metric("Tendencia (kg/ha/año)", f"{slope:+.1f}")
    c3.metric("Coef. de variación", f"{stats_['cv']:.0%}")
    c4.metric("Peor campaña", str(worst.iloc[0, 0]))

    st.plotly_chart(serie_chart(det, show_trend=True), use_container_width=True)

elif page == "🌱 Análisis de rindes":
    st.title("Análisis de rindes")
    st.plotly_chart(serie_chart(serie), use_container_width=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Estadística descriptiva")
        st.dataframe(ystats.describe(serie["rendimiento_kgxha"])
                     .rename("valor").to_frame().style.format("{:,.1f}"))
    with c2:
        st.subheader("Distribución")
        fig = px.histogram(serie, x="rendimiento_kgxha", nbins=20,
                           labels={"rendimiento_kgxha": "kg/ha"})
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Peores campañas")
        st.dataframe(ystats.worst_years(det, n=5))

elif page == "📈 Detrending":
    st.title("Detrending")
    st.caption(f"Método: **{method}** · Año de referencia: "
               f"**{det.attrs.get('reference_year', '—')}**")

    st.plotly_chart(serie_chart(det, show_trend=True), use_container_width=True)

    fig = px.scatter(det, x="anio", y="detrended", hover_data=["campania"],
                     labels={"anio": "Campaña",
                             "detrended": "kg/ha (tecnología actual)"})
    fig.update_traces(mode="lines+markers")
    fig.update_layout(title="Serie detrendeada", height=420)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("CV original", f"{serie['rendimiento_kgxha'].std(ddof=1) / serie['rendimiento_kgxha'].mean():.0%}")
    c2.metric("CV detrendeado", f"{det['detrended'].std(ddof=1) / det['detrended'].mean():.0%}")


elif page == "🧠 Normalización":
    st.title("Yield Risk Normalization Engine")
    st.caption("Selección automática de detrending con validación "
               "out-of-sample · regla one-standard-error")

    mode = st.radio("Modo de ejecución", ["FAST", "FULL"], horizontal=True,
                    help="FAST: interactivo. FULL: grid completo + "
                         "Historical Information Value (governance/pricing).")
    res = run_engine(cultivo, depto, desde, hasta, mode, len(serie))

    if res.status != "OK":
        st.warning("INSUFFICIENT_DATA: la serie no alcanza para modelar "
                   "sin inventar precisión.")
        st.stop()

    rec, hl = res.recommended, res.recommended.half_life_years
    hl_txt = "∞ (pesos iguales)" if hl == float("inf") else f"{hl:g} años"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modelo recomendado", rec.trend_model)
    c2.metric("Detrending", rec.detrending_mode)
    c3.metric("Historia", f"{rec.start_year}–{int(det['anio'].max())}")
    c4.metric("Half-life", hl_txt)
    c1, c2, c3, c4 = st.columns(4)
    ci = res.expected_yield_ci
    c1.metric(f"Rinde esperado {res.target_year}",
              f"{res.expected_yield:,.0f} kg/ha",
              help=f"CI 90%: {ci[0]:,.0f} – {ci[1]:,.0f}")
    c2.metric("N efectivo", f"{res.effective_sample_size:.1f}",
              help=f"status: {res.sample_status}")
    c3.metric("Confianza", res.selection_confidence,
              help=f"estabilidad de selección: {res.selection_stability_pct:.0f}%")
    c4.metric("Score actuarial", f"{res.validation_score:.4f}",
              help=f"± {res.score_se:.4f} (SE) · tail {res.tail_score:.4f}")

    with st.expander("¿Por qué este modelo?", expanded=True):
        for r in res.selection_reason:
            st.write("–", r)
        if res.numerical_champion != res.recommended:
            st.caption(f"Campeón numérico: {res.numerical_champion.label()} "
                       f"(score {res.numerical_champion_score:.4f})")

    nh = res.normalized_history

    # Chart 1 — observado vs trend + target
    fig = go.Figure()
    fig.add_scatter(x=nh["year"], y=nh["observed_yield"],
                    mode="lines+markers", name="Observado")
    fig.add_scatter(x=nh["year"], y=nh["historical_trend"],
                    mode="lines", name="Trend seleccionado",
                    line=dict(dash="dash"))
    fig.add_scatter(x=[res.target_year], y=[res.expected_yield],
                    mode="markers", name=f"E[{res.target_year}]",
                    marker=dict(size=13, symbol="star"))
    fig.update_layout(title="Rinde observado y tendencia tecnológica",
                      height=400, xaxis_title="Campaña", yaxis_title="kg/ha")
    st.plotly_chart(fig, use_container_width=True)

    # Chart 2 — historia normalizada
    fig = px.bar(nh, x="year", y="normalized_yield",
                 title=f"Historia normalizada a tecnología {res.target_year}",
                 labels={"year": "Campaña", "normalized_yield": "kg/ha equivalentes"})
    fig.add_hline(y=res.expected_yield, line_dash="dot",
                  annotation_text="rinde esperado")
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        # Chart 3 — comparación de modelos (mejor score por modelo)
        r = res.model_ranking
        bpm = (r.loc[r.groupby("model")["cv_score"].idxmin()]
               .sort_values("cv_score"))
        colors = ["SELECTED" in s or "champion" in s for s in bpm["status"]]
        fig = px.bar(bpm, x="cv_score", y="model", orientation="h",
                     title="Actuarial Validation Score (mejor por modelo)",
                     labels={"cv_score": "score (menor = mejor)", "model": ""},
                     color=colors, color_discrete_map={True: "#2e7d32",
                                                       False: "#9e9e9e"})
        fig.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        # Chart 5 — peso histórico por año
        fig = px.bar(nh, x="year", y="historical_weight",
                     title="Peso histórico por campaña (half-life)",
                     labels={"year": "Campaña", "historical_weight": "peso relativo"})
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)

    # Chart 4 — HIV (solo FULL)
    if res.historical_information_value is not None and len(res.historical_information_value):
        hiv = res.historical_information_value
        fig = px.bar(hiv, x="period", y="information_value",
                     title="Historical Information Value por bloque "
                           "(positivo = el bloque ayuda)",
                     labels={"period": "", "information_value": "ΔScore sin el bloque"},
                     hover_data=["recommendation"])
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)
    elif mode == "FAST":
        st.caption("Historical Information Value disponible en modo FULL.")

    with st.expander("Diagnósticos actuariales (advanced)"):
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Diagnósticos**")
            st.json(res.diagnostics)
            sb = res.structural_break
            st.write("**Structural break**")
            st.write(f"{sb.confidence}"
                     + (f" · año {sb.year} · p={sb.p_value:.3f} · "
                        f"acción sugerida: {sb.suggested_action}"
                        if sb.detected else " (no detectado)"))
            if sb.note:
                st.caption(sb.note)
        with c2:
            st.write("**Calidad de datos**")
            if res.data_quality:
                st.dataframe(pd.DataFrame(res.data_quality), height=200)
            else:
                st.write("Sin observaciones.")
            if res.warnings:
                st.write("**Warnings**")
                for w in res.warnings:
                    st.warning(w)
        st.write("**Ranking completo**")
        st.dataframe(res.model_ranking, height=300, use_container_width=True)
        st.caption(f"Ejecución: {res.timings.get('total_s','—')} s · "
                   f"{res.timings.get('n_candidates','—')} configuraciones · "
                   f"dataset hash {res.governance.get('dataset_hash','—')} · "
                   f"engine v{res.governance.get('engine_version','—')}")

elif page == "💰 Pricing":
    st.title("Pricing — Yield Shortfall")

    c1, c2, c3 = st.columns(3)
    _default_expected = (auto_result.expected_yield
                         if auto_result is not None and auto_result.status == "OK"
                         else det["trend"].iloc[-1])
    expected = c1.number_input("Rinde esperado (kg/ha)",
                               value=float(round(_default_expected, -1)),
                               step=50.0)
    guarantee = c2.slider("Garantía", 0.50, 0.90, 0.70, 0.05)
    price = c3.number_input("Precio (USD/kg)", value=0.40, step=0.05)

    cov = Coverage(expected_yield=expected, guarantee=guarantee, price=price)
    res = bc.burning_cost(det["detrended"], cov)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Burning Cost", f"{res['burning_cost']:.2%}")
    c2.metric("Prima pura", f"{res['pure_premium_usd_ha']:,.2f} USD/ha")
    c3.metric("Frecuencia", f"{res['frequency']:.1%}")
    c4.metric("Severidad", f"{res['severity']:.1%}")

    st.caption(f"Rinde garantizado: {cov.guaranteed_yield:,.0f} kg/ha · "
               f"Suma asegurada: {cov.sum_insured:,.0f} USD/ha · "
               f"{res['n_years']} campañas")

    st.subheader("Curva de garantías")
    curve = bc.guarantee_curve(det["detrended"], expected, price)
    st.dataframe(curve.style.format({
        "guarantee": "{:.0%}", "guaranteed_yield": "{:,.0f}",
        "pure_premium_rate": "{:.2%}", "frequency": "{:.1%}",
        "severity": "{:.1%}"}))

    st.subheader("Loss cost por campaña")
    tabla = bc.yearly_table(det, cov)
    fig = px.bar(tabla, x="campania", y="loss_cost",
                 labels={"campania": "Campaña", "loss_cost": "Loss cost"})
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

elif page == "🎲 Monte Carlo":
    st.title("Simulación Monte Carlo")

    c1, c2, c3 = st.columns(3)
    _default_expected = (auto_result.expected_yield
                         if auto_result is not None and auto_result.status == "OK"
                         else det["trend"].iloc[-1])
    expected = c1.number_input("Rinde esperado (kg/ha)",
                               value=float(round(_default_expected, -1)),
                               step=50.0)
    guarantee = c2.slider("Garantía", 0.50, 0.90, 0.70, 0.05)
    price = c3.number_input("Precio (USD/kg)", value=0.40, step=0.05)

    c1, c2, c3 = st.columns(3)
    distribution = c1.selectbox(
        "Distribución",
        ["empirical bootstrap", "kde", "normal", "lognormal", "gamma", "weibull"])
    n_sim = c2.select_slider("Simulaciones", [10_000, 50_000, 100_000],
                             value=10_000)
    with c3:
        expense = st.number_input("Gastos", value=0.25, step=0.05)
        margin = st.number_input("Margen de riesgo", value=0.10, step=0.05)

    cov = Coverage(expected_yield=expected, guarantee=guarantee, price=price)
    result = mc.simulate(det["detrended"], cov, distribution=distribution,
                         n_sim=n_sim, expense_ratio=expense, risk_margin=margin)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pérdida esperada", f"{result['expected_loss_usd_ha']:,.2f} USD/ha")
    c2.metric("Prob. de siniestro", f"{result['probability_of_loss']:.1%}")
    c3.metric("VaR 99%", f"{result['var99']:,.0f} USD/ha")
    c4.metric("Prima bruta", f"{result['gross_premium_usd_ha']:,.2f} USD/ha")

    st.dataframe(mc.summary_table(result).style.format({"Valor": "{:,.4f}"}))

    ind = result["indemnities"]
    fig = px.histogram(x=ind[ind > 0], nbins=60,
                       labels={"x": "Indemnización USD/ha"})
    fig.update_layout(title="Distribución de pérdidas (siniestros > 0)",
                      height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Ajuste de distribuciones (AIC)")
    st.dataframe(fit_all(det["detrended"]).drop(columns="params"))

elif page == "📦 Datos":
    st.title("Datos")
    st.write(f"**Fuente activa:** `{df.attrs.get('source', '—')}`")
    st.write(f"{len(df):,} filas · cultivos: {', '.join(cultivos)}")

    st.dataframe(serie, use_container_width=True)
    st.download_button("Descargar serie filtrada (CSV)",
                       serie.to_csv(index=False).encode(),
                       file_name=f"{cultivo}_{depto}.csv")

    st.divider()
    st.subheader("Actualizar desde MAGyP")
    st.caption("Descarga el dataset completo de Estimaciones Agrícolas "
               "(datosestimaciones.magyp.gob.ar) y regenera el Parquet local. "
               "En producción esto lo hace la GitHub Action mensual.")
    if st.button("Descargar dataset completo"):
        from src.data import magyp
        with st.spinner("Descargando Estimaciones.csv..."):
            try:
                csv_path = magyp.download_full_dataset()
                full = magyp.load_dataset(csv_path)
                magyp.to_parquet(database._normalize(full), database.PARQUET)
                st.success(f"OK: {len(full):,} filas → {database.PARQUET}")
                st.cache_data.clear()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Falló la descarga: {exc}")
