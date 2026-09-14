"""AGRO RATE — Agricultural Risk & Insurance Analytics (V0.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import detrending, statistics as ystats
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
     "💰 Pricing", "🎲 Monte Carlo", "📦 Datos"],
)

cultivos = sorted(df["cultivo"].unique())
cultivo = st.sidebar.selectbox("Cultivo", cultivos)
deptos = sorted(df.loc[df["cultivo"] == cultivo, "departamento"].unique())
depto = st.sidebar.selectbox("Departamento", deptos)

anios = df.loc[(df["cultivo"] == cultivo) & (df["departamento"] == depto), "anio"]
a_min, a_max = int(anios.min()), int(anios.max())
desde, hasta = st.sidebar.slider("Período", a_min, a_max, (a_min, a_max))

serie = database.series(df, cultivo, depto, desde, hasta)

method = st.sidebar.selectbox(
    "Detrending", ["linear", "quadratic", "loess", "moving_average", "none"])
det = detrending.detrend(serie, method=method) if len(serie) >= 3 else serie.assign(
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

elif page == "💰 Pricing":
    st.title("Pricing — Yield Shortfall")

    c1, c2, c3 = st.columns(3)
    expected = c1.number_input("Rinde esperado (kg/ha)",
                               value=float(round(det["trend"].iloc[-1], -1)),
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
    expected = c1.number_input("Rinde esperado (kg/ha)",
                               value=float(round(det["trend"].iloc[-1], -1)),
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
