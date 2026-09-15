"""AGRO RATE — Agricultural Risk & Insurance Analytics (V0.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import detrending, statistics as ystats
from src.normalization import YieldNormalizationConfig, YieldNormalizationEngine
from src.pricing import portfolio as pf
from src.distribution import RiskDistributionConfig, RiskDistributionEngine
from src.ui.i18n import tr
from src.ui import edu
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

lang = st.sidebar.radio("🌐", ["ES", "EN"], horizontal=True,
                        label_visibility="collapsed")
st.session_state["lang"] = lang.lower()
st.session_state["mode"] = st.sidebar.radio(
    tr("Modo"), ["Guiado", "Experto"], horizontal=True, key="mode_sel",
    help=tr("Guiado agrega la capa educativa (conceptos, fórmulas, experimentos). Experto la oculta. El cálculo es EXACTAMENTE el mismo."))

edu.glossary_sidebar()

page = st.sidebar.radio(
    tr("Módulo"),
    ["📊 Resultados — cartera", "1️⃣ Datos", "2️⃣ Exploración",
     "3️⃣ Normalización (trend)", "4️⃣ Riesgo + Monte Carlo", "5️⃣ Pricing"],
    format_func=tr, key="nav_page",
)
st.sidebar.caption(tr("Pipeline: 1 datos → 2 exploración → 3 trend/normalización → 4 distribución de riesgo y MC → 5 pricing. Resultados consolida la cartera."))

cultivos = sorted(df["cultivo"].unique())
_pref = next((i for i, c in enumerate(cultivos)
              if c.casefold() in ("soja total", "girasol", "maíz")), 0)
cultivo = st.sidebar.selectbox(tr("Cultivo"), cultivos, index=_pref, key="f_cultivo")
_sub = df.loc[df["cultivo"] == cultivo]
deptos = sorted(_sub["departamento"].unique())
_counts = _sub.groupby("departamento")["anio"].nunique()
_best = _counts.idxmax() if len(_counts) else deptos[0]
depto = st.sidebar.selectbox(tr("Departamento"), deptos,
                             index=deptos.index(_best), key="f_depto")

anios = df.loc[(df["cultivo"] == cultivo) & (df["departamento"] == depto), "anio"]
a_min, a_max = int(anios.min()), int(anios.max())
desde, hasta = st.sidebar.slider(tr("Período"), a_min, a_max, (a_min, a_max), key="f_periodo")

serie = database.series(df, cultivo, depto, desde, hasta)

method = st.sidebar.selectbox(
    tr("Detrending"),
    ["auto (motor)", "linear", "quadratic", "loess", "kernel_regression", "moving_average", "none"],
    help=tr("'auto' selecciona modelo, modo, start year y half-life con validación out-of-sample (Yield Risk Normalization Engine)"), key="f_det")

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


@st.cache_resource(show_spinner="Ejecutando Yield Risk Distribution Engine...")
def run_risk_engine(cultivo: str, depto: str, desde_: int, hasta_: int,
                    level: str, mode_exec: str, n_rows: int):
    data = database.series(get_data(), cultivo, depto, desde_, hasta_)
    eng = RiskDistributionEngine(RiskDistributionConfig(execution_mode=mode_exec))
    return eng.run(data, crop=cultivo, location=depto,
                   aggregation_level=level, target_year=hasta_ + 1)


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
    st.warning(tr("No hay datos para esa combinación de filtros."))
    st.stop()

# ---------------------------------------------------------------- pages

@st.cache_resource(show_spinner=False)
def compute_portfolio(cultivo: str, provincia: str, desde: int, hasta: int,
                      min_years: int, trigger_mode: str, trigger_value: float,
                      sa_ha: float, deductions: float, margin: float,
                      loss_cap, return_period: int, uniform: bool, n_rows: int):
    """Motor (FAST) + tarificación por departamento y consolidación.

    uniform=True → metodología uniforme de cartera: metodología MODAL
    (modelo de trend + modo) entre las recomendaciones AUTO de la
    provincia, re-estimada con grilla restringida en los deptos que
    diferían. Parámetros (start year, half-life) siguen siendo locales.
    """
    from collections import Counter
    data = get_data()
    dfc = data[(data["cultivo"] == cultivo) & (data["provincia"] == provincia)
               & (data["anio"] >= desde) & (data["anio"] <= hasta)]
    params = pf.PortfolioParams(
        trigger_mode=trigger_mode, trigger_value=trigger_value,
        sum_insured_ha=sa_ha, deductions=deductions, mr_margin=margin,
        loss_cap=loss_cap, return_period=return_period)
    counts = (dfc.dropna(subset=["rendimiento_kgxha"])
              .groupby("departamento")["anio"].nunique())
    deptos = sorted(counts[counts >= min_years].index)
    skipped, autos = [], {}
    prog = st.progress(0.0, text="Ejecutando motor por departamento...")
    eng = YieldNormalizationEngine(YieldNormalizationConfig(execution_mode="FAST"))

    def _serie(d):
        return (dfc[dfc["departamento"] == d]
                .dropna(subset=["rendimiento_kgxha"])
                .sort_values("anio").reset_index(drop=True))

    for i, d in enumerate(deptos):
        prog.progress((i + 1) / max(len(deptos), 1),
                      text=f"Normalizando {d} ({i+1}/{len(deptos)})")
        serie_d = _serie(d)
        try:
            res = eng.run(serie_d, crop=cultivo, location=d,
                          target_year=int(serie_d["anio"].max()) + 1)
        except Exception as exc:
            skipped.append((d, f"motor: {exc}")); continue
        if res.status != "OK":
            skipped.append((d, res.status)); continue
        if pf.superficie_estimada(dfc, d) <= 0:
            skipped.append((d, "sin superficie reciente")); continue
        autos[d] = res

    def _consolidate(res_map):
        rows = []
        for d, r in res_map.items():
            dr = pf.department_result(d, r.normalized_history,
                                      r.expected_yield,
                                      pf.superficie_estimada(dfc, d), params)
            dr["modelo"] = (f"{r.recommended.trend_model} · "
                            f"{r.recommended.detrending_mode[:4]}")
            rows.append(dr)
        return pf.consolidate(rows, params) if rows else None

    cons_auto = _consolidate(autos)
    cons_uni, modal, changed = None, None, []
    if uniform and autos:
        combos = Counter((r.recommended.trend_model,
                          r.recommended.detrending_mode)
                         for r in autos.values())
        modal = combos.most_common(1)[0][0]
        cfg_u = YieldNormalizationConfig(
            execution_mode="FAST", candidate_models=(modal[0],),
            candidate_detrending_modes=(modal[1],))
        eng_u = YieldNormalizationEngine(cfg_u)
        unis = dict(autos)
        diff = [d for d, r in autos.items()
                if (r.recommended.trend_model,
                    r.recommended.detrending_mode) != modal]
        for i, d in enumerate(diff):
            prog.progress((i + 1) / max(len(diff), 1),
                          text=f"Re-estimando con metodología uniforme: {d}")
            try:
                r_u = eng_u.run(_serie(d), crop=cultivo, location=d,
                                target_year=int(_serie(d)["anio"].max()) + 1)
                if r_u.status == "OK":
                    unis[d] = r_u
                    changed.append(d)
            except Exception:
                pass
        cons_uni = _consolidate(unis)
    prog.empty()
    return cons_auto, cons_uni, modal, changed, skipped, params


if page == "📊 Resultados — cartera":
    edu.journey(8)
    st.title(tr("Resultados — consolidado de cartera"))
    st.caption(tr("Tarifación técnica propia (modelo area-yield sobre serie normalizada) · escenario CAT · peor año histórico · loss cap"))

    st.caption(tr("**Del panel lateral aplican:** cultivo (**{c}**) y período (**{d}–{h}**). El departamento del panel no aplica acá: se consolidan todos los elegibles de la provincia seleccionada.", c=cultivo, d=desde, h=hasta))
    data = get_data()
    provincias = sorted(data.loc[data["cultivo"] == cultivo, "provincia"].unique())
    _prov_default = data.loc[(data["cultivo"] == cultivo)
                             & (data["departamento"] == depto), "provincia"]
    _prov_idx = (provincias.index(_prov_default.iloc[0])
                 if len(_prov_default) and _prov_default.iloc[0] in provincias else 0)

    _sv = st.session_state.get("pf_saved", {})   # última selección del usuario
    with st.form("portfolio_form"):
        c1, c2, c3, c4 = st.columns(4)
        _pidx = (provincias.index(_sv["provincia"])
                 if _sv.get("provincia") in provincias else _prov_idx)
        provincia = c1.selectbox(tr("Provincia"), provincias, index=_pidx)
        min_years = c2.number_input(tr("Mín. campañas por depto"), 10, 50,
                                    int(_sv.get("min_years", 25)))
        _tmodes = ["% del rinde esperado", "rinde fijo (kg/ha)"]
        trigger_mode_lbl = c3.selectbox(
            tr("Modo de garantía"), _tmodes, format_func=tr,
            index=_tmodes.index(_sv.get("tmode", _tmodes[0])))
        trigger_value = c4.number_input(
            tr("Valor de garantía"), 10.0, 5000.0,
            float(_sv.get("tval", 65.0 if trigger_mode_lbl.startswith("%")
                          else 1000.0)), step=5.0,
            help="65 ⇒ trigger = 65% del E[rinde] del motor · o kg/ha fijos")
        c1, c2, c3, c4 = st.columns(4)
        sa_ha = c1.number_input(tr("Suma asegurada (USD/ha)"), 10.0, 5000.0,
                                float(_sv.get("sa", 200.0)), step=10.0)
        deductions = c2.number_input("Deductions", 0.0, 0.6,
                                     float(_sv.get("ded", 0.25)), step=0.01)
        margin = c3.number_input("MR Margin", 0.0, 0.5,
                                 float(_sv.get("mr", 0.10)), step=0.01)
        _rps = [50, 100, 200, 250]
        ret_period = c4.selectbox(tr("Período de retorno CAT"), _rps,
                                  index=_rps.index(int(_sv.get("rp", 100))))
        c1, c2 = st.columns([1, 3])
        uniform = c1.checkbox(
            tr("Metodología uniforme de cartera"),
            help=tr("Aplica la metodología modal de la provincia (modelo de trend + modo) a todos los departamentos, re-estimando parámetros localmente. Compara la prima contra AUTO por departamento."))
        cap_on = c1.checkbox(tr("Aplicar LOSS CAP"))
        cap_pct = c2.slider(tr("Loss cap (% de la suma asegurada)"), 10, 100, 50,
                            disabled=not cap_on)
        run = st.form_submit_button(tr("Calcular consolidado"), type="primary")

    if not run and "pf_result2" not in st.session_state:
        st.info(tr("Configurá los parámetros y presioná **Calcular consolidado**. El motor de normalización corre por cada departamento elegible (~2 s c/u la primera vez; después queda cacheado)."))
        st.stop()
    if run:
        st.session_state["pf_saved"] = {
            "provincia": provincia, "min_years": int(min_years),
            "tmode": trigger_mode_lbl, "tval": float(trigger_value),
            "sa": float(sa_ha), "ded": float(deductions),
            "mr": float(margin), "rp": int(ret_period)}
        loss_cap = cap_pct / 100.0 if cap_on else None
        tm = "pct" if trigger_mode_lbl.startswith("%") else "fixed"
        st.session_state["pf_result2"] = compute_portfolio(
            cultivo, provincia, int(desde), int(hasta),
            int(min_years), tm, float(trigger_value),
            float(sa_ha), float(deductions), float(margin), loss_cap,
            int(ret_period), bool(uniform), len(data))
    cons, cons_uni, modal_meth, changed_deptos, skipped, params = st.session_state["pf_result2"]

    if cons is None:
        st.warning(tr("Ningún departamento elegible con esos filtros."))
        st.stop()

    t = cons["table"]
    cap_active = params.loss_cap is not None

    st.subheader(tr("Totales de cartera"))
    st.caption(tr("Base de la tasa: **burning cost histórico** sobre la serie normalizada (trigger {t}); el CAT sí es simulado. La página Pricing muestra la **tasa del modelo** a la garantía que elijas — por eso pueden diferir: distinta cobertura, distinta base (observado vs simulado) y cartera vs departamento individual.",
                  t=(f"{trigger_value:.0f}%" if trigger_mode_lbl.startswith("%")
                     else f"{trigger_value:,.0f} kg/ha")))
    st.caption(tr("Serie utilizada: campañas {d}–{h} · triggers sobre E[rinde] estimado con esa ventana.", d=desde, h=hasta))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tr("Prima técnica total"), f"${cons['prima_tecnica']:,.0f}",
              delta=(f"{cons['prima_tecnica']-cons['prima_tecnica_nocap']:+,.0f} vs sin cap"
                     if cap_active else None), delta_color="inverse")
    c2.metric(tr("Suma asegurada total"), f"${cons['sa_total']:,.0f}",
              help=f"{cons['superficie']:,.0f} ha × {params.sum_insured_ha:,.0f} USD/ha")
    c3.metric(tr("Tasa técnica media"), f"{cons['tasa_media']:.2%}")
    c4.metric(tr("Departamentos"), f"{len(t)}",
              help=f"excluidos: {len(skipped)}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tr("Pérdida CAT (RP {rp} años)", rp=params.return_period),
              f"${cons['perdida_cat']:,.0f}",
              delta=(f"{cons['perdida_cat']-cons['perdida_cat_nocap']:+,.0f} vs sin cap"
                     if cap_active else None), delta_color="inverse")
    c2.metric(tr("Siniestralidad CAT (siniestros/prima)"),
              f"{cons['pml_x']:.0%}",
              delta=(f"{cons['pml_x']-cons['pml_x_nocap']:+.0%} vs sin cap"
                     if cap_active else None), delta_color="inverse",
              help=f"equivale a PML {cons['pml_x']:.1f}× la prima técnica")
    if cons["worst_year"]:
        c3.metric(tr("Peor año histórico ({y})", y=cons['worst_year']),
                  f"${cons['worst_loss']:,.0f}")
        c4.metric(tr("Siniestralidad {y}", y=cons['worst_year']),
                  f"{cons['worst_x']:.0%}",
                  help=f"equivale a {cons['worst_x']:.1f}× la prima técnica")

    if cap_active:
        st.success(
            f"**Efecto del loss cap {params.loss_cap:.0%}:** "
            f"prima técnica ${cons['prima_tecnica_nocap']:,.0f} → "
            f"${cons['prima_tecnica']:,.0f} "
            f"({cons['prima_tecnica']/max(cons['prima_tecnica_nocap'],1)-1:+.1%}) · "
            f"pérdida CAT ${cons['perdida_cat_nocap']:,.0f} → "
            f"${cons['perdida_cat']:,.0f} "
            f"({cons['perdida_cat']/max(cons['perdida_cat_nocap'],1)-1:+.1%}) · "
            f"siniestralidad CAT {cons['pml_x_nocap']:.0%} → {cons['pml_x']:.0%}. "
            "El cap recorta la cola (CAT/PML) mucho más que la prima: "
            "es capacidad de reaseguro implícita.")

    if cons_uni is not None and modal_meth is not None:
        st.subheader(tr("Metodología uniforme vs AUTO por departamento"))
        st.caption(tr("Metodología modal aplicada a toda la provincia: **{m}** · departamentos re-estimados: {n} ({lst})",
                      m=f"{modal_meth[0]} · {modal_meth[1]}",
                      n=len(changed_deptos),
                      lst=", ".join(changed_deptos) if changed_deptos else "—"))
        u1, u2, u3 = st.columns(3)
        u1.metric(tr("Prima técnica (uniforme)"),
                  f"${cons_uni['prima_tecnica']:,.0f}",
                  delta=f"{cons_uni['prima_tecnica']-cons['prima_tecnica']:+,.0f} vs AUTO",
                  delta_color="off")
        u2.metric(tr("Pérdida CAT (uniforme)"),
                  f"${cons_uni['perdida_cat']:,.0f}",
                  delta=f"{cons_uni['perdida_cat']-cons['perdida_cat']:+,.0f} vs AUTO",
                  delta_color="off")
        u3.metric(tr("Tasa técnica media (uniforme)"),
                  f"{cons_uni['tasa_media']:.2%}",
                  delta=f"{cons_uni['tasa_media']-cons['tasa_media']:+.2%} vs AUTO",
                  delta_color="off")
        st.caption(tr("AUTO sigue siendo la referencia (respeta trends locales con evidencia fuerte); la uniforme sirve como sensibilidad de governance y para discutir con reaseguro."))

    st.subheader(tr("Detalle por departamento"))
    show = pd.DataFrame({
        "Departamento": t["departamento"],
        "E[rinde] kg/ha": t["expected_yield"].round(0),
        "Trigger %": (t["trigger_pct"] * 100).round(1),
        "Rinde gatillo": t["trigger_yield"].round(0),
        "Superficie ha": t["superficie_ha"].round(1),
        "SA total USD": t["sa_total"].round(0),
        "Tasa pura": (t["pure_rate"] * 100).round(2),
        "Tasa técnica": (t["tech_rate"] * 100).round(2),
        "Modelo": t.get("modelo", ""),
        "Prima técnica USD": t["prima_tecnica"].round(0),
        f"LC CAT {params.return_period}a %": (t["lc_cat"] * 100).round(1),
        "Pérdida CAT USD": t["perdida_cat"].round(0),
        "LR CAT %": (t["perdida_cat"] / t["prima_tecnica"].clip(lower=1) * 100).round(0),
        "Campañas": t["n_years"],
    })
    if cap_active:
        show["Tasa técnica s/cap"] = (t["tech_rate_nocap"] * 100).round(2)
    tot = {"Departamento": "TOTAL", "Superficie ha": show["Superficie ha"].sum(),
           "SA total USD": show["SA total USD"].sum(),
           "Tasa técnica": round(cons["tasa_media"] * 100, 2),
           "Prima técnica USD": show["Prima técnica USD"].sum(),
           "Pérdida CAT USD": show["Pérdida CAT USD"].sum(),
           "LR CAT %": round(cons["pml_x"] * 100)}
    show = pd.concat([show, pd.DataFrame([tot])], ignore_index=True)
    st.dataframe(show, use_container_width=True, height=560)
    st.download_button(tr("⬇ Descargar CSV"), show.to_csv(index=False).encode(),
                       file_name=f"resultados_{cultivo}_{provincia}.csv")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(t.sort_values("prima_tecnica"), x="prima_tecnica",
                     y="departamento", orientation="h",
                     title=tr("Prima técnica por departamento (USD)"),
                     labels={"prima_tecnica": "USD", "departamento": ""})
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        unidad = st.radio(tr("Unidad del gráfico histórico"),
                          ["% Loss Ratio (siniestros/prima)", "USD"],
                          format_func=tr,
                          horizontal=True, label_visibility="collapsed")
        py = cons["per_year"].copy()
        prima = max(cons["prima_tecnica"], 1e-9)
        prima_nocap = max(cons["prima_tecnica_nocap"], 1e-9)
        as_lr = unidad.startswith("%")
        if as_lr:
            py["v"] = py["loss"] / prima * 100
            py["v_nocap"] = py["loss_nocap"] / prima_nocap * 100
            ylab, tit = "% de la prima técnica", "Siniestralidad histórica por campaña (a valores actuales)"
        else:
            py["v"] = py["loss"]
            py["v_nocap"] = py["loss_nocap"]
            ylab, tit = "USD", "Pérdida histórica agregada por campaña (USD, a valores actuales)"
        if cap_active:
            fig = go.Figure()
            fig.add_bar(x=py["year"], y=py["v_nocap"], name=tr("sin cap"),
                        marker_color="#bcd3ae")
            fig.add_bar(x=py["year"], y=py["v"],
                        name=tr("con cap {c:.0%}", c=params.loss_cap),
                        marker_color="#2e7d32")
            fig.update_layout(barmode="overlay", title=tit,
                              xaxis_title="Campaña", yaxis_title=ylab,
                              legend=dict(orientation="h", y=1.08))
        else:
            fig = px.bar(py, x="year", y="v", title=tit,
                         labels={"year": "Campaña", "v": ylab})
        if as_lr:
            fig.add_hline(y=100, line_dash="dot", line_color="#B85042",
                          annotation_text=tr("LR 100% (siniestros = prima)"))
        if cons["worst_year"]:
            wv = (cons["worst_loss"] / prima * 100) if as_lr else cons["worst_loss"]
            fig.add_annotation(x=cons["worst_year"], y=wv,
                               text=tr("peor año: {y}", y=cons['worst_year']),
                               showarrow=True, arrowhead=2)
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)
        share = (py["sa"] / max(cons["sa_total"], 1e-9)).clip(lower=1e-9)
        lr_media = float((py["loss"] / (prima * share)).mean())
        st.caption(f"Siniestralidad media histórica (base cartera reportante "
                   f"en cada campaña): {lr_media:.0%} de la prima técnica — "
                   f"teórico ≈ {1-params.deductions-params.mr_margin:.0%} por "
                   "construcción del recargo; difiere por mezcla de "
                   "departamentos/campañas.")

    if cons["worst_detail"] is not None:
        with st.expander(tr("Detalle del peor año ({y})", y=cons['worst_year'])):
            wd = cons["worst_detail"].copy()
            primas = t.set_index("departamento")["prima_tecnica"]
            wd["lr"] = (wd["loss"] / wd["departamento"].map(primas).clip(lower=1) * 100).round(0)
            wd["lc"] = (wd["lc"] * 100).round(1)
            wd["loss"] = wd["loss"].round(0)
            wd = wd[["departamento", "lc", "loss", "lr"]]
            wd.columns = ["Departamento", "LC %", "Pérdida USD", "LR % (vs prima depto)"]
            st.dataframe(wd, use_container_width=True)
    if skipped:
        with st.expander(tr("Departamentos excluidos ({n})", n=len(skipped))):
            st.dataframe(pd.DataFrame(skipped, columns=["Departamento", "Motivo"]))
    st.caption("Metodología: trigger sobre E[rinde] del motor de normalización "
               "(FAST) · LC area-yield = max(0, T−y)/T sobre la serie "
               "normalizada · recargo = pura / (1 − deductions − margin) · "
               "CAT por Monte Carlo (20.000 sims) sobre la distribución "
               "seleccionada por AIC · agregación comonótona (sin "
               "diversificación espacial, criterio conservador).")


elif page == "1️⃣ Datos":
    edu.journey(0)
    st.caption(tr("Paso 1 de 5 · fuente MAGyP → dataset parquet listo para modelar"))

    st.title(tr("Datos"))
    st.write(f"**Fuente activa:** `{df.attrs.get('source', '—')}`")
    st.write(f"{len(df):,} filas · cultivos: {', '.join(cultivos)}")

    st.dataframe(serie, use_container_width=True)
    st.download_button(tr("Descargar serie filtrada (CSV)"),
                       serie.to_csv(index=False).encode(),
                       file_name=f"{cultivo}_{depto}.csv")

    st.divider()
    st.subheader(tr("Actualizar desde MAGyP"))
    st.caption("Descarga el dataset completo de Estimaciones Agrícolas "
               "(datosestimaciones.magyp.gob.ar) y regenera el Parquet local. "
               "En producción esto lo hace la GitHub Action mensual.")
    if st.button(tr("Descargar dataset completo")):
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


elif page == "2️⃣ Exploración":
    edu.journey(0)
    # ---- credibilidad de la cola (§6) ----
    _y = serie.dropna(subset=["rendimiento_kgxha"])["rendimiento_kgxha"]
    _g_prop = st.slider(tr("Garantía propuesta (para diagnóstico de credibilidad)"),
                        0.50, 0.90, 0.70, 0.05, key="cred_g")
    _med = float(_y.median())
    _n_tail = int((_y < _g_prop * _med).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tr("Campañas totales"), len(_y))
    c2.metric(tr("Campañas en la cola asegurada"), _n_tail,
              help=tr("Rindes por debajo de garantía×mediana. Solo estas observaciones informan directamente el precio del seguro."))
    c3.metric("CV", f"{_y.std()/_y.mean():.0%}")
    c4.metric(tr("Rindes cero"), int((_y == 0).sum()))
    if _n_tail < 5:
        st.warning(tr("**CREDIBILIDAD BAJA en la cola**: con {n} observaciones bajo la garantía, un promedio estable NO implica una cola estable. La prima descansa en muy pocos años.", n=_n_tail))
    edu.edu("¿Por qué importa la credibilidad de la cola?",
        "De toda la historia, solo las campañas que caen bajo la garantía generan siniestros.",
        None,
        "Podés tener 40 años de datos y aun así estimar la prima con 5 observaciones efectivas.",
        "Menos observaciones en la cola ⇒ más incertidumbre en la tasa (no necesariamente tasa más alta).",
        "No confundir una serie larga con una cola bien estimada.")

    # ---- timeline clasificado (§6) ----
    _cls = edu.classify_years(serie)
    _colors = {"EXTREMO": "#B85042", "BAJO": "#C9A227",
               "NORMAL": "#97BC62", "BUENO": "#2C5F2D"}
    figt = px.bar(_cls, x="anio", y="rendimiento_kgxha", color="clase",
                  color_discrete_map=_colors,
                  category_orders={"clase": list(_colors)},
                  labels={"anio": tr("Campaña"), "rendimiento_kgxha": "kg/ha"})
    figt.add_hline(y=_g_prop * _med, line_dash="dot",
                   annotation_text=tr("garantía propuesta"))
    figt.update_layout(height=360, legend_title="")
    st.plotly_chart(figt, use_container_width=True)
    st.caption(tr("Los años EXTREMOS no son outliers a limpiar: pueden ser la razón principal por la que existe este seguro."))

    st.title("AGRO RATE")
    st.caption(f"{cultivo.title()} · {depto} · {desde}–{hasta}")

    stats_ = ystats.describe(serie["rendimiento_kgxha"])
    slope = detrending.trend_slope(serie)
    worst = ystats.worst_years(det, n=1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rinde promedio", f"{stats_['mean']:,.0f} kg/ha")
    c2.metric("Tendencia (kg/ha/año)", f"{slope:+.1f}")
    c3.metric("Coef. de variación", f"{stats_['cv']:.0%}")
    c4.metric("Peor campaña", str(worst.iloc[0, 0]))

    st.plotly_chart(serie_chart(det, show_trend=True), use_container_width=True)


    st.subheader(tr("Análisis de rindes"))
    st.plotly_chart(serie_chart(serie), use_container_width=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader(tr("Estadística descriptiva"))
        st.dataframe(ystats.describe(serie["rendimiento_kgxha"])
                     .rename("valor").to_frame().style.format("{:,.1f}"))
    with c2:
        st.subheader("Distribución")
        fig = px.histogram(serie, x="rendimiento_kgxha", nbins=20,
                           labels={"rendimiento_kgxha": "kg/ha"})
        st.plotly_chart(fig, use_container_width=True)
        st.subheader(tr("Peores campañas"))
        st.dataframe(ystats.worst_years(det, n=5))


elif page == "3️⃣ Normalización (trend)":
    edu.journey(1)
    st.caption(tr("Paso 3 de 5 · separar tecnología μ(t) del shock agrícola. Kernel REGRESSION (suavizado local de μ(t), benchmark de mercado) sí es candidato; kernel DENSITY sigue prohibido acá — la forma de los shocks vive en el paso 4"))
    tab_auto, tab_manual = st.tabs([tr("AUTO (motor)"), tr("Manual (didáctico)")])
    with tab_auto:

        st.title(tr("Yield Risk Normalization Engine"))
        st.caption("Selección automática de detrending con validación "
                   "out-of-sample · regla one-standard-error")

        mode = st.radio(tr("Modo de ejecución"), ["FAST", "FULL"], horizontal=True,
                        help=tr("FAST: interactivo. FULL: grid completo + Historical Information Value (governance/pricing)."))
        res = run_engine(cultivo, depto, desde, hasta, mode, len(serie))

        if res.status != "OK":
            st.warning("INSUFFICIENT_DATA: la serie no alcanza para modelar "
                       "sin inventar precisión.")
            st.stop()

        rec, hl = res.recommended, res.recommended.half_life_years
        hl_txt = "∞ (pesos iguales)" if hl == float("inf") else f"{hl:g} años"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(tr("Modelo recomendado"), rec.trend_model)
        c2.metric("Detrending", rec.detrending_mode)
        c3.metric(tr("Historia"), f"{rec.start_year}–{int(det['anio'].max())}")
        c4.metric(tr("Half-life"), hl_txt)
        c1, c2, c3, c4 = st.columns(4)
        ci = res.expected_yield_ci
        c1.metric(tr("Rinde esperado {y}", y=res.target_year),
                  f"{res.expected_yield:,.0f} kg/ha",
                  help=f"CI 90%: {ci[0]:,.0f} – {ci[1]:,.0f}")
        c2.metric(tr("N efectivo"), f"{res.effective_sample_size:.1f}",
                  help=f"status: {res.sample_status}")
        c3.metric(tr("Confianza"), res.selection_confidence,
                  help=f"estabilidad de selección: {res.selection_stability_pct:.0f}%")
        c4.metric(tr("Score actuarial"), f"{res.validation_score:.4f}",
                  help=f"± {res.score_se:.4f} (SE) · tail {res.tail_score:.4f}")

        with st.expander(tr("¿Por qué este modelo?"), expanded=True):
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
        fig.update_layout(title=tr("Rinde observado y tendencia tecnológica"),
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
                         title=tr("Actuarial Validation Score (mejor por modelo)"),
                         labels={"cv_score": "score (menor = mejor)", "model": ""},
                         color=colors, color_discrete_map={True: "#2e7d32",
                                                           False: "#9e9e9e"})
            fig.update_layout(height=340, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            # Chart 5 — peso histórico por año
            fig = px.bar(nh, x="year", y="historical_weight",
                         title=tr("Peso histórico por campaña (half-life)"),
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
            st.caption(tr("Historical Information Value disponible en modo FULL."))

        with st.expander(tr("Diagnósticos actuariales (advanced)")):
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
    with tab_manual:
        st.caption(tr("Versión manual: elegís vos el modelo de trend y ves el efecto. Para tarifar usá siempre AUTO."))

        st.subheader(tr("Detrending manual"))
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


elif page == "4️⃣ Riesgo + Monte Carlo":
    edu.journey(2)
    st.caption(tr("Paso 4 de 5 · distribución de shocks (KDE benchmark, gates de cola) + masa en cero + Monte Carlo — reemplaza al módulo AIC viejo"))
    tab_auto, tab_manual = st.tabs([tr("Motor AUTO (recomendado)"), tr("Manual (legacy)")])
    with tab_auto:

        st.title(tr("Yield Risk Distribution Engine"))
        st.caption(tr("Distribución de riesgo tail-first: volatilidad · KDE "
                      "benchmark · masa en cero · backtest asegurador"))
        c1, c2 = st.columns([1, 1])
        level = c1.selectbox(tr("Nivel de agregación"),
                             ["DEPARTMENT", "PORTFOLIO", "FARM", "FIELD",
                              "PROVINCE"],
                             help=tr("Los datos fuente son departamentales; "
                                     "niveles más finos aplican priors de "
                                     "volatilidad y P(Y=0) configurables."))
        mode_exec = c2.radio(tr("Modo de ejecución"), ["FAST", "FULL"],
                             horizontal=True)
        rk = run_risk_engine(cultivo, depto, desde, hasta, level, mode_exec,
                             len(serie))
        if rk.status != "OK":
            st.warning(tr("INSUFFICIENT_DATA: la serie no alcanza para modelar "
                          "sin inventar precisión."))
            st.stop()

        # -------- card por defecto (§60) ---------------------------------
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(tr("Modelo recomendado"), rk.trend["model"],
                  help=f"{rk.trend['detrending_mode']} · desde {rk.trend['start_year']}")
        c2.metric(tr("Distribución"), rk.distribution["label"],
                  help=tr("Campeón AIC") + f": {rk.distribution['numerical_aic_champion']}")
        c3.metric(tr("Volatilidad"), rk.volatility["model"],
                  help=f"down/up SD = {rk.volatility['down_up_ratio']:.2f}")
        c4.metric(tr("P(Y=0)"), f"{rk.zero_mass['p_zero']:.2%}",
                  help=tr("a nivel {x}", x=rk.zero_mass["aggregation_level"])
                       + f" · {rk.zero_mass['estimation_method']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(tr("Rinde esperado {y}", y=hasta + 1),
                  f"{rk.trend['target_yield']:,.0f} kg/ha")
        c2.metric("P10", f"{rk.tail['p10']:,.0f} kg/ha",
                  help=f"P05 {rk.tail['p05']:,.0f} · P20 {rk.tail['p20']:,.0f} · "
                       f"ES10 {rk.tail['es_10']:,.0f}")
        c3.metric(tr("Prob. falla catastrófica"),
                  f"{rk.tail['prob_catastrophic']:.2%}",
                  help=tr("P(Y < 25% del esperado)"))
        c4.metric(tr("Confianza"), rk.selection["confidence"],
                  help=f"estab. {rk.selection['stability_pct']:.0f}% · "
                       f"score {rk.selection['risk_score']:.4f}")

        with st.expander(tr("¿Por qué este modelo de riesgo?"), expanded=True):
            for r in rk.selection["reason"]:
                st.write("–", r)

        # -------- distribución + zoom cola (§65-66, §68) ------------------
        sims = rk.simulated_yields
        zoom = st.radio(tr("Distribución de rindes simulada — zoom cola inferior"),
                        ["Densidad completa", "Zoom cola (≤ P30)"],
                        format_func=tr, horizontal=True)
        pos = sims[sims > 0]
        upper = float(np.quantile(sims, 0.30)) if zoom != "Densidad completa" else float(pos.max())
        fig = go.Figure()
        fig.add_histogram(x=pos[pos <= upper], nbinsx=60, histnorm="probability",
                          marker_color="#97BC62", name=tr("rinde simulado (kg/ha)"))
        p0 = rk.zero_mass["p_zero"]
        if p0 > 0:
            fig.add_bar(x=[0], y=[p0], width=[upper / 60], marker_color="#B85042",
                        name="P(Y=0)")
        for q, lbl in [(rk.tail["p05"], "P05"), (rk.tail["p10"], "P10"),
                       (rk.tail["p20"], "P20")]:
            if q <= upper:
                fig.add_vline(x=q, line_dash="dot", annotation_text=lbl)
        _g70 = 0.70 * rk.trend["target_yield"]
        fig.add_vline(x=_g70, line_color="#B85042", line_width=2,
                      annotation_text=tr("GARANTÍA 70%"))
        fig.add_vrect(x0=0, x1=_g70, fillcolor="#B85042", opacity=0.08,
                      line_width=0,
                      annotation_text=tr("región asegurada"),
                      annotation_position="top left")
        fig.update_layout(height=400, barmode="overlay",
                          xaxis_title=tr("rinde simulado (kg/ha)"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(tr("Masa en cero (P(Y=0) = {p:.2%}) mostrada como barra "
                      "discreta — no se esconde en la densidad.", p=p0))

        if edu.guided():
            with st.expander("🧪 " + tr("Experimento: bandwidth del kernel (no afecta la tarifa)")):
                st.caption(tr("SUB-SUAVIZADO ← bandwidth → SOBRE-SUAVIZADO · La tarifa real usa el bandwidth elegido estadísticamente (LOO-likelihood); este slider es para VER por qué no es una perilla cosmética."))
                from src.distribution.candidates import ReflectedKDE
                _nh = rk.normalized_history
                _shx = _nh.loc[~_nh["zero_event"], "standardized_shock"].to_numpy(float)
                _bwf = st.slider("× Silverman", 0.3, 2.5, 1.0, 0.1, key="bw_demo")
                _kde = ReflectedKDE("kde", bw_factor=_bwf,
                                    positive_support=True).fit(_shx)
                _s = np.maximum(_kde.sample(8000, np.random.default_rng(3)), 0)
                _ys = rk.trend["target_yield"] * _s
                _pg = float((_ys < _g70).mean())
                _ind = float(np.maximum(_g70 - _ys, 0).mean() / _g70)
                b1, b2, b3 = st.columns(3)
                b1.metric("P(Y < garantía 70%)", f"{_pg:.1%}")
                b2.metric(tr("Prima pura (garantía 70%)"), f"{_ind:.2%}")
                b3.metric("P10", f"{np.quantile(_ys, 0.10):,.0f} kg/ha")
                _figk = go.Figure()
                _figk.add_histogram(x=rk.trend["target_yield"] * _shx,
                                    histnorm="probability density",
                                    nbinsx=30, opacity=0.4,
                                    marker_color="#C9A227",
                                    name=tr("shocks históricos"))
                _figk.add_histogram(x=_ys, histnorm="probability density",
                                    nbinsx=60, opacity=0.55,
                                    marker_color="#2C5F2D",
                                    name=f"KDE ×{_bwf:.1f}")
                _figk.add_vline(x=_g70, line_color="#B85042", line_dash="dot")
                _figk.update_layout(height=300, barmode="overlay")
                st.plotly_chart(_figk, use_container_width=True)
        edu.edu("Distribución de shocks: ¿por qué no elegir solo por AIC?",
            "El AIC mide ajuste global; el seguro paga en la cola inferior.",
            r"RiskScore = w_t\,Tail + w_i\,Indemnity + w_d\,Dist + w_v\,Vol + w_s\,Stab + w_c\,Compl",
            "Una paramétrica puede ganar el AIC comprimiendo justo la parte que genera siniestros.",
            "Subdispersión de cola ⇒ tasa subestimada sistemáticamente (el error caro).",
            "Un candidato con mejor AIC pero gate de cola fallido queda RECHAZADO — mirá el ranking abajo.")

        # -------- claim curve + backtest (§67, §61) -----------------------
        c1, c2 = st.columns(2)
        with c1:
            cc = rk.claim_curve
            fig = go.Figure()
            fig.add_scatter(x=cc["coverage"] * 100, y=cc["expected_indemnity"] * 100,
                            mode="lines+markers", name=tr("modelo"),
                            line=dict(color="#2C5F2D", width=3))
            fig.add_scatter(x=cc["coverage"] * 100, y=cc["hist_loss_cost"] * 100,
                            mode="lines+markers", name=tr("histórico"),
                            line=dict(dash="dash", color="#C9A227"))
            fig.update_layout(title=tr("Curva de siniestros: prima pura vs nivel de cobertura"),
                              xaxis_title=tr("Nivel de cobertura (% del rinde esperado)"),
                              yaxis_title=tr("Prima pura (% de la garantía)"),
                              height=380)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            bt = pd.DataFrame(rk.insurance_validation["per_coverage"])
            if len(bt):
                fig = go.Figure()
                fig.add_bar(x=bt["coverage"] * 100, y=bt["obs_freq"] * 100,
                            name=tr("histórico"), marker_color="#C9A227")
                fig.add_bar(x=bt["coverage"] * 100, y=bt["pred_freq"] * 100,
                            name=tr("modelo"), marker_color="#2C5F2D")
                fig.update_layout(title=tr("Frecuencia de siniestro: predicha vs observada"),
                                  barmode="group", height=380,
                                  xaxis_title=tr("Nivel de cobertura (% del rinde esperado)"),
                                  yaxis_title="%")
                st.plotly_chart(fig, use_container_width=True)

        with st.expander("❓ " + tr("Preguntas metodológicas frecuentes")):
            st.markdown(tr("**¿Cómo se simulan las colas? ¿Usan el coeficiente de variación?** No. Un CV resume toda la variabilidad en un número y presupone una forma (aprox. simétrica) — justo lo que falla en la cola. Acá la cola sale de tres piezas: (1) los años malos reales del departamento, normalizados a tecnología actual, que quedan como anclas; (2) la forma completa de la distribución ajustada a esos shocks (weibull, kernel, etc.), seleccionada por backtest con gates que rechazan a quien comprima la cola; (3) la masa en cero. El CV del departamento queda implícito en la distribución; nunca se usa el de la provincia."))
            st.markdown(tr("**¿Los datos son del departamento o de la provincia?** La distribución se ajusta SIEMPRE a la serie del departamento seleccionado. La provincia no entra en la forma de la cola; solo aparece en dos lugares acotados: el prior de P(Y=0) por nivel de agregación, y (en Resultados) la metodología modal si activás el modo uniforme. Si simulás a nivel FIELD/FARM se aplican multiplicadores de volatilidad prior — un supuesto documentado en config, no una estimación."))
            st.markdown(tr("**¿Cómo se generan los rindes cero?** Con un modelo hurdle en dos partes: P(Y=0) se estima clasificando los ceros históricos por superficie (sembrada sin cosechar ≠ error de encuesta) y aplicando shrinkage beta-binomial hacia un prior por nivel de agregación — por eso P(Y=0)>0 aunque el departamento nunca haya registrado un cero. En la simulación: si u<P(Y=0) el rinde es exactamente 0; si no, se muestrea de la distribución positiva. Una densidad continua pura no puede producir un cero verdadero; la mezcla sí."))

        # ---------- Kernel Risk Model (§31, §36-37): distribución vs motor ----------
        st.subheader("🧬 " + tr("Kernel Risk Model — misma distribución, tres motores numéricos"))
        from src.distribution.kernel_engine import KernelRiskModel
        _nhk = rk.normalized_history
        _posk = _nhk[~_nhk["zero_event"]]
        _kmod = KernelRiskModel.fit(
            _posk["standardized_shock"].to_numpy(float),
            years=_posk["year"].to_numpy(),
            weights=_posk["historical_weight"].to_numpy(float),
            weighted=True, p_zero=rk.zero_mass["p_zero"])
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Bandwidth (log)", f"{_kmod.bandwidth:.4f}",
                  help=f"{_kmod.bandwidth_method} · ×{_kmod.bandwidth_factor} Silverman")
        k2.metric(tr("Anclas históricas"), len(_kmod.anchors_shock))
        _nodes = st.session_state.get("k_nodes", 20)
        k3.metric(tr("Escenarios determinísticos"),
                  len(_kmod.anchors_shock) * _nodes + (1 if _kmod.p_zero > 0 else 0))
        k4.metric("N_eff", f"{_kmod.governance()['effective_sample']:.1f}")

        _gk = 0.70
        _det = _kmod.expected_loss_deterministic(_gk, _nodes)
        _mck = _kmod.expected_loss_mc(_gk, 50_000, np.random.default_rng(1))
        _ana = _kmod.expected_loss_integration(_gk)
        st.markdown(tr("**Expected loss (garantía 70%) por motor** — misma distribución calibrada: ")
                    + f"KERNEL_DETERMINISTIC **{_det['expected_loss']:.3%}** · "
                    + f"KERNEL_MONTE_CARLO **{_mck['expected_loss']:.3%}** · "
                    + f"KERNEL_INTEGRATION (analítica) **{_ana['expected_loss']:.3%}**")
        st.caption(tr("El kernel ES la distribución; determinístico/MC/integración son solo formas de consumirla. Convergen porque no hay una segunda calibración. El determinístico no tiene ruido de muestreo y cada escenario es trazable a una campaña real."))

        with st.expander("🔎 " + tr("Trazabilidad: ¿por qué existe cada escenario? (§35)")):
            _sc = _kmod.deterministic_scenarios(_nodes)
            _worst_anchor = _posk.loc[_posk["standardized_shock"].idxmin()]
            _ay = st.selectbox(tr("Ancla histórica"),
                               sorted(_posk["year"].astype(int)),
                               index=int(np.argmin(_posk["standardized_shock"].to_numpy())))
            _sub = _sc[_sc["historical_anchor_year"] == _ay]
            _base_s = float(_sub["base_relative_shock"].iloc[0])
            st.caption(tr("Campaña {y}: shock normalizado {b:.3f} → {n} escenarios vecinos vía bandwidth {h:.3f}. El extremo histórico sigue siendo un centro de masa de probabilidad — el suavizado no lo borra.",
                          y=_ay, b=_base_s, n=len(_sub), h=_kmod.bandwidth))
            _figs = go.Figure()
            _figs.add_scatter(x=_sub["simulated_relative_shock"],
                              y=_sub["weight"], mode="markers",
                              marker=dict(size=7, color="#2C5F2D"),
                              name=tr("escenarios kernel"))
            _figs.add_vline(x=_base_s, line_color="#B85042",
                            annotation_text=tr("shock observado"))
            _figs.update_layout(height=280,
                                xaxis_title=tr("shock relativo simulado"),
                                yaxis_title=tr("peso de probabilidad"))
            st.plotly_chart(_figs, use_container_width=True)
            st.dataframe(_sub[["kernel_quantile", "perturbation",
                               "simulated_relative_shock", "weight"]]
                         .round(4), use_container_width=True, height=200)

        with st.expander(tr("Diagnósticos actuariales (advanced)")):
            st.write(f"**{tr('Ranking de distribuciones (score asegurador)')}**")
            st.dataframe(rk.ranking, use_container_width=True)
            st.write(f"**{tr('Backtest por nivel de cobertura')}**")
            if len(bt):
                st.dataframe(bt.round(4), use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**{tr('Volatilidad downside vs upside')}**")
                st.json({k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in rk.volatility.items() if k != "evidence"})
            with c2:
                st.write("**Zero mass**")
                st.json({k: v for k, v in rk.zero_mass.items()
                         if k != "classification"})
            if rk.warnings:
                st.write(f"**{tr('Advertencias')}**")
                for w in rk.warnings:
                    st.warning(w)
            st.caption(f"engine v{rk.governance['engine_version']} · "
                       f"hash {rk.governance['dataset_hash']} · "
                       f"{rk.timings['total_s']} s · "
                       f"{rk.timings['n_candidates']} candidatos")
    with tab_manual:
        st.caption(tr("Módulo previo al motor: ajuste AIC simple sobre la serie detrendeada, sin gates ni masa en cero. Se mantiene para comparar."))

        st.subheader(tr("Simulación Monte Carlo manual"))

        c1, c2, c3 = st.columns(3)
        _default_expected = (auto_result.expected_yield
                             if auto_result is not None and auto_result.status == "OK"
                             else det["trend"].iloc[-1])
        expected = c1.number_input(tr("Rinde esperado (kg/ha)"),
                                   value=float(round(_default_expected, -1)),
                                   step=50.0)
        guarantee = c2.slider("Garantía", 0.50, 0.90, 0.70, 0.05)
        price = c3.number_input("Precio (USD/kg)", value=0.40, step=0.05)

        c1, c2, c3 = st.columns(3)
        distribution = c1.selectbox(
            "Distribución",
            ["empirical bootstrap", "kde", "normal", "lognormal", "gamma",
             "weibull", "student_t"])
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

        st.subheader(tr("Ajuste de distribuciones (AIC)"))
        st.dataframe(fit_all(det["detrended"]).drop(columns="params"))


elif page == "5️⃣ Pricing":
    edu.journey(8)
    st.caption(tr("Paso 5 de 5 · del rinde al precio: prima pura por nivel de garantía"))

    st.title(tr("Pricing — Yield Shortfall"))
    st.caption(tr("Construcción de tasa transparente: qué aporta cada fuente (histórico observado → modelo simulado → recargos) y por qué."))

    rk = run_risk_engine(cultivo, depto, desde, hasta, "DEPARTMENT", "FAST",
                         len(serie))
    if rk.status != "OK":
        st.warning(tr("INSUFFICIENT_DATA: la serie no alcanza para modelar sin inventar precisión."))
        st.stop()
    mu = float(rk.trend["target_yield"])

    c1, c2, c3 = st.columns(3)
    expected = c1.number_input(tr("Rinde esperado (kg/ha)"),
                               value=float(round(mu, -1)), step=50.0,
                               help=tr("Default: E[rinde] del motor. Si lo cambiás, las simulaciones se re-escalan proporcionalmente."))
    guarantee = c2.slider("Garantía", 0.50, 0.90, 0.70, 0.05)
    price = c3.number_input("Precio (USD/kg)", value=0.40, step=0.05)
    c1, c2 = st.columns(2)
    deductions_p = c1.number_input(tr("Deductions"), value=0.25, step=0.05)
    margin_p = c2.number_input(tr("MR Margin"), value=0.10, step=0.05)

    g_kg = guarantee * expected
    sa = g_kg * price

    # --- 1) tasa pura observada (burning cost sobre serie normalizada) ---
    nh = rk.normalized_history
    y_hist = nh["normalized_yield"].to_numpy(float) * (expected / mu)
    lc_hist = np.maximum(g_kg - y_hist, 0.0) / g_kg
    tasa_obs = float(lc_hist.mean())
    freq_obs = float((lc_hist > 0).mean())

    # --- 2) tasa pura simulada: distribución seleccionada × motor numérico ---
    sims = rk.simulated_yields * (expected / mu)
    lc_sim = np.maximum(g_kg - sims, 0.0) / g_kg
    _tasa_mc = float(lc_sim.mean())
    _freq_mc = float((lc_sim > 0).mean())
    _sev_mc = float(lc_sim[lc_sim > 0].mean()) if (lc_sim > 0).any() else 0.0

    from src.distribution.kernel_engine import KernelRiskModel
    _nhp = rk.normalized_history
    _posp = _nhp[~_nhp["zero_event"]]
    _kmp = KernelRiskModel.fit(
        _posp["standardized_shock"].to_numpy(float),
        years=_posp["year"].to_numpy(),
        weights=_posp["historical_weight"].to_numpy(float),
        weighted=True, p_zero=rk.zero_mass["p_zero"])
    _det = _kmp.expected_loss_deterministic(guarantee, 20)
    _ana = _kmp.expected_loss_integration(guarantee)

    _eng = st.radio(
        tr("Motor numérico de la tasa simulada"),
        ["MC_SELECCIONADA", "KERNEL_DETERMINISTICO", "KERNEL_INTEGRACION"],
        format_func=lambda k: {
            "MC_SELECCIONADA": tr("Monte Carlo · distribución seleccionada ({d})",
                                  d=rk.distribution["label"]),
            "KERNEL_DETERMINISTICO": tr("Kernel determinístico · anclas × nodos (log-kernel ponderado)"),
            "KERNEL_INTEGRACION": tr("Kernel integración analítica (exacta)"),
        }[k], horizontal=True, key="pr_engine")
    st.caption(tr("Los tres a esta garantía: MC {a:.2%} · kernel determinístico {b:.2%} · integración {c:.2%}. El kernel NO es una alternativa a Monte Carlo: es una distribución; MC/determinístico/integración son motores. Regla práctica: riesgo individual → determinístico (sin ruido, trazable); cartera/reaseguro → Monte Carlo; integración → benchmark exacto.",
                  a=_tasa_mc, b=_det["expected_loss"], c=_ana["expected_loss"]))
    if _eng == "KERNEL_DETERMINISTICO":
        tasa_sim, freq_sim, sev_sim = (_det["expected_loss"],
                                       _det["claim_probability"],
                                       _det["severity"])
        _eng_label = "weighted log-kernel · KERNEL_DETERMINISTIC"
    elif _eng == "KERNEL_INTEGRACION":
        tasa_sim, freq_sim, sev_sim = (_ana["expected_loss"],
                                       _ana["claim_probability"],
                                       _ana["severity"])
        _eng_label = "weighted log-kernel · KERNEL_INTEGRATION"
    else:
        tasa_sim, freq_sim, sev_sim = _tasa_mc, _freq_mc, _sev_mc
        _eng_label = f"{rk.distribution['label']} · MONTE_CARLO"
    edu.edu("¿Monte Carlo o kernel determinístico? La regla del comité técnico",
        "El kernel es una distribución; Monte Carlo es un motor numérico. No compiten en el mismo eje.",
        None,
        "Para UN riesgo, el determinístico da el mismo número siempre (sin semillas), y cada escenario se explica con una campaña real — defendible ante auditoría y reaseguro. Para CARTERA, MC es necesario: correlación entre departamentos, estructuras de reaseguro y curvas EP no se enumeran.",
        "Si la distribución seleccionada por el motor (p.ej. weibull) y el kernel difieren materialmente en tasa, eso es MODEL RISK y está cuantificado abajo — no lo resuelve elegir motor.",
        "No uses el determinístico para agregar cartera asumiendo independencia: subestima el CAT. Y no re-calibres el kernel por motor: es UNA calibración.")

    # --- guardrail de experiencia (§24): opcional, default filosofía actual ---
    _gmode = st.selectbox(
        tr("Modo de tasa pura (guardrail de experiencia)"),
        ["MODEL_ONLY", "MAX_MODEL_OBSERVED", "CREDIBILITY_BLEND",
         "OBSERVED_ONLY"],
        help=tr("Existe para que un modelo estadístico no borre experiencia observada creíble — no como conservadurismo arbitrario. MODEL_ONLY es el default de la app."))
    if _gmode == "MAX_MODEL_OBSERVED":
        tasa_pura = max(tasa_sim, tasa_obs)
    elif _gmode == "OBSERVED_ONLY":
        tasa_pura = tasa_obs
    elif _gmode == "CREDIBILITY_BLEND":
        _z = st.slider("Z (credibilidad del observado)", 0.0, 1.0, 0.3, 0.05)
        tasa_pura = _z * tasa_obs + (1 - _z) * tasa_sim
    else:
        tasa_pura = tasa_sim
    if _gmode != "MODEL_ONLY":
        st.caption(tr("Guardrail activo: tasa pura = {m} → {t:.2%} (modelo {a:.2%} · observado {b:.2%})",
                      m=_gmode, t=tasa_pura, a=tasa_sim, b=tasa_obs))

    # --- 3) tasa técnica ---
    loading = 1.0 - deductions_p - margin_p
    tasa_tec = tasa_pura / max(loading, 1e-9)
    st.session_state["live_rate"] = tasa_pura

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tr("Tasa pura observada"), f"{tasa_obs:.2%}",
              help=tr("Burning cost histórico: promedio del loss cost sobre la serie normalizada por el motor ({n} campañas). Es lo que efectivamente pasó, a tecnología actual.", n=len(y_hist)))
    c2.metric(tr("Tasa pura simulada"), f"{tasa_sim:.2%}",
              delta=f"{tasa_sim-tasa_obs:+.2%} vs observada", delta_color="off",
              help=tr("Monte Carlo ({n} sims) de la distribución seleccionada por el motor de riesgo ({d}), incluyendo masa en cero (P(Y=0)={p:.2%}) y la cola completa — no solo los años que tocaron pasar.",
                      n=len(sims), d=rk.distribution["label"],
                      p=rk.zero_mass["p_zero"]))
    c3.metric(tr("Tasa técnica"), f"{tasa_tec:.2%}",
              help=tr("pura simulada / (1 − deductions − margin) = {t:.2%} / {l:.2f}",
                      t=tasa_pura, l=loading))
    c4.metric(tr("Prima técnica"), f"{tasa_tec*sa:,.2f} USD/ha",
              help=tr("SA = garantía × precio = {sa:,.0f} USD/ha", sa=sa))

    # --- buildup explicado paso a paso ---
    st.subheader(tr("Construcción de la tasa"))
    build = pd.DataFrame([
        {"Concepto": tr("1 · Tasa pura observada (burning cost)"),
         "Valor": f"{tasa_obs:.2%}",
         "Fuente": tr("Serie normalizada del motor · {n} campañas · frecuencia {f:.0%}",
                      n=len(y_hist), f=freq_obs)},
        {"Concepto": tr("2 · Ajuste por modelo de riesgo"),
         "Valor": f"{tasa_sim-tasa_obs:+.2%}",
         "Fuente": tr("Distribución {d} + P(Y=0)={p:.2%}: completa la cola que la muestra finita no vio (o suaviza la que sobre-representó)",
                      d=rk.distribution["label"], p=rk.zero_mass["p_zero"])},
        {"Concepto": tr("3 · Tasa pura simulada"),
         "Valor": f"{tasa_sim:.2%}",
         "Fuente": f"{_eng_label} · " + tr("frecuencia {f:.1%} · severidad {s:.1%}",
                      f=freq_sim, s=sev_sim)},
        {"Concepto": tr("4 · Recargo estructura"),
         "Valor": f"÷ {loading:.2f}",
         "Fuente": tr("1 − deductions ({d:.0%}) − margen de riesgo ({m:.0%})",
                      d=deductions_p, m=margin_p)},
        {"Concepto": tr("5 · Tasa técnica final"),
         "Valor": f"{tasa_tec:.2%}",
         "Fuente": tr("Se detiene antes de recargos comerciales (gastos de venta, utilidad, reaseguro) — spec §88")},
    ])
    st.dataframe(build, use_container_width=True, hide_index=True)

    # --- curva de garantías: observado vs modelo ---
    st.subheader(tr("Curva de garantías — observado vs modelo"))
    cc = rk.claim_curve.copy()
    scale = expected / mu
    fig = go.Figure()
    fig.add_scatter(x=cc["coverage"]*100, y=cc["hist_loss_cost"]*100,
                    mode="lines+markers", name=tr("tasa pura observada"),
                    line=dict(dash="dash", color="#C9A227"))
    fig.add_scatter(x=cc["coverage"]*100, y=cc["expected_indemnity"]*100,
                    mode="lines+markers", name=tr("tasa pura simulada"),
                    line=dict(color="#2C5F2D", width=3))
    fig.add_vline(x=guarantee*100, line_dash="dot",
                  annotation_text=tr("garantía elegida"))
    fig.update_layout(height=400,
                      xaxis_title=tr("Nivel de cobertura (% del rinde esperado)"),
                      yaxis_title=tr("Tasa pura (% de la garantía)"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(tr("Dónde separan las curvas es donde el modelo aporta: en garantías bajas manda la cola (y los ceros); en garantías altas ambas convergen porque los siniestros leves sí están bien representados en la muestra."))

    # ---------- frecuencia × severidad (§15) ----------
    st.subheader(tr("Descomposición: frecuencia × severidad"))
    st.markdown(f"**P(siniestro) × severidad media = pérdida esperada**  \n"
                f"{freq_sim:.1%} × {sev_sim:.1%} ≈ **{freq_sim*sev_sim:.2%}**"
                f" &nbsp;·&nbsp; (tasa pura simulada exacta: {tasa_sim:.2%})")
    edu.edu("Frecuencia × Severidad",
        "Toda prima pura se descompone en cuán seguido pasa y cuán grave es cuando pasa.",
        r"E[L] = P(Y<G)\times E\left[\frac{G-Y}{G}\,\Big|\,Y<G\right]",
        "Dos riesgos con la misma prima pueden ser opuestos: frecuente-leve vs raro-severo. El reaseguro los trata distinto.",
        "Subir la cobertura sube sobre todo la frecuencia; el riesgo CAT vive en la severidad.",
        "Un modelo puede clavar la frecuencia y subestimar la severidad — el backtest del motor valida ambas.")

    # ---------- incertidumbre de la tasa (§18) ----------
    st.subheader(tr("Incertidumbre de la tasa"))
    _ck = f"{cultivo}|{depto}|{desde}|{hasta}|{guarantee:.2f}"
    _lo, _hi, _B = edu.bootstrap_rate_ci(rk, guarantee, _ck)
    c1, c2 = st.columns([1, 2])
    c1.metric(tr("Tasa pura simulada"), f"{tasa_sim:.2%}",
              help=tr("El modelo estima el riesgo; no observa el riesgo verdadero."))
    c2.markdown(f"**{tr('Intervalo bootstrap 90%')}**: {_lo:.2%} – {_hi:.2%} "
                f"&nbsp;·&nbsp; {_B} " + tr("re-muestreos de la historia, re-ajustando la distribución seleccionada"))
    edu.edu("¿Por qué la tasa tiene un intervalo?",
        "La tasa sale de una muestra finita de campañas; otra historia igualmente plausible daría otra tasa.",
        None,
        "Presentar 2.24% como si fuera exacto oculta el riesgo de estimación — clave frente a reaseguro.",
        "Muestra chica o cola escasa ⇒ intervalo más ancho (no necesariamente tasa más alta).",
        "El intervalo mide riesgo de DATOS con el modelo fijo; el model risk de abajo es otra cosa.")

    # ---------- leave-one-year-out (§21) ----------
    st.subheader(tr("Estabilidad leave-one-year-out"))
    _loyo = edu.loyo_rates(rk, guarantee, _ck)
    _base = _loyo.attrs["base"]
    _figl = px.bar(_loyo, x="year", y="rate",
                   labels={"year": tr("campaña quitada"), "rate": tr("tasa sin esa campaña")})
    _figl.add_hline(y=_base, line_dash="dot",
                    annotation_text=tr("base {b:.2%}", b=_base))
    _figl.update_layout(height=320)
    st.plotly_chart(_figl, use_container_width=True)
    _inf = _loyo.loc[_loyo["delta"].abs().idxmax()]
    st.caption(tr("Rango: {mn:.2%} – {mx:.2%} · campaña más influyente: {y} ({d:+.2%})",
                  mn=_loyo['rate'].min(), mx=_loyo['rate'].max(),
                  y=int(_inf['year']), d=_inf['delta']))

    # ---------- model risk (§19) ----------
    st.subheader(tr("Model risk: la tasa según cada metodología"))
    _mr = edu.model_risk_table(rk, guarantee, _ck)
    _sel_name = rk.distribution["recommended"]
    _mr["Δ vs seleccionada"] = _mr["rate"] - tasa_sim
    st.dataframe(_mr.style.format({"rate": "{:.2%}", "Δ vs seleccionada": "{:+.2%}"}),
                 use_container_width=True, hide_index=True)
    st.caption(tr("min {mn:.2%} · mediana {md:.2%} · max {mx:.2%} — esta dispersión es MODEL RISK: distinta del riesgo de datos (bootstrap) y del de proceso.",
                  mn=_mr['rate'].min(), md=_mr['rate'].median(), mx=_mr['rate'].max()))

    # ---------- sanity checks (§20) ----------
    st.subheader(tr("Controles de sanidad actuarial"))
    _checks = edu.sanity_checks(rk, rk.claim_curve, _loyo, guarantee, sims)
    st.dataframe(pd.DataFrame(_checks), use_container_width=True, hide_index=True)

    # ---------- formula inspector (§24) ----------
    with st.expander("🔍 " + tr("Ver fórmulas con los valores reales")):
        st.latex(rf"G = c \times E[Y] = {guarantee:.0%} \times {expected:,.0f} = {g_kg:,.0f}\ kg/ha")
        st.latex(rf"LC = E\left[\frac{{\max(G-Y,0)}}{{G}}\right] = {tasa_sim:.4f}")
        st.latex(rf"tasa\ t\'ecnica = \frac{{LC}}{{1-ded-mr}} = \frac{{{tasa_sim:.4f}}}{{{loading:.2f}}} = {tasa_tec:.4f}")
        st.latex(rf"prima = tasa \times SA = {tasa_tec:.4f} \times {sa:,.0f} = {tasa_tec*sa:,.2f}\ USD/ha")

    # ---------- rate story (§23) ----------
    st.subheader("📜 " + tr("La historia de tu tasa"))
    _pg_sel = float((sims < g_kg).mean())
    _short = g_kg - sims[sims < g_kg]
    _story = tr(
        "Tu cobertura del {c:.0%} garantiza {g:,.0f} kg/ha. Bajo la distribución seleccionada ({d}), el {p:.0%} de las campañas simuladas cae por debajo de esa garantía; cuando hay siniestro, el faltante promedio es {s:,.0f} kg/ha ({sev:.0%} de la garantía). En {n:,} campañas simuladas eso produce una indemnización esperada del {t:.2%} de la responsabilidad — la tasa pura. El bootstrap indica un rango plausible de {lo:.2%}–{hi:.2%}, reflejo de la historia limitada; y entre metodologías defendibles la tasa va de {mn:.2%} a {mx:.2%}. La mayor fuente de sensibilidad es la representación de la cola inferior.",
        c=guarantee, g=g_kg, d=rk.distribution["label"], p=_pg_sel,
        s=float(_short.mean()) if len(_short) else 0.0,
        sev=sev_sim, n=len(sims), t=tasa_sim, lo=_lo, hi=_hi,
        mn=_mr['rate'].min(), mx=_mr['rate'].max())
    st.markdown(f"> {_story}")

    with st.expander(tr("Loss cost por campaña (observado)")):
        lc_df = pd.DataFrame({"campania": nh["year"],
                              "loss_cost": lc_hist})
        fig = px.bar(lc_df, x="campania", y="loss_cost",
                     labels={"campania": tr("Campaña"), "loss_cost": "Loss cost"})
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)
