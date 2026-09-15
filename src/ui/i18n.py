"""i18n mínimo: español canónico → inglés, con tr() leyendo el idioma
de st.session_state. Las claves internas (values de widgets comparados
en código) nunca se traducen; solo la presentación (format_func/labels).
"""

from __future__ import annotations

import streamlit as st

EN = {
    # --- sidebar / navegación ---
    "Módulo": "Module", "Cultivo": "Crop", "Departamento": "Department",
    "Período": "Period", "Fuente": "Source", "Idioma": "Language",
    "Agricultural Risk & Insurance Analytics": "Agricultural Risk & Insurance Analytics",
    "📊 Resultados": "📊 Results", "🏠 Dashboard": "🏠 Dashboard",
    "🌱 Análisis de rindes": "🌱 Yield analysis", "📈 Detrending": "📈 Detrending",
    "🧠 Normalización": "🧠 Normalization", "📉 Riesgo": "📉 Risk distribution",
    "💰 Pricing": "💰 Pricing", "🎲 Monte Carlo": "🎲 Monte Carlo",
    "📦 Datos": "📦 Data",
    "'auto' selecciona modelo, modo, start year y half-life con validación out-of-sample (Yield Risk Normalization Engine)":
        "'auto' selects model, mode, start year and half-life via out-of-sample validation (Yield Risk Normalization Engine)",
    "Serie insuficiente para el motor; fallback linear":
        "Series too short for the engine; linear fallback",

    # --- Resultados ---
    "Resultados — consolidado de cartera": "Results — portfolio consolidation",
    "Tarifación técnica propia (modelo area-yield sobre serie normalizada) · escenario CAT · peor año histórico · loss cap":
        "In-house technical rating (area-yield on normalized series) · CAT scenario · worst historical year · loss cap",
    "**Del panel lateral aplican:** cultivo (**{c}**) y período (**{d}–{h}**). El departamento del panel no aplica acá: se consolidan todos los elegibles de la provincia seleccionada.":
        "**From the sidebar apply:** crop (**{c}**) and period (**{d}–{h}**). The sidebar department does not apply here: all eligible departments of the selected province are consolidated.",
    "Provincia": "Province", "Mín. campañas por depto": "Min. seasons per dept",
    "Modo de garantía": "Guarantee mode",
    "% del rinde esperado": "% of expected yield",
    "rinde fijo (kg/ha)": "fixed yield (kg/ha)",
    "Valor de garantía": "Guarantee value",
    "65 ⇒ trigger = 65% del E[rinde] del motor · o kg/ha fijos":
        "65 ⇒ trigger = 65% of the engine's expected yield · or fixed kg/ha",
    "Suma asegurada (USD/ha)": "Sum insured (USD/ha)",
    "Deductions": "Deductions", "MR Margin": "MR Margin",
    "Período de retorno CAT": "CAT return period",
    "Aplicar LOSS CAP": "Apply LOSS CAP",
    "Loss cap (% de la suma asegurada)": "Loss cap (% of sum insured)",
    "Calcular consolidado": "Compute portfolio",
    "Configurá los parámetros y presioná **Calcular consolidado**. El motor de normalización corre por cada departamento elegible (~2 s c/u la primera vez; después queda cacheado).":
        "Set the parameters and press **Compute portfolio**. The normalization engine runs per eligible department (~2 s each on first run; cached afterwards).",
    "Ningún departamento elegible con esos filtros.":
        "No eligible department with those filters.",
    "Totales de cartera": "Portfolio totals",
    "Serie utilizada: campañas {d}–{h} · triggers sobre E[rinde] estimado con esa ventana.":
        "Series used: seasons {d}–{h} · triggers on the expected yield estimated with that window.",
    "Prima técnica total": "Total technical premium",
    "Suma asegurada total": "Total sum insured",
    "Tasa técnica media": "Average technical rate",
    "Departamentos": "Departments",
    "Pérdida CAT (RP {rp} años)": "CAT loss (RP {rp} years)",
    "Siniestralidad CAT (siniestros/prima)": "CAT loss ratio (losses/premium)",
    "equivale a PML {x:.1f}× la prima técnica": "equals a PML of {x:.1f}× technical premium",
    "Peor año histórico ({y})": "Worst historical year ({y})",
    "Siniestralidad {y}": "Loss ratio {y}",
    "equivale a {x:.1f}× la prima técnica": "equals {x:.1f}× technical premium",
    "Detalle por departamento": "Department detail",
    "⬇ Descargar CSV": "⬇ Download CSV",
    "Prima técnica por departamento (USD)": "Technical premium by department (USD)",
    "Unidad del gráfico histórico": "Historical chart unit",
    "% Loss Ratio (siniestros/prima)": "% Loss Ratio (losses/premium)",
    "Siniestralidad histórica por campaña (a valores actuales)":
        "Historical loss ratio by season (at current values)",
    "Pérdida histórica agregada por campaña (USD, a valores actuales)":
        "Aggregate historical loss by season (USD, current values)",
    "% de la prima técnica": "% of technical premium",
    "LR 100% (siniestros = prima)": "LR 100% (losses = premium)",
    "sin cap": "no cap", "con cap {c:.0%}": "cap {c:.0%}",
    "peor año: {y}": "worst year: {y}", "Campaña": "Season",
    "Detalle del peor año ({y})": "Worst-year detail ({y})",
    "Departamentos excluidos ({n})": "Excluded departments ({n})",

    # --- Normalización ---
    "Yield Risk Normalization Engine": "Yield Risk Normalization Engine",
    "Selección automática de detrending con validación out-of-sample · regla one-standard-error":
        "Automatic detrending selection with out-of-sample validation · one-standard-error rule",
    "Modo de ejecución": "Execution mode",
    "FAST: interactivo. FULL: grid completo + Historical Information Value (governance/pricing).":
        "FAST: interactive. FULL: complete grid + Historical Information Value (governance/pricing).",
    "INSUFFICIENT_DATA: la serie no alcanza para modelar sin inventar precisión.":
        "INSUFFICIENT_DATA: the series is too short to model without inventing precision.",
    "Modelo recomendado": "Recommended model", "Historia": "History",
    "Half-life": "Half-life", "∞ (pesos iguales)": "∞ (equal weights)",
    "Rinde esperado {y}": "Expected yield {y}",
    "N efectivo": "Effective N", "Confianza": "Confidence",
    "Score actuarial": "Actuarial score",
    "¿Por qué este modelo?": "Why this model?",
    "Campeón numérico: {x}": "Numerical champion: {x}",
    "Rinde observado y tendencia tecnológica": "Observed yield and technological trend",
    "Historia normalizada a tecnología {y}": "History normalized to {y} technology",
    "Actuarial Validation Score (mejor por modelo)": "Actuarial Validation Score (best per model)",
    "Peso histórico por campaña (half-life)": "Historical weight by season (half-life)",
    "Historical Information Value disponible en modo FULL.":
        "Historical Information Value available in FULL mode.",
    "Diagnósticos actuariales (advanced)": "Actuarial diagnostics (advanced)",
    "Observado": "Observed", "Trend seleccionado": "Selected trend",

    # --- Riesgo (nueva) ---
    "Yield Risk Distribution Engine": "Yield Risk Distribution Engine",
    "Distribución de riesgo tail-first: volatilidad · KDE benchmark · masa en cero · backtest asegurador":
        "Tail-first risk distribution: volatility · KDE benchmark · zero mass · insurance backtest",
    "Nivel de agregación": "Aggregation level",
    "Los datos fuente son departamentales; niveles más finos aplican priors de volatilidad y P(Y=0) configurables.":
        "Source data is department-level; finer levels apply configurable volatility and P(Y=0) priors.",
    "Distribución": "Distribution", "Volatilidad": "Volatility",
    "P(Y=0)": "P(Y=0)", "a nivel {x}": "at {x} level",
    "Prob. falla catastrófica": "Catastrophic failure prob.",
    "P(Y < 25% del esperado)": "P(Y < 25% of expected)",
    "Campeón AIC": "AIC champion",
    "¿Por qué este modelo de riesgo?": "Why this risk model?",
    "Distribución de rindes simulada — zoom cola inferior":
        "Simulated yield distribution — lower-tail zoom",
    "Densidad completa": "Full density", "Zoom cola (≤ P30)": "Tail zoom (≤ P30)",
    "Curva de siniestros: prima pura vs nivel de cobertura":
        "Claim curve: pure premium vs coverage level",
    "modelo": "model", "histórico": "historical",
    "Nivel de cobertura (% del rinde esperado)": "Coverage level (% of expected yield)",
    "Prima pura (% de la garantía)": "Pure premium (% of guarantee)",
    "Volatilidad downside vs upside": "Downside vs upside volatility",
    "Ranking de distribuciones (score asegurador)": "Distribution ranking (insurance score)",
    "Backtest por nivel de cobertura": "Backtest by coverage level",
    "Frecuencia de siniestro: predicha vs observada": "Claim frequency: predicted vs observed",
    "rinde simulado (kg/ha)": "simulated yield (kg/ha)",
    "Masa en cero (P(Y=0) = {p:.2%}) mostrada como barra discreta — no se esconde en la densidad.":
        "Zero mass (P(Y=0) = {p:.2%}) shown as a discrete bar — not hidden inside the density.",
    "Advertencias": "Warnings",


    # --- pipeline reordenado ---
    "📊 Resultados — cartera": "📊 Results — portfolio",
    "1️⃣ Datos": "1️⃣ Data", "2️⃣ Exploración": "2️⃣ Exploration",
    "3️⃣ Normalización (trend)": "3️⃣ Normalization (trend)",
    "4️⃣ Riesgo + Monte Carlo": "4️⃣ Risk + Monte Carlo",
    "5️⃣ Pricing": "5️⃣ Pricing",
    "Pipeline: 1 datos → 2 exploración → 3 trend/normalización → 4 distribución de riesgo y MC → 5 pricing. Resultados consolida la cartera.":
        "Pipeline: 1 data → 2 exploration → 3 trend/normalization → 4 risk distribution & MC → 5 pricing. Results consolidates the portfolio.",
    "Paso 1 de 5 · fuente MAGyP → dataset parquet listo para modelar":
        "Step 1 of 5 · MAGyP source → parquet dataset ready for modelling",
    "Paso 2 de 5 · mirar la serie cruda antes de modelar":
        "Step 2 of 5 · look at the raw series before modelling",
    "Paso 3 de 5 · separar tecnología μ(t) del shock agrícola — acá NO se usa kernel: el KDE es solo para la distribución de shocks (paso 4)":
        "Step 3 of 5 · separate technology μ(t) from the agricultural shock — NO kernel here: KDE only models the shock distribution (step 4)",
    "Paso 4 de 5 · distribución de shocks (KDE benchmark, gates de cola) + masa en cero + Monte Carlo — reemplaza al módulo AIC viejo":
        "Step 4 of 5 · shock distribution (KDE benchmark, tail gates) + zero mass + Monte Carlo — supersedes the old AIC module",
    "Paso 5 de 5 · del rinde al precio: prima pura por nivel de garantía":
        "Step 5 of 5 · from yield to price: pure premium by guarantee level",
    "AUTO (motor)": "AUTO (engine)", "Manual (didáctico)": "Manual (didactic)",
    "Motor AUTO (recomendado)": "AUTO engine (recommended)",
    "Manual (legacy)": "Manual (legacy)",
    "Versión manual: elegís vos el modelo de trend y ves el efecto. Para tarifar usá siempre AUTO.":
        "Manual version: you pick the trend model and see the effect. For rating always use AUTO.",
    "Módulo previo al motor: ajuste AIC simple sobre la serie detrendeada, sin gates ni masa en cero. Se mantiene para comparar.":
        "Pre-engine module: plain AIC fitting on the detrended series, no gates, no zero mass. Kept for comparison.",
    "Detrending manual": "Manual detrending",
    "Simulación Monte Carlo manual": "Manual Monte Carlo simulation",


    # --- modo uniforme + pricing buildup ---
    "Metodología uniforme de cartera": "Uniform portfolio methodology",
    "Aplica la metodología modal de la provincia (modelo de trend + modo) a todos los departamentos, re-estimando parámetros localmente. Compara la prima contra AUTO por departamento.":
        "Applies the province's modal methodology (trend model + mode) to every department, re-estimating parameters locally. Compares the premium against per-department AUTO.",
    "Metodología uniforme vs AUTO por departamento": "Uniform methodology vs per-department AUTO",
    "Metodología modal aplicada a toda la provincia: **{m}** · departamentos re-estimados: {n} ({lst})":
        "Modal methodology applied province-wide: **{m}** · re-estimated departments: {n} ({lst})",
    "Prima técnica (uniforme)": "Technical premium (uniform)",
    "Pérdida CAT (uniforme)": "CAT loss (uniform)",
    "Tasa técnica media (uniforme)": "Average technical rate (uniform)",
    "AUTO sigue siendo la referencia (respeta trends locales con evidencia fuerte); la uniforme sirve como sensibilidad de governance y para discutir con reaseguro.":
        "AUTO remains the reference (it respects local trends with strong evidence); the uniform run serves as a governance sensitivity and for reinsurance discussions.",
    "Construcción de tasa transparente: qué aporta cada fuente (histórico observado → modelo simulado → recargos) y por qué.":
        "Transparent rate buildup: what each source contributes (observed history → simulated model → loadings) and why.",
    "Default: E[rinde] del motor. Si lo cambiás, las simulaciones se re-escalan proporcionalmente.":
        "Default: engine expected yield. If you change it, simulations rescale proportionally.",
    "Tasa pura observada": "Observed pure rate",
    "Tasa pura simulada": "Simulated pure rate",
    "Tasa técnica": "Technical rate", "Prima técnica": "Technical premium",
    "Burning cost histórico: promedio del loss cost sobre la serie normalizada por el motor ({n} campañas). Es lo que efectivamente pasó, a tecnología actual.":
        "Historical burning cost: average loss cost over the engine-normalized series ({n} seasons). What actually happened, at current technology.",
    "Monte Carlo ({n} sims) de la distribución seleccionada por el motor de riesgo ({d}), incluyendo masa en cero (P(Y=0)={p:.2%}) y la cola completa — no solo los años que tocaron pasar.":
        "Monte Carlo ({n} sims) from the risk engine's selected distribution ({d}), including zero mass (P(Y=0)={p:.2%}) and the full tail — not only the years that happened to occur.",
    "pura simulada / (1 − deductions − margin) = {t:.2%} / {l:.2f}":
        "simulated pure / (1 − deductions − margin) = {t:.2%} / {l:.2f}",
    "SA = garantía × precio = {sa:,.0f} USD/ha": "SI = guarantee × price = {sa:,.0f} USD/ha",
    "Construcción de la tasa": "Rate buildup",
    "1 · Tasa pura observada (burning cost)": "1 · Observed pure rate (burning cost)",
    "Serie normalizada del motor · {n} campañas · frecuencia {f:.0%}":
        "Engine-normalized series · {n} seasons · frequency {f:.0%}",
    "2 · Ajuste por modelo de riesgo": "2 · Risk-model adjustment",
    "Distribución {d} + P(Y=0)={p:.2%}: completa la cola que la muestra finita no vio (o suaviza la que sobre-representó)":
        "Distribution {d} + P(Y=0)={p:.2%}: completes the tail the finite sample never saw (or smooths what it over-represented)",
    "3 · Tasa pura simulada": "3 · Simulated pure rate",
    "MC {n} sims · frecuencia {f:.1%} · severidad {s:.1%}":
        "MC {n} sims · frequency {f:.1%} · severity {s:.1%}",
    "4 · Recargo estructura": "4 · Structural loading",
    "1 − deductions ({d:.0%}) − margen de riesgo ({m:.0%})":
        "1 − deductions ({d:.0%}) − risk margin ({m:.0%})",
    "5 · Tasa técnica final": "5 · Final technical rate",
    "Se detiene antes de recargos comerciales (gastos de venta, utilidad, reaseguro) — spec §88":
        "Stops before commercial loadings (acquisition costs, profit, reinsurance) — spec §88",
    "Curva de garantías — observado vs modelo": "Guarantee curve — observed vs model",
    "tasa pura observada": "observed pure rate", "tasa pura simulada": "simulated pure rate",
    "garantía elegida": "chosen guarantee",
    "Tasa pura (% de la garantía)": "Pure rate (% of guarantee)",
    "Dónde separan las curvas es donde el modelo aporta: en garantías bajas manda la cola (y los ceros); en garantías altas ambas convergen porque los siniestros leves sí están bien representados en la muestra.":
        "Where the curves separate is where the model adds value: at low guarantees the tail (and zeros) dominate; at high guarantees both converge because mild claims are well represented in the sample.",
    "Loss cost por campaña (observado)": "Loss cost by season (observed)",


    # --- pricing lab ---
    "Modo": "Mode", "Guiado": "Guided", "Experto": "Expert",
    "Guiado agrega la capa educativa (conceptos, fórmulas, experimentos). Experto la oculta. El cálculo es EXACTAMENTE el mismo.":
        "Guided adds the educational layer (concepts, formulas, experiments). Expert hides it. The calculation is EXACTLY the same.",
    "Datos": "Data", "Trend": "Trend", "Distribución": "Distribution",
    "Garantía": "Guarantee", "Cola": "Tail", "Simulación": "Simulation",
    "Loss Cost": "Loss Cost", "Incertidumbre": "Uncertainty", "Tasa": "Rate",
    "tasa pura actual": "current pure rate",
    "Concepto": "Concept", "Por qué importa": "Why it matters",
    "Impacto en la tasa": "Impact on rate",
    "Advertencia actuarial": "Actuarial warning", "Glosario": "Glossary",
    "Garantía propuesta (para diagnóstico de credibilidad)":
        "Proposed guarantee (for credibility diagnostics)",
    "Campañas totales": "Total seasons",
    "Campañas en la cola asegurada": "Seasons in the insured tail",
    "Rindes cero": "Zero yields",
    "Rindes por debajo de garantía×mediana. Solo estas observaciones informan directamente el precio del seguro.":
        "Yields below guarantee×median. Only these observations directly inform the insurance price.",
    "**CREDIBILIDAD BAJA en la cola**: con {n} observaciones bajo la garantía, un promedio estable NO implica una cola estable. La prima descansa en muy pocos años.":
        "**LOW TAIL CREDIBILITY**: with {n} observations below the guarantee, a stable average does NOT imply a stable tail. The premium rests on very few years.",
    "garantía propuesta": "proposed guarantee",
    "Los años EXTREMOS no son outliers a limpiar: pueden ser la razón principal por la que existe este seguro.":
        "EXTREME years are not outliers to clean: they may be the main reason this insurance exists.",
    "GARANTÍA 70%": "70% GUARANTEE", "región asegurada": "insured region",
    "Experimento: bandwidth del kernel (no afecta la tarifa)":
        "Experiment: kernel bandwidth (does not affect the rate)",
    "SUB-SUAVIZADO ← bandwidth → SOBRE-SUAVIZADO · La tarifa real usa el bandwidth elegido estadísticamente (LOO-likelihood); este slider es para VER por qué no es una perilla cosmética.":
        "UNDER-SMOOTHED ← bandwidth → OVER-SMOOTHED · The actual rate uses the statistically selected bandwidth (LOO-likelihood); this slider is to SEE why it is not a cosmetic knob.",
    "Prima pura (garantía 70%)": "Pure premium (70% guarantee)",
    "shocks históricos": "historical shocks",
    "Descomposición: frecuencia × severidad": "Decomposition: frequency × severity",
    "Experimento mental: si la cobertura sube de 70% a 80%, ¿qué esperás de la tasa pura?":
        "Thought experiment: if coverage rises from 70% to 80%, what do you expect from the pure rate?",
    "Baja": "It falls", "Queda parecida": "About the same",
    "Sube más que proporcionalmente": "It rises more than proportionally",
    "Con esta serie: 70% → {a:.2%} · 80% → {b:.2%} ({x:.1f}×). Sube MÁS que proporcionalmente: al subir la garantía entran a la región asegurada los años moderadamente malos, que son muchos más que los extremos.":
        "With this series: 70% → {a:.2%} · 80% → {b:.2%} ({x:.1f}×). It rises MORE than proportionally: raising the guarantee pulls moderately bad years into the insured region, and there are many more of those than extremes.",
    "Incertidumbre de la tasa": "Rate uncertainty",
    "El modelo estima el riesgo; no observa el riesgo verdadero.":
        "The model estimates the risk; it does not observe the true risk.",
    "Intervalo bootstrap 90%": "Bootstrap 90% interval",
    "re-muestreos de la historia, re-ajustando la distribución seleccionada":
        "resamples of history, re-fitting the selected distribution",
    "Estabilidad leave-one-year-out": "Leave-one-year-out stability",
    "campaña quitada": "season removed",
    "tasa sin esa campaña": "rate without that season",
    "base {b:.2%}": "base {b:.2%}",
    "Rango: {mn:.2%} – {mx:.2%} · campaña más influyente: {y} ({d:+.2%})":
        "Range: {mn:.2%} – {mx:.2%} · most influential season: {y} ({d:+.2%})",
    "Model risk: la tasa según cada metodología": "Model risk: the rate under each methodology",
    "min {mn:.2%} · mediana {md:.2%} · max {mx:.2%} — esta dispersión es MODEL RISK: distinta del riesgo de datos (bootstrap) y del de proceso.":
        "min {mn:.2%} · median {md:.2%} · max {mx:.2%} — this spread is MODEL RISK: distinct from data risk (bootstrap) and process risk.",
    "Controles de sanidad actuarial": "Actuarial sanity checks",
    "una cobertura mayor debería costar más": "higher coverage should cost more",
    "{n} de {t} campañas por debajo de la garantía — solo esas informan directamente el precio":
        "{n} of {t} seasons below the guarantee — only those directly inform the price",
    "quitar {y} mueve la tasa {d:+.2%} ({r:.0%} de la base)":
        "removing {y} moves the rate {d:+.2%} ({r:.0%} of base)",
    "{n} candidato(s) rechazados por comprimir la cola — el seleccionado los pasó":
        "{n} candidate(s) rejected for compressing the tail — the selected one passed",
    "Monotonicidad: tasa crece con la cobertura": "Monotonicity: rate grows with coverage",
    "Sin rindes simulados negativos": "No negative simulated yields",
    "Observaciones en la cola asegurada": "Observations in the insured tail",
    "¿Domina la prima una sola campaña?": "Does a single season dominate the premium?",
    "Gates de subdispersión de riesgo": "Risk underdispersion gates",
    "Tamaño de muestra efectivo": "Effective sample size",
    "Ver fórmulas con los valores reales": "Show formulas with actual values",
    "La historia de tu tasa": "Your rate story",
    "Tu cobertura del {c:.0%} garantiza {g:,.0f} kg/ha. Bajo la distribución seleccionada ({d}), el {p:.0%} de las campañas simuladas cae por debajo de esa garantía; cuando hay siniestro, el faltante promedio es {s:,.0f} kg/ha ({sev:.0%} de la garantía). En {n:,} campañas simuladas eso produce una indemnización esperada del {t:.2%} de la responsabilidad — la tasa pura. El bootstrap indica un rango plausible de {lo:.2%}–{hi:.2%}, reflejo de la historia limitada; y entre metodologías defendibles la tasa va de {mn:.2%} a {mx:.2%}. La mayor fuente de sensibilidad es la representación de la cola inferior.":
        "Your {c:.0%} coverage guarantees {g:,.0f} kg/ha. Under the selected distribution ({d}), {p:.0%} of simulated seasons fall below that guarantee; when a claim occurs, the average shortfall is {s:,.0f} kg/ha ({sev:.0%} of the guarantee). Across {n:,} simulated seasons this produces an expected indemnity of {t:.2%} of liability — the pure rate. Bootstrap indicates a plausible range of {lo:.2%}–{hi:.2%}, reflecting limited history; and across defensible methodologies the rate spans {mn:.2%} to {mx:.2%}. The largest source of sensitivity is the representation of the lower tail.",

    # --- otras páginas ---
    "Análisis de rindes": "Yield analysis",
    "Estadística descriptiva": "Descriptive statistics",
    "Peores campañas": "Worst seasons",
    "Detrending": "Detrending",
    "Pricing — Yield Shortfall": "Pricing — Yield Shortfall",
    "Curva de garantías": "Guarantee curve",
    "Loss cost por campaña": "Loss cost by season",
    "Simulación Monte Carlo": "Monte Carlo simulation",
    "Ajuste de distribuciones (AIC)": "Distribution fitting (AIC)",
    "Datos": "Data", "Actualizar desde MAGyP": "Update from MAGyP",
    "Descargar dataset completo": "Download full dataset",
    "Descargar serie filtrada (CSV)": "Download filtered series (CSV)",
    "No hay datos para esa combinación de filtros.":
        "No data for that filter combination.",
    "Rinde esperado (kg/ha)": "Expected yield (kg/ha)",
    "Cobertura (%)": "Coverage (%)", "Deducible (%)": "Deductible (%)",
    "Gastos": "Expenses", "Margen de riesgo": "Risk margin",
    "Simulaciones": "Simulations", "Método": "Method",
}


def tr(_s: str, **kw) -> str:
    lang = st.session_state.get("lang", "es")
    out = EN.get(_s, _s) if lang == "en" else _s
    return out.format(**kw) if kw else out
