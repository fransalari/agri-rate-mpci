# 🌾 AGRO RATE

**A modular platform for agricultural risk analysis and insurance pricing.**

Filosofía: `Data → Risk → Loss → Pricing → Portfolio`

Aplicación Streamlit + biblioteca actuarial en Python para análisis de
rendimientos agrícolas y pricing de coberturas de rinde (MPCI / yield
shortfall), con datos oficiales de MAGyP a nivel departamento.

## Demo local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Sin configuración adicional: el repo incluye un dataset de ejemplo real
(`data/sample/chaco_girasol.csv` — girasol, Chaco, 1969–2024, 25
departamentos, fuente MAGyP).

## Módulos (V0.1)

| Módulo | Qué hace |
|---|---|
| 🏠 Dashboard | KPIs de la serie: rinde promedio, tendencia, CV, peor campaña |
| 🌱 Análisis de rindes | Serie histórica, estadística descriptiva, percentiles, histograma |
| 📈 Detrending | Linear / quadratic / LOESS / moving average — serie en "tecnología actual" |
| 🧠 Normalización | **Yield Risk Normalization Engine**: selección automática de modelo, modo (mult/add), start year y half-life con validación rolling-origin OOS, regla one-SE, structural breaks, bootstrap y explicación auditable (`REPORT_NORMALIZATION.md`) |
| 💰 Pricing | Burning cost, frecuencia, severidad, curva de garantías 50–90%, loss cost anual |
| 🎲 Monte Carlo | Bootstrap/KDE/paramétricas (AIC), VaR/TVaR 95-99, prima bruta con gastos y margen |
| 📦 Datos | Fuente activa, descarga de la serie filtrada, actualización desde MAGyP |

## Datos

- **Primaria — MAGyP Estimaciones Agrícolas**: descarga automática del
  dataset completo (`src/data/magyp.py`). Endpoint documentado en
  [`HALLAZGOS_CONECTORES.md`](HALLAZGOS_CONECTORES.md).
- **Secundaria — CEDEI Chaco** (Tableau Public, serie 1959–2024):
  validación cruzada y extensión histórica (`src/data/cedei.py`).
- Almacenamiento local en Parquet + consultas con DuckDB
  (`src/data/database.py`).

La GitHub Action [`update_data.yml`](.github/workflows/update_data.yml)
regenera `data/estimaciones.parquet` el día 3 de cada mes y commitea si
hay cambios → Streamlit Cloud redeploya automáticamente.

## Estructura

```
agro-rate/
├── app.py                      # aplicación Streamlit
├── src/
│   ├── data/                   # magyp.py · cedei.py · database.py
│   ├── analytics/              # detrending.py · statistics.py
│   ├── pricing/                # yield_insurance.py · burning_cost.py
│   └── simulation/             # distributions.py · monte_carlo.py
├── data/sample/                # dataset de ejemplo (girasol Chaco)
├── tests/                      # pytest (17 tests)
└── .github/workflows/          # actualización mensual de datos
```

## Tests

```bash
python -m pytest tests/ -v
```

## Publicación

1. Push del repo a GitHub.
2. [share.streamlit.io](https://share.streamlit.io) → New app → apuntar a
   `app.py` → URL pública `https://agro-rate.streamlit.app`.
3. Correr una vez la Action (workflow_dispatch) para generar el Parquet
   completo con todos los cultivos y provincias.

## Roadmap

- **V0.2** ✅ curva de garantías (ya incluida) · pricing curves por departamento
- **V0.3** deducibles/franquicias, límites, piso–paga, stop loss (el motor ya soporta deducible y límite)
- **V0.4** NDVI / TVDI / precipitación / temperatura → Climate Risk Explorer
- Portfolio: agregación, correlación espacial, PML de cartera
