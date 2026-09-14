# Yield Risk Normalization Engine — Informe final (v0.1.0)

## 1–2. Arquitectura encontrada e implementada

**Encontrada**: biblioteca actuarial Python (`src/`) + app Streamlit
(`app.py`); sin FastAPI/PostgreSQL/React. Ya existía un selector
automático de **distribuciones** por AIC (`src/simulation/distributions.py`)
— el análogo conceptual del spec §58 — y detrending manual
(`src/analytics/detrending.py`). No existía selección automática de trend:
no hubo duplicación.

**Implementada**: servicio desacoplado `src/normalization/` con interfaz
`YieldNormalizationEngine(config).run(serie, crop, location, target_year)`.
La "API" del spec (§48) es esta interfaz Python; el "frontend" (§49–55) es
la página Streamlit "🧠 Normalización" + la opción `auto (motor)` del
selector de detrending, que propaga la serie normalizada y el rinde
esperado a Pricing y Monte Carlo (§58). `detrending.py` queda como modo
MANUAL.

## 3–4. Archivos

**Nuevos**: `src/normalization/{__init__,config,schemas,trend_models,
validation,scoring→(en engine),data_quality,diagnostics,engine}.py`,
`tests/test_normalization.py`, este informe.
**Modificados**: `app.py` (opción auto + página + defaults), `README.md`.

## 5–7. Base de datos / dependencias / endpoints

Sin cambios de base (Parquet/DuckDB se mantienen). **Cero dependencias
nuevas**: robusto = `statsmodels.RLM` (Huber); spline natural cúbico
restringido implementado en numpy (lineal fuera del rango observado →
extrapolación segura). Endpoint = `engine.run(...)` (§66).

## 8–9. Metodología y lógica de selección

- **Candidatos** (§8–13): constant, linear, robust_linear (Huber),
  log_linear (con corrección de sesgo smearing), quadratic (solo FULL,
  guardrails), piecewise (1 breakpoint continuo, segmento mínimo 8),
  spline natural restringido (df=4 + ridge). Interfaz común
  `YieldTrendModel` (fit/predict/information_criteria/metadata).
- **Grid** (§16–17): modelo × modo (mult/add) × start year (paso 5) ×
  half-life {10,15,20,25,30,40,∞} (FAST: {15,25,∞}, sin quadratic).
- **Validación** (§6–7, §18): rolling-origin expanding, horizonte 1,
  ventana común de validación (últimas ≤12 campañas) idéntica para todos
  los candidatos; inelegibles los que no puedan entrenar
  `minimum_training_years` en el primer fold. Anti-leakage: pesos con
  referencia en el último año de training; test específico que altera el
  futuro y verifica predicciones pasadas idénticas.
- **Pesos históricos** (§19–22): `w=2^(−(T−t)/h)`; N_eff=(Σw)²/Σw²;
  status FAVORABLE ≥30 / MODERATE 20–30 / LOW <20 (warning extra para
  colas). Hard cutoff (start year) y decay compiten en el mismo grid (§21).
- **ActuarialValidationScore** (§26–31): 0.30·nRMSE(OOS) + 0.25·pinball
  P10/P20/P30 normalizado + 0.15·CRPS ponderado + 0.10·penalidad de
  residuos (trend residual, autocorrelación, varianza) + 0.10·complejidad
  (df/20, centralizado) + 0.10·inestabilidad (leave-out de los 3 shocks
  más extremos → Δ relativo del trend target). AICc/BIC calculados como
  secundarios (§15).
- **Selección** (§32–34): champion = argmin score; conjunto final =
  score ≤ champion + SE(champion) con SE de scores por fold; dentro del
  conjunto, orden §33: complejidad → inestabilidad → tail → N_eff →
  historia → AICc → score. "Simplest statistically equivalent model".
- **Breaks** (§23–24): Pettitt sobre shocks + Chow de corroboración;
  NONE/WEAK/MODERATE/STRONG → KEEP/DOWNWEIGHT/REVIEW; el hard cutoff solo
  se sugiere ante causas estructurales, nunca por un año extremo.
- **Data quality** (§25): errores vs warnings vs
  POTENTIAL_CLIMATE_EXTREME (z robusto sobre residuos detrendeados — un
  rinde bajísimo se conserva, no se borra).
- **Bootstrap** (§37–39): CI 90% del rinde esperado (resampleo de shocks
  + refit) y estabilidad de selección a nivel modelo (resampleo de folds
  + re-selección 1-SE); FAST 80 / FULL 400 iteraciones.
- **HIV** (§35, FULL): ΔScore al excluir bloques de 5 años del training
  con la misma ventana de validación.
- **Confianza** (§39): HIGH ≥75% estabilidad + sin FAIL + N_eff
  favorable; MEDIUM/LOW en cascada.
- **Fallback** (§64–65): candidato fallido → FAILED y sigue; todos
  fallan → LINEAR/CONSTANT; serie corta → `INSUFFICIENT_DATA`.

## 10–12. Tests

`tests/test_normalization.py`: 17 tests — casos A (linear), B (sequías/
robusto), C (crecimiento exponencial), D (break detectado ±3 años),
E (sin trend → constant), F (historia vieja irrelevante → start
tardío/half-life corto), G (extremo antiguo retenido y clasificado como
clima), H (muestra chica → conf ≠ HIGH), I (spline no gana sobre ruido,
FULL), J (regla 1-SE: robust sobre "GAM" en empate; campeón se mantiene
con brecha material), K (shocks multiplicativos → mult gana), L (aditivos
→ add puede ganar), anti-leakage (§7), INSUFFICIENT_DATA,
reproducibilidad + governance (§61), contrato de salida (§66).
**Resultado: 34/34 del repo (17 previos + 17 nuevos) en ~24 s.**

## 13–15. Ejemplo real (soja, Comandante Fernández, target 2026)

```python
from src.normalization import YieldNormalizationEngine
res = YieldNormalizationEngine().run(serie, crop="soja",
                                     location="Cte Fernández",
                                     target_year=2026)
```
Salida (FAST, 2 s, 324 configuraciones): recomendado
`constant · multiplicative · desde 1970 · h=∞`; E[2026]=1.490 kg/ha
(CI 1.347–1.577); N_eff 54 (FAVORABLE); estabilidad 100 %; confianza
HIGH; diagnósticos PASS; break STRONG 1986 → REVIEW. En sintéticos con
tendencia limpia el motor selecciona linear/log-linear y recupera el
nivel teórico (test A/C) — es decir, en las series reales de Chaco la
conclusión "un promedio reciente predice igual que un trend" es un
resultado, no una limitación.

## 16. Limitaciones conocidas

- SE del score con ~12 folds es ancho → el 1-SE favorece fuerte la
  parsimonia (por diseño del spec; el ranking siempre muestra el campeón).
- LOESS no incluido como candidato (predicción 1-paso mal definida);
  quadratic solo en FULL.
- CI del expected yield no incorpora incertidumbre de selección de
  modelo (solo del fit condicional al seleccionado).
- Breaks: Pettitt+Chow (un solo break); Bai-Perron múltiple queda para
  fase 2 (requeriría `ruptures`).

## 17. Performance

FAST: 300–360 configs, 1.9–2.6 s por serie (incluye bootstrap 80 y CI).
FULL: ~1.100 configs + bootstrap 400 + HIV, 15–35 s. Sin caching de
folds todavía (candidato natural de optimización fase 2).

## 18. Fase 2 sugerida

Modelo jerárquico (partial pooling entre departamentos, §59) — la firma
`run()` ya recibe crop/location para ello; caching de folds; Bai-Perron;
incertidumbre de selección en el CI; integración del normalized_history
como default silencioso del módulo Pricing.

## Tabla de requirements

| Requirement (spec) | Status | Implementación | Test |
|---|---|---|---|
| §0 inspección/reutilización | ✅ | informe §1–2; patrón AIC reutilizado | — |
| §1–3 aditivo y multiplicativo | ✅ | `validation.run_cv`, `fit_full` | K, L |
| §4 AUTO/MANUAL/BENCHMARK | ✅ | `config.selection_mode`, `engine._evaluate_grid` | contrato |
| §5–7 OOS primario, sin leakage | ✅ | rolling-origin; ref de pesos = último año train | `test_no_future_leakage` |
| §8–13 7 modelos, interfaz común | ✅ | `trend_models.py` (registry) | A–E, I |
| §14–15 complejidad + AICc/BIC secundarios | ✅ | `complexity_penalty`, `information_criteria` | J |
| §16–18 grid + ventana común | ✅ | `_grid`, elegibilidad §18 | implícito CV |
| §19–22 half-life, N_eff, cutoff vs decay | ✅ | `half_life_weights`, grid conjunto | F |
| §23–24 breaks clasificados | ✅ | `data_quality.detect_structural_break` | D |
| §25 DataQualityReport | ✅ | `data_quality_report` (error≠extremo) | G |
| §26–31 score 6 componentes | ✅ | `engine._evaluate_one` | A–L |
| §32–34 champion + 1-SE + simplicidad | ✅ | `_parsimony_pick` | J ×2 |
| §35 HIV | ✅ (FULL) | `engine._hiv` | smoke FULL (I) |
| §36–37 normalized history + target CI | ✅ | `normalized_history`, bootstrap CI | G, contrato |
| §38–39 bootstrap + estabilidad + confianza | ✅ | `engine._bootstrap`, `_confidence` | H |
| §40–43 schemas, ranking, explainability | ✅ | `schemas.py`, `_ranking_table`, `_explain` | governance/contrato |
| §44–46 config central, FAST/FULL, perf | ✅ | `config.py`, `effective_candidates` | tiempos §17 |
| §47–50 servicio + UI default/advanced | ✅ | `src/normalization/` + página + expander | Playwright |
| §51–55 charts 1–5 | ✅ | página Normalización (HIV en FULL) | Playwright |
| §56 guardrails agronómicos | ✅ | `agronomic_guardrails` (REJECTED) | I |
| §57 casos A–L | ✅ | `tests/test_normalization.py` | 17 tests |
| §58 compatibilidad Monte Carlo | ✅ | `detrended`=normalized alimenta `sampler()`/pricing | app |
| §59–60 fase 2 documentada / no implementar premium | ✅ | §18 de este informe | — |
| §61–62 governance + logging | ✅ | `governance_metadata`, logger | reproducibilidad |
| §63–65 deps mínimas, errores, fallback | ✅ | 0 deps nuevas; FAILED/INSUFFICIENT_DATA | fallback test |
| §66–68 criterio funcional + UX | ✅ | `run()` completo; card AUTO + WHY | contrato + Playwright |
