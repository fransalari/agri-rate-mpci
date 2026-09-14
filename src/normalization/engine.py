"""YIELD RISK NORMALIZATION ENGINE — orquestador.

Pipeline (spec §71):
    YIELD HISTORY
    → fit de candidatos (modelo × modo × start year × half-life)
    → rolling out-of-sample validation (sin leakage)
    → ActuarialValidationScore (+ complejidad, estabilidad, residuos)
    → one-standard-error parsimony rule
    → mejor configuración de normalización + historia normalizada
      + rinde esperado + incertidumbre + explicación.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from .config import YieldNormalizationConfig
from .data_quality import data_quality_report, detect_structural_break
from .diagnostics import (agronomic_guardrails, diagnostics_summary,
                          residual_diagnostics)
from .schemas import (CandidateResult, ModelConfiguration,
                      NormalizationResult, governance_metadata)
from .validation import (effective_sample_size, fit_full, half_life_weights,
                         instability_penalty, run_cv, validation_years)

log = logging.getLogger("agro_rate.normalization")


class YieldNormalizationEngine:
    def __init__(self, config: YieldNormalizationConfig | None = None):
        self.config = config or YieldNormalizationConfig()

    # ------------------------------------------------------------------
    def run(self, yield_series: pd.DataFrame, *, crop: str = "",
            location: str = "", target_year: int | None = None,
            manual: ModelConfiguration | None = None) -> NormalizationResult:
        """Entrada: DataFrame con columnas year|anio y yield|rendimiento_kgxha."""
        cfg = self.config
        t0 = time.perf_counter()
        rng = np.random.default_rng(cfg.random_seed)

        series = _coerce_input(yield_series)
        years = series["year"].to_numpy()
        y = series["yield"].to_numpy(dtype=float)
        target = int(target_year if target_year is not None else years.max() + 1)
        log.info("run start: %s/%s n=%d target=%d mode=%s",
                 crop, location, len(y), target, cfg.execution_mode)

        result = NormalizationResult(status="OK", selection_mode=cfg.selection_mode,
                                     execution_mode=cfg.execution_mode,
                                     target_year=target, raw_years=len(y))
        result.data_quality = data_quality_report(years, y)
        errors = [i for i in result.data_quality
                  if i["level"] == "DATA_QUALITY_ERROR"]
        if errors:
            result.warnings += [f"calidad de datos: {e['detail']}" for e in errors]

        if len(y) < cfg.minimum_series_length or np.std(y) == 0:
            result.status = "INSUFFICIENT_DATA"
            result.warnings.append(
                f"serie insuficiente para modelar (n={len(y)}); "
                "no se inventa precisión (spec §65)")
            return result

        # ---------------- grid de candidatos --------------------------
        val_years = validation_years(years, cfg.minimum_training_years,
                                     cfg.max_validation_years)
        if len(val_years) < 4:
            result.warnings.append("ventana de validación muy corta")
        candidates = self._evaluate_grid(years, y, val_years, target,
                                         manual=manual)
        result.candidates = candidates
        ok = [c for c in candidates if c.status == "OK"]
        log.info("candidatos: %d evaluados, %d válidos", len(candidates), len(ok))
        if not ok:  # fallback seguro (spec §65)
            ok = [self._fallback_candidate(years, y, val_years, target)]
            result.warnings.append("todos los candidatos fallaron: fallback LINEAR")

        # ---------------- selección: champion + 1-SE -------------------
        ok.sort(key=lambda c: c.score)
        champion = ok[0]
        threshold = champion.score + champion.score_se
        final_set = [c for c in ok if c.score <= threshold]
        selected = _parsimony_pick(final_set, self.config)
        reasons = _explain(champion, selected, final_set)

        result.numerical_champion = champion.config
        result.numerical_champion_score = champion.score
        result.recommended = selected.config
        result.validation_score = selected.score
        result.score_se = selected.score_se
        result.tail_score = selected.tail_error
        result.aicc, result.bic = selected.aicc, selected.bic
        result.selection_reason = reasons
        result.model_ranking = _ranking_table(ok, selected, champion)

        # ---------------- fit final + normalización -------------------
        sc = selected.config
        full = fit_full(years, y, model_name=sc.trend_model,
                        mode=sc.detrending_mode, start_year=sc.start_year,
                        half_life=sc.half_life_years, config=cfg,
                        ref_year=years.max())
        trend_target = float(full["model"].predict(np.array([target]))[0])
        result.expected_yield = trend_target

        if sc.detrending_mode == "multiplicative":
            normalized = full["shocks"] * trend_target
        else:
            normalized = trend_target + full["shocks"]
        w_target = half_life_weights(full["years"], ref_year=float(target),
                                     half_life=sc.half_life_years)
        result.normalized_history = pd.DataFrame({
            "year": full["years"],
            "observed_yield": full["yields"],
            "historical_trend": full["trend"],
            "relative_shock": full["shocks"],
            "normalized_yield": np.maximum(normalized, 0.0),
            "historical_weight": w_target / w_target.max(),
        })
        result.effective_sample_size = effective_sample_size(w_target)
        result.sample_status = ("FAVORABLE" if result.effective_sample_size >= cfg.n_eff_favorable
                                else "MODERATE" if result.effective_sample_size >= cfg.minimum_effective_sample_size
                                else "LOW")
        if result.sample_status == "LOW":
            result.warnings.append(
                f"N_eff={result.effective_sample_size:.1f} < "
                f"{cfg.minimum_effective_sample_size:g}: baja confianza, "
                "especialmente para colas")

        if target - years.max() > cfg.max_extrapolation_years:
            result.warnings.append(
                f"target {target} extrapola {target - years.max()} años "
                "más allá del último dato")

        # ---------------- diagnósticos / breaks ------------------------
        diags = residual_diagnostics(full["years"], full["shocks"],
                                     sc.detrending_mode)
        result.diagnostics = diagnostics_summary(
            diags, result.effective_sample_size, cfg)
        result.diagnostics_detail = diags
        if cfg.structural_break_enabled:
            result.structural_break = detect_structural_break(
                full["years"], full["shocks"], cfg.break_alpha)

        # ---------------- bootstrap: CI + estabilidad ------------------
        boot = self._bootstrap(years, y, ok, selected, target, rng)
        result.expected_yield_ci = boot["ci"]
        result.selection_stability_pct = boot["stability_pct"]

        # ---------------- HIV (solo FULL) ------------------------------
        if (cfg.historical_information_value_enabled
                and cfg.execution_mode == "FULL"):
            result.historical_information_value = self._hiv(
                years, y, selected, val_years)

        # ---------------- confianza ------------------------------------
        result.selection_confidence = _confidence(result, cfg)
        result.governance = governance_metadata(
            series, cfg, selected.config, champion.config, reasons)
        result.timings = {"total_s": round(time.perf_counter() - t0, 2),
                          "n_candidates": len(candidates)}
        log.info("run done in %.2fs: recomendado=%s score=%.4f conf=%s",
                 result.timings["total_s"], sc.label(),
                 selected.score, result.selection_confidence)
        return result

    # ------------------------------------------------------------------
    def _grid(self, years: np.ndarray) -> list[ModelConfiguration]:
        cfg = self.config
        models, half_lives = cfg.effective_candidates()
        y0, y1 = int(years.min()), int(years.max())
        starts = list(range(y0, y1 - cfg.minimum_training_years + 1,
                            cfg.candidate_start_year_step)) or [y0]
        grid = [ModelConfiguration(m, mode, s, h)
                for m in models
                for mode in cfg.candidate_detrending_modes
                for s in starts
                for h in half_lives]
        return grid

    def _evaluate_grid(self, years, y, val_years, target, manual=None
                       ) -> list[CandidateResult]:
        cfg = self.config
        grid = ([manual] if (cfg.selection_mode == "MANUAL" and manual)
                else self._grid(years))
        if cfg.selection_mode == "BENCHMARK" and manual:
            grid = self._grid(years) + [manual]
        out = []
        for mc in grid:
            out.append(self._evaluate_one(years, y, mc, val_years, target))
        return out

    def _evaluate_one(self, years, y, mc: ModelConfiguration, val_years,
                      target) -> CandidateResult:
        cfg = self.config
        cand = CandidateResult(config=mc)
        try:
            cv = run_cv(years, y, model_name=mc.trend_model,
                        mode=mc.detrending_mode, start_year=mc.start_year,
                        half_life=mc.half_life_years, val_years=val_years,
                        config=cfg)
        except RuntimeError as exc:
            cand.status, cand.reason = "FAILED", str(exc)
            return cand
        if cv is None:
            cand.status = "INELIGIBLE"
            cand.reason = "entrenamiento insuficiente en la ventana común (spec §18)"
            return cand

        full = fit_full(years, y, model_name=mc.trend_model,
                        mode=mc.detrending_mode, start_year=mc.start_year,
                        half_life=mc.half_life_years, config=cfg)
        rejected = agronomic_guardrails(full["model"], full["years"], cfg,
                                        mc.detrending_mode)
        if rejected:
            cand.status, cand.reason = "REJECTED", "; ".join(rejected)
            return cand

        diags = residual_diagnostics(full["years"], full["shocks"],
                                     mc.detrending_mode)
        resid_pen = np.mean([
            {"PASS": 0.0, "WARNING": 0.5, "FAIL": 1.0}[d["status"]]
            for d in diags.values()])
        inst = instability_penalty(
            years, y, model_name=mc.trend_model, mode=mc.detrending_mode,
            start_year=mc.start_year, half_life=mc.half_life_years,
            target_year=target, config=cfg)
        comp = cfg.complexity_penalty(mc.trend_model)

        w = cfg.score_weights
        cand.prediction_error = cv["prediction_error"]
        cand.tail_error = cv["tail_error"]
        cand.distribution_error = cv["distribution_error"]
        cand.residual_penalty = float(resid_pen)
        cand.complexity_penalty = comp
        cand.instability_penalty = inst
        cand.fold_scores = cv["fold_scores"]
        cand.score_se = cv["score_se"]
        cand.score = (w.prediction * cand.prediction_error
                      + w.tail * cand.tail_error
                      + w.distribution * cand.distribution_error
                      + w.residual * cand.residual_penalty
                      + w.complexity * comp
                      + w.instability * inst)
        ic = full["model"].information_criteria(full["years"], full["yields"],
                                                full["weights"])
        cand.aicc, cand.bic = ic["aicc"], ic["bic"]
        cand.n_eff = effective_sample_size(full["weights"])
        cand.n_train = len(full["years"])
        return cand

    def _fallback_candidate(self, years, y, val_years, target) -> CandidateResult:
        mc = ModelConfiguration("linear", "additive", int(years.min()),
                                float("inf"))
        cand = self._evaluate_one(years, y, mc, val_years, target)
        if cand.status != "OK":
            mc = ModelConfiguration("constant", "additive", int(years.min()),
                                    float("inf"))
            cand = self._evaluate_one(years, y, mc, val_years, target)
        return cand

    # ------------------------------------------------------------------
    def _bootstrap(self, years, y, ok, selected, target, rng) -> dict:
        """Estabilidad de la selección (resampleo de folds, spec §39) y
        CI del rinde esperado (resampleo de shocks, spec §37–38)."""
        cfg = self.config
        n_iter = cfg.bootstrap_iterations

        # -- estabilidad: re-selección con folds bootstrapeados ----------
        wins = 0
        n_folds = len(selected.fold_scores)
        # contendientes: banda amplia alrededor del champion (cubre todo el
        # conjunto 1-SE, incluido el seleccionado por parsimonia)
        champ = min(ok, key=lambda c: c.score)
        band = champ.score + 2 * max(champ.score_se, selected.score_se)
        contenders = [c for c in ok
                      if len(c.fold_scores) == n_folds and c.score <= band]
        if selected not in contenders:
            contenders.append(selected)
        contenders = sorted(contenders, key=lambda c: c.score)[:40]
        base_fixed = {id(c): (c.score - float(np.mean(c.fold_scores)))
                      for c in contenders}
        for _ in range(n_iter):
            idx = rng.integers(0, n_folds, n_folds)
            best, best_score = None, np.inf
            scores = {}
            for c in contenders:
                s = base_fixed[id(c)] + float(np.mean(np.asarray(c.fold_scores)[idx]))
                scores[id(c)] = s
                if s < best_score:
                    best, best_score = c, s
            se_b = float(np.std(np.asarray(best.fold_scores)[idx], ddof=1)
                         / np.sqrt(n_folds)) if n_folds > 1 else 0.0
            fset = [c for c in contenders if scores[id(c)] <= best_score + se_b]
            pick = _parsimony_pick_by(fset, scores, self.config)
            if pick.config.trend_model == selected.config.trend_model:
                wins += 1
        stability = 100.0 * wins / max(n_iter, 1)

        # -- CI del expected yield: resampleo de shocks ------------------
        sc = selected.config
        full = fit_full(years, y, model_name=sc.trend_model,
                        mode=sc.detrending_mode, start_year=sc.start_year,
                        half_life=sc.half_life_years, config=cfg)
        preds = []
        n = len(full["years"])
        for _ in range(n_iter):
            idx = rng.integers(0, n, n)
            if sc.detrending_mode == "multiplicative":
                y_b = full["trend"] * full["shocks"][idx]
            else:
                y_b = full["trend"] + full["shocks"][idx]
            try:
                fb = fit_full(full["years"], np.maximum(y_b, 0.0),
                              model_name=sc.trend_model,
                              mode=sc.detrending_mode,
                              start_year=sc.start_year,
                              half_life=sc.half_life_years, config=cfg)
                preds.append(float(fb["model"].predict(np.array([target]))[0]))
            except Exception:
                continue
        ci = (tuple(np.percentile(preds, [5, 95])) if len(preds) >= 20
              else (float("nan"), float("nan")))
        return {"stability_pct": stability, "ci": (float(ci[0]), float(ci[1]))}

    # ------------------------------------------------------------------
    def _hiv(self, years, y, selected, val_years) -> pd.DataFrame:
        """Historical Information Value por bloques (spec §35):
        HIV = Score_sin_bloque − Score_con_bloque (positivo ⇒ el bloque
        ayuda). Se recalcula el CV excluyendo el bloque del training."""
        cfg = self.config
        sc = selected.config
        rows = []
        y0 = int(years.min())
        first_val = int(val_years.min()) if len(val_years) else int(years.max())
        for b0 in range(y0, first_val, cfg.hiv_block_years):
            b1 = min(b0 + cfg.hiv_block_years - 1, first_val - 1)
            mask = ~((years >= b0) & (years <= b1))
            if mask.sum() < cfg.minimum_training_years + len(val_years):
                continue
            try:
                cv = run_cv(years[mask], y[mask], model_name=sc.trend_model,
                            mode=sc.detrending_mode, start_year=sc.start_year,
                            half_life=sc.half_life_years,
                            val_years=val_years, config=cfg)
            except RuntimeError:
                cv = None
            if cv is None:
                continue
            w = cfg.score_weights
            score_wo = (w.prediction * cv["prediction_error"]
                        + w.tail * cv["tail_error"]
                        + w.distribution * cv["distribution_error"])
            score_with = (w.prediction * selected.prediction_error
                          + w.tail * selected.tail_error
                          + w.distribution * selected.distribution_error)
            hiv = score_wo - score_with
            rec = ("útil" if hiv > 0.005 else
                   "aporte marginal" if hiv > -0.005 else
                   "revisar (empeora calibración)")
            rows.append({"period": f"{b0}–{b1}",
                         "information_value": round(float(hiv), 4),
                         "recommendation": rec})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------- utils
def _coerce_input(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    cols = {c.lower(): c for c in d.columns}
    year_col = cols.get("year") or cols.get("anio")
    yield_col = cols.get("yield") or cols.get("rendimiento_kgxha") or cols.get("rinde")
    if not year_col or not yield_col:
        raise ValueError("se requieren columnas year/anio y yield/rendimiento_kgxha")
    d = (d[[year_col, yield_col]]
         .rename(columns={year_col: "year", yield_col: "yield"})
         .dropna())
    d["year"] = d["year"].astype(int)
    d = (d.groupby("year", as_index=False)["yield"].mean()
         .sort_values("year").reset_index(drop=True))
    return d


def _parsimony_pick(final_set: list[CandidateResult], cfg) -> CandidateResult:
    scores = {id(c): c.score for c in final_set}
    return _parsimony_pick_by(final_set, scores, cfg)


def _parsimony_pick_by(final_set, scores, cfg) -> CandidateResult:
    """Orden del spec §33: complejidad → estabilidad → tail → N_eff →
    historia → AICc → score."""
    def key(c: CandidateResult):
        return (cfg.complexity_penalty(c.config.trend_model),
                round(c.instability_penalty, 3) if np.isfinite(c.instability_penalty) else 9,
                round(c.tail_error, 4) if np.isfinite(c.tail_error) else 9,
                -c.n_eff if np.isfinite(c.n_eff) else 0,
                c.config.start_year,
                round(c.aicc, 1) if np.isfinite(c.aicc) else 9e9,
                scores[id(c)])
    return sorted(final_set, key=key)[0]


def _explain(champion, selected, final_set) -> list[str]:
    reasons = [
        f"{champion.config.trend_model} obtuvo el menor score numérico "
        f"({champion.score:.4f} ± {champion.score_se:.4f}).",
        f"{len(final_set)} configuraciones quedaron dentro de un error "
        "estándar del campeón (regla one-SE, spec §33).",
    ]
    if selected.config == champion.config:
        reasons.append("El campeón numérico también fue el más parsimonioso "
                       "del conjunto final, por lo que fue seleccionado.")
    else:
        reasons += [
            f"{selected.config.trend_model} quedó dentro del mismo error "
            f"estándar (score {selected.score:.4f}) con menor complejidad "
            "y/o mejor estabilidad y calibración de cola.",
            "No hay evidencia suficiente de que la mejora marginal del "
            "campeón justifique su complejidad adicional "
            "(simplest statistically equivalent model, spec §34).",
        ]
    reasons.append(
        f"Configuración final: {selected.config.label()} "
        f"(tail {selected.tail_error:.4f}, N_eff {selected.n_eff:.1f}).")
    return reasons


def _ranking_table(ok, selected, champion) -> pd.DataFrame:
    rows = []
    for i, c in enumerate(sorted(ok, key=lambda x: x.score), 1):
        status = []
        if c.config == champion.config:
            status.append("Numerical champion")
        if c.config == selected.config:
            status.append("SELECTED")
        hl = "∞" if c.config.half_life_years == float("inf") else f"{c.config.half_life_years:g}"
        rows.append({
            "rank": i, "model": c.config.trend_model,
            "mode": c.config.detrending_mode[:4],
            "start": c.config.start_year, "half_life": hl,
            "cv_score": round(c.score, 4), "tail": round(c.tail_error, 4),
            "aicc": round(c.aicc, 1), "n_eff": round(c.n_eff, 1),
            "complexity": round(c.complexity_penalty, 3),
            "instability": round(c.instability_penalty, 3),
            "status": " · ".join(status) if status else "",
        })
    return pd.DataFrame(rows)


def _confidence(result: NormalizationResult, cfg) -> str:
    fails = sum(1 for v in result.diagnostics.values() if v == "FAIL")
    warns = sum(1 for v in result.diagnostics.values() if v == "WARNING")
    stab = result.selection_stability_pct / 100.0
    if (stab >= cfg.stability_high and fails == 0
            and result.sample_status == "FAVORABLE"):
        return "HIGH"
    if stab >= cfg.stability_medium and fails == 0 and warns <= 2 \
            and result.sample_status != "LOW":
        return "MEDIUM"
    return "LOW"
