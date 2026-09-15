"""YIELD RISK DISTRIBUTION ENGINE — orquestador (spec §46-56, §91).

Pipeline:
    serie cruda
    → normalización tecnológica (REUSA YieldNormalizationEngine: etapas
      trend + memoria histórica ya validadas out-of-sample)
    → clasificación de ceros → shocks positivos
    → modelo de volatilidad (evidencia, asimetría downside)
    → candidatos de distribución (KDE benchmark obligatorio)
    → backtest asegurador rolling (cola, frecuencia, severidad, prima)
    → GATES DE RIESGO PRIMERO, score después (§50)
    → 1-SE / ensemble / estabilidad
    → Monte Carlo final: masa en cero × distribución positiva ×
      multiplicador del nivel de agregación
    → métricas de riesgo puro (P05..P30, ES, claim curve).

Principio: NO dejar que la automatización estadística borre el riesgo.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from src.normalization import (YieldNormalizationConfig,
                               YieldNormalizationEngine)
from src.normalization.validation import half_life_weights

from .backtest import _indemnity, rolling_insurance_backtest
from .candidates import EnsembleCandidate, build_candidate
from .config import RiskDistributionConfig
from .schemas import DistCandidateResult, RiskModelResult, dist_governance
from .volatility import (apply_target_volatility, devolatilize,
                         downside_upside_sd, select_volatility_model)
from .zero_mass import classify_zeros, estimate_p_zero

log = logging.getLogger("agro_rate.distribution")
_EPS = 1e-9
_PARAM_FAMILY = {"normal", "lognormal", "gamma", "weibull", "student_t"}


class RiskDistributionEngine:
    def __init__(self, config: RiskDistributionConfig | None = None,
                 norm_config: YieldNormalizationConfig | None = None):
        self.config = config or RiskDistributionConfig()
        self.norm_config = norm_config or YieldNormalizationConfig(
            execution_mode=self.config.execution_mode)

    # ------------------------------------------------------------------
    def run(self, raw_serie: pd.DataFrame, *, crop: str = "",
            location: str = "", aggregation_level: str = "DEPARTMENT",
            target_year: int | None = None,
            normalization_result=None) -> RiskModelResult:
        cfg = self.config
        t0 = time.perf_counter()
        rng = np.random.default_rng(cfg.random_seed)
        out = RiskModelResult(status="OK", selection_mode=cfg.selection_mode,
                              execution_mode=cfg.execution_mode)

        # ---------- etapa 1-2: trend + memoria (motor existente) --------
        norm = normalization_result or YieldNormalizationEngine(
            self.norm_config).run(raw_serie, crop=crop, location=location,
                                  target_year=target_year)
        if norm.status != "OK":
            out.status = "INSUFFICIENT_DATA"
            out.warnings = list(norm.warnings)
            return out
        nh = norm.normalized_history.copy()
        mode = norm.recommended.detrending_mode
        mu_target = float(norm.expected_yield)
        half_life = norm.recommended.half_life_years
        out.trend = {
            "model": norm.recommended.trend_model,
            "detrending_mode": mode,
            "start_year": norm.recommended.start_year,
            "half_life_years": half_life,
            "target_yield": mu_target,
            "confidence": norm.selection_confidence,
        }
        out.sample = {"raw_years": int(norm.raw_years),
                      "effective_sample_size": float(norm.effective_sample_size)}

        # ---------- ceros (§33-39) --------------------------------------
        zero_rep = classify_zeros(raw_serie)
        zero = estimate_p_zero(zero_rep, n_years=len(nh),
                               aggregation_level=aggregation_level, config=cfg)
        zero["classification"] = zero_rep.to_dict("records")
        out.zero_mass = zero
        zero_years = set(zero_rep["year"]) if len(zero_rep) else set()
        nh["zero_event"] = nh["year"].isin(zero_years)
        pos = nh[~nh["zero_event"]].copy()
        years = pos["year"].to_numpy()
        shocks = pos["relative_shock"].to_numpy(float)
        if len(shocks) < cfg.min_training_shocks + 4:
            out.status = "INSUFFICIENT_DATA"
            out.warnings.append("shocks positivos insuficientes para la "
                                "etapa de distribución")
            return out

        # ---------- volatilidad (§16-20) --------------------------------
        vol = select_volatility_model(years, shocks, mode, cfg)
        out.volatility = {k: v for k, v in vol.items() if k != "sigma_t"}
        std = devolatilize(shocks, vol, mode)
        obs_down_sd, obs_up_sd = downside_upside_sd(
            std, 1.0 if mode == "multiplicative" else 0.0)

        def weights_fn(train_years, ref):
            return half_life_weights(train_years, ref_year=ref,
                                     half_life=half_life)

        w_full = weights_fn(years, float(years.max()))
        level_mult = cfg.vol_multiplier_by_level.get(
            aggregation_level.upper(), 1.0)

        # ---------- candidatos (§25-31) ---------------------------------
        cands: list[DistCandidateResult] = []
        for name in cfg.candidate_distributions(mode):
            cands.append(self._evaluate(name, years, std, weights_fn, w_full,
                                        mode, vol, obs_down_sd, cfg, rng))
        out.candidates = cands
        ok = [c for c in cands if c.status == "OK"]
        param_ok = [c for c in cands if c.name in _PARAM_FAMILY
                    and np.isfinite(c.aicc)]
        aic_champion = (min(param_ok, key=lambda c: c.aicc).name
                        if param_ok else None)

        if not ok:                                   # fallback §65-style
            fallback = min(cands, key=lambda c: (c.risk_score if
                           np.isfinite(c.risk_score) else np.inf))
            fallback.status = "OK"
            fallback.reason += " · FALLBACK: todos violaron gates; se toma el menos malo con warning"
            ok = [fallback]
            out.warnings.append("todos los candidatos violaron algún gate de "
                                "riesgo; revisar serie (fallback aplicado)")

        # ---------- selección: gates primero, score después (§50) --------
        ok.sort(key=lambda c: c.risk_score)
        best = ok[0]
        within = [c for c in ok if c.risk_score <= best.risk_score + best.score_se]
        selected = min(within, key=lambda c: (c.complexity, c.risk_score))

        # ---------- estabilidad + ensemble (§48-49) ----------------------
        stability, runner = self._stability(ok, selected, rng)
        use_ensemble = (len(within) >= 2 and stability < cfg.ensemble_stability_threshold * 100
                        and runner is not None and runner.name != selected.name)
        final_members = [selected, runner] if use_ensemble else [selected]

        # ---------- fit final + Monte Carlo (§56) ------------------------
        fitted = []
        for c in final_members:
            f = build_candidate(c.name, mode, cfg, weights=w_full, x=std)
            if c.name not in ("kde", "weighted_kde"):
                f.fit(std, w_full)
            fitted.append(f)
        if use_ensemble:
            inv = [1.0 / max(c.risk_score, _EPS) for c in final_members]
            model = EnsembleCandidate(fitted, inv)
            model_name = "MODEL_ENSEMBLE"
            model_label = model.label
        else:
            model = fitted[0]
            model_name = selected.name
            model_label = selected.name

        sims_shock = model.sample(cfg.n_sims, rng)
        sims_shock = apply_target_volatility(
            sims_shock, vol, mode, level_multiplier=level_mult,
            floor=0.0 if mode == "multiplicative" else None)
        u = rng.uniform(size=cfg.n_sims)
        zero_mask = u < zero["p_zero"]
        if mode == "multiplicative":
            y_sim = np.where(zero_mask, 0.0, mu_target * sims_shock)
        else:
            y_sim = np.where(zero_mask, 0.0,
                             np.maximum(mu_target + sims_shock, 0.0))
        assert (y_sim >= 0).all(), "soporte negativo en simulación (§57)"
        out.simulated_yields = y_sim

        # ---------- métricas de riesgo (§2, §41) -------------------------
        qs = np.quantile(y_sim, [0.05, 0.10, 0.20, 0.30])
        es = {}
        for a in cfg.es_alphas:
            qa = np.quantile(y_sim, a)
            es[f"es_{int(a*100):02d}"] = float(y_sim[y_sim <= qa].mean()
                                               if (y_sim <= qa).any() else qa)
        out.tail = {"p05": float(qs[0]), "p10": float(qs[1]),
                    "p20": float(qs[2]), "p30": float(qs[3]), **es,
                    "prob_catastrophic": float((y_sim < 0.25 * mu_target).mean())}

        rows = []
        hist_shock_all = np.where(nh["zero_event"], 0.0,
                                  nh["relative_shock"]).astype(float)
        for c in cfg.coverage_levels:
            g = c * mu_target
            li = np.maximum(g - y_sim, 0.0) / g
            hist_li = _indemnity(hist_shock_all, c)
            rows.append({
                "coverage": c, "guarantee_kg": g,
                "claim_prob": float((y_sim < g).mean()),
                "hist_claim_freq": float((hist_shock_all < c).mean()),
                "expected_indemnity": float(li.mean()),
                "hist_loss_cost": float(hist_li.mean()),
                "severity": float(li[li > 0].mean()) if (li > 0).any() else 0.0,
            })
        out.claim_curve = pd.DataFrame(rows)
        out.insurance_validation = {
            "per_coverage": selected.coverage_table,
            "val_years": (selected.coverage_table and None) or None,
        }
        out.insurance_validation["val_years"] = getattr(selected, "val_years", None)

        # ---------- salida (§54-55) --------------------------------------
        nh["standardized_shock"] = np.where(
            nh["zero_event"], 0.0,
            pd.Series(std, index=pos.index).reindex(nh.index))
        nh["zero_category"] = nh["year"].map(
            {r["year"]: r["category"] for r in zero["classification"]}
        ).fillna("")
        out.normalized_history = nh
        out.distribution = {
            "recommended": model_name, "label": model_label,
            "numerical_aic_champion": aic_champion,
            "bandwidth": getattr(model, "bandwidth",
                                 getattr(fitted[0], "bandwidth", None)),
            "level_vol_multiplier": level_mult,
        }
        out.ranking = self._ranking(cands, selected, aic_champion,
                                    model_name if use_ensemble else None)
        reasons = self._explain(selected, best, aic_champion, use_ensemble,
                                model_label, vol, zero, cands)
        confidence = self._confidence(stability, selected, norm, cfg)
        out.selection = {"risk_score": selected.risk_score,
                         "score_se": selected.score_se,
                         "confidence": confidence,
                         "stability_pct": stability, "reason": reasons}
        out.warnings += [f"volatilidad: modelo {vol['model']}"] if vol["model"] != "constant" else []
        if aic_champion and aic_champion != selected.name:
            ch = next(c for c in cands if c.name == aic_champion)
            if ch.status != "OK":
                out.warnings.append(
                    f"el campeón AIC ({aic_champion}) fue RECHAZADO por "
                    f"subdispersión de riesgo: {ch.reason}")
        out.governance = dist_governance(raw_serie, cfg, out)
        out.timings = {"total_s": round(time.perf_counter() - t0, 2),
                       "n_candidates": len(cands)}
        log.info("distribución %s/%s: %s conf=%s %.1fs", crop, location,
                 model_label, confidence, out.timings["total_s"])
        return out

    # ------------------------------------------------------------------
    def _evaluate(self, name, years, std, weights_fn, w_full, mode, vol,
                  obs_down_sd, cfg, rng) -> DistCandidateResult:
        c = DistCandidateResult(name=name)
        bt = rolling_insurance_backtest(years, std, weights_fn, name, mode,
                                        vol, cfg, rng)
        if bt is None:
            c.status, c.reason = "FAILED", "backtest no computable"
            return c
        try:
            full = build_candidate(name, mode, cfg, weights=w_full, x=std)
            if name not in ("kde", "weighted_kde"):
                full.fit(std, w_full)
            sims = apply_target_volatility(
                full.sample(6000, rng), vol, mode,
                floor=0.0 if mode == "multiplicative" else None)
        except Exception as exc:
            c.status, c.reason = "FAILED", f"fit: {exc}"
            return c

        center = 1.0 if mode == "multiplicative" else 0.0
        sim_down, _ = downside_upside_sd(sims, center)
        c.downside_sd_ratio = (sim_down / obs_down_sd
                               if obs_down_sd and np.isfinite(obs_down_sd)
                               else float("nan"))
        c.p10_sim = float(np.quantile(sims, 0.10))
        p10_emp = float(np.quantile(std, 0.10))
        c.freq_ratio = bt["freq_ratio"]
        c.aicc = full.information_criteria(std)["aicc"]
        c.coverage_table = bt["coverage_table"]
        c.val_years = bt["val_years"]
        c.fold_scores = bt["fold_scores"]
        c.score_se = bt["score_se"]

        # penalidad de volatilidad asimétrica (§19-20)
        r = c.downside_sd_ratio
        vol_pen = 0.0
        if np.isfinite(r):
            vol_pen = (cfg.underdispersion_multiplier * max(0.0, 1.0 - r)
                       + 0.5 * max(0.0, r - 1.0))
        c.volatility_pen = float(vol_pen)
        c.stability_pen = float(c.score_se / max(np.mean(bt["fold_scores"]), _EPS))
        c.complexity = cfg.DIST_COMPLEXITY.get(name, 0.5)
        c.tail, c.indemnity, c.distribution = bt["tail"], bt["indemnity"], bt["distribution"]

        w = cfg.weights
        c.risk_score = (w.tail * c.tail + w.indemnity * c.indemnity
                        + w.distribution * c.distribution
                        + w.volatility * c.volatility_pen
                        + w.stability * c.stability_pen
                        + w.complexity * c.complexity)

        # ---- gates duros (§19, §30, §51): RIESGO PRIMERO -----------------
        fails = []
        if np.isfinite(r) and r < cfg.gate_downside_sd_ratio:
            fails.append(f"downside SD sim/obs {r:.2f} < {cfg.gate_downside_sd_ratio}")
        if np.isfinite(c.freq_ratio) and c.freq_ratio < cfg.gate_claim_freq_ratio:
            fails.append(f"frecuencia de siniestro predicha {c.freq_ratio:.0%} de la observada")
        if p10_emp > 0 and c.p10_sim > p10_emp * cfg.gate_p10_ratio:
            fails.append(f"P10 simulado {c.p10_sim:.3f} > {cfg.gate_p10_ratio}× P10 empírico")
        if fails:
            c.status = "REJECTED_RISK_UNDERDISPERSION"
            c.reason = "; ".join(fails)
        return c

    # ------------------------------------------------------------------
    def _stability(self, ok, selected, rng):
        n_folds = min(len(c.fold_scores) for c in ok)
        contenders = [c for c in ok if len(c.fold_scores) >= n_folds][:10]
        fixed = {c.name: c.risk_score - float(np.mean(c.fold_scores[:n_folds]))
                 for c in contenders}
        wins, runner_count = 0, {}
        for _ in range(self.config.boots):
            idx = rng.integers(0, n_folds, n_folds)
            scores = {c.name: fixed[c.name]
                      + float(np.mean(np.asarray(c.fold_scores[:n_folds])[idx]))
                      for c in contenders}
            winner = min(scores, key=scores.get)
            if winner == selected.name:
                wins += 1
            else:
                runner_count[winner] = runner_count.get(winner, 0) + 1
        stability = 100.0 * wins / max(self.config.boots, 1)
        runner = None
        if runner_count:
            rname = max(runner_count, key=runner_count.get)
            runner = next(c for c in contenders if c.name == rname)
        return stability, runner

    # ------------------------------------------------------------------
    @staticmethod
    def _ranking(cands, selected, aic_champion, ensemble_name):
        rows = []
        for c in sorted(cands, key=lambda c: (c.risk_score if np.isfinite(c.risk_score) else 9e9)):
            status = []
            if c.name == selected.name:
                status.append("SELECTED" if not ensemble_name else "ENSEMBLE BASE")
            if c.name == aic_champion:
                status.append("AIC champion")
            if c.status != "OK":
                status.append(c.status)
            rows.append({"distribution": c.name,
                         "risk_score": round(c.risk_score, 4),
                         "tail": round(c.tail, 4),
                         "indemnity": round(c.indemnity, 4),
                         "downside_sd_ratio": round(c.downside_sd_ratio, 2)
                         if np.isfinite(c.downside_sd_ratio) else None,
                         "aicc": round(c.aicc, 1) if np.isfinite(c.aicc) else None,
                         "complexity": c.complexity,
                         "status": " · ".join(status) if status else ""})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    @staticmethod
    def _explain(selected, best, aic_champion, use_ensemble, label, vol,
                 zero, cands):
        rs = []
        if aic_champion and aic_champion != selected.name:
            ch = next(c for c in cands if c.name == aic_champion)
            if ch.status != "OK":
                rs.append(f"{aic_champion} tuvo el mejor AICc pero fue "
                          f"RECHAZADO: {ch.reason} (§30: el AIC no tiene "
                          "autoridad para borrar riesgo).")
            else:
                rs.append(f"{aic_champion} fue el campeón AICc; "
                          f"{selected.name} lo superó en el score asegurador "
                          "(cola + indemnización + volatilidad).")
        if selected.name != best.name:
            rs.append(f"{best.name} tuvo el menor RiskScore; {selected.name} "
                      "quedó dentro de 1 SE con menor complejidad (§32).")
        rs.append(f"Seleccionado: {label} — preserva la volatilidad downside "
                  f"(ratio {selected.downside_sd_ratio:.2f}) y calibra "
                  "frecuencia/severidad de siniestros en el backtest rolling.")
        if use_ensemble:
            rs.append("Estabilidad de selección insuficiente para un único "
                      "modelo → ensemble ponderado por score (§49).")
        if vol["model"] != "constant":
            rs.append(f"Volatilidad {vol['model']}: evidencia de "
                      f"{vol['evidence']['variance_drift'].lower()} / "
                      f"{vol['evidence']['asymmetry'].lower()} "
                      f"(down/up SD = {vol['down_up_ratio']:.2f}).")
        rs.append(f"P(Y=0) = {zero['p_zero']:.2%} a nivel "
                  f"{zero['aggregation_level']} ({zero['estimation_method']}); "
                  "el Monte Carlo produce ceros exactos (§56).")
        return rs

    # ------------------------------------------------------------------
    @staticmethod
    def _confidence(stability, selected, norm, cfg):
        if (stability >= cfg.stability_high * 100
                and selected.status == "OK"
                and norm.selection_confidence == "HIGH"
                and norm.effective_sample_size >= 30):
            return "HIGH"
        if stability >= cfg.stability_medium * 100 and selected.status == "OK":
            return "MEDIUM"
        return "LOW"
