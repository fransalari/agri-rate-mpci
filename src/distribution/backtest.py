"""Backtest asegurador rolling (spec §21-22, §41-45).

Para cada año t de la ventana común de validación: ajustar el candidato
SOLO con shocks < t (sin leakage), simular, y comparar contra el shock
realizado en t:

  - cola: pinball P05/P10/P20/P30 + error de Expected Shortfall;
  - distribución: CRPS;
  - indemnización por nivel de cobertura c: frecuencia, severidad y
    prima pura predichas vs realizadas, con penalidad asimétrica por
    SUBESTIMACIÓN (§45).

Todo en términos relativos (shocks), comparable entre regiones.
"""

from __future__ import annotations

import numpy as np

from .candidates import build_candidate
from .volatility import apply_target_volatility

_EPS = 1e-9


def _indemnity(shock: np.ndarray | float, c: float) -> np.ndarray | float:
    """Pérdida relativa a la garantía c (en unidades de rinde esperado):
    LC = max(c − R, 0)/c ∈ [0,1]."""
    return np.maximum(c - shock, 0.0) / c


def rolling_insurance_backtest(years: np.ndarray, std_shocks: np.ndarray,
                               weights_fn, name: str, mode: str, vol: dict,
                               config, rng: np.random.Generator,
                               n_sims: int = 4000) -> dict | None:
    """Backtest de un candidato. weights_fn(train_years, ref) → pesos."""
    years = np.asarray(years)
    x = np.asarray(std_shocks, float)
    order = np.argsort(years)
    years, x = years[order], x[order]

    eligible = years[config.min_training_shocks:]
    val_years = eligible[-config.max_validation_years:]
    if len(val_years) < 4:
        return None

    pin_all, crps_all, es_err = [], [], []
    cov = {c: {"pred_freq": [], "obs_claim": [], "pred_sev": [], "obs_sev": [],
               "pred_prem": [], "obs_loss": []} for c in config.coverage_levels}

    for t in val_years:
        m = years < t
        xt, yt = x[m], float(x[years == t][0])
        w = weights_fn(years[m], float(years[m].max()))
        try:
            cand = build_candidate(name, mode, config, weights=w, x=xt)
            if name not in ("kde", "weighted_kde"):
                cand.fit(xt, w)
            sims = cand.sample(n_sims, rng)
        except Exception:
            return None
        sims = apply_target_volatility(sims, vol, mode)
        if mode == "multiplicative":
            sims = np.maximum(sims, 0.0)

        # ---- cola (§21) ------------------------------------------------
        qs = np.quantile(sims, config.tail_quantiles)
        pin = [max(q * (yt - qp), (q - 1) * (yt - qp))
               for qp, q in zip(qs, config.tail_quantiles)]
        pin_all.append(float(np.mean(pin)))
        crps_all.append(float(np.mean(np.abs(sims - yt))
                              - 0.5 * np.mean(np.abs(np.diff(
                                  rng.permutation(sims)[:2000])))))
        for a in config.es_alphas:
            qa = np.quantile(sims, a)
            es_sim = float(sims[sims <= qa].mean()) if (sims <= qa).any() else qa
            # realizado: solo aporta cuando el año cae en la cola empírica
            if yt <= np.quantile(xt, a + 0.10):
                es_err.append(max(0.0, yt - es_sim) * config.underdispersion_multiplier
                              + max(0.0, es_sim - yt))

        # ---- indemnización por cobertura (§22, §42-44) -------------------
        for c in config.coverage_levels:
            li = _indemnity(sims, c)
            cov[c]["pred_freq"].append(float((sims < c).mean()))
            cov[c]["obs_claim"].append(1.0 if yt < c else 0.0)
            sev = li[li > 0]
            cov[c]["pred_sev"].append(float(sev.mean()) if len(sev) else 0.0)
            cov[c]["obs_sev"].append(float(_indemnity(yt, c)) if yt < c else np.nan)
            cov[c]["pred_prem"].append(float(li.mean()))
            cov[c]["obs_loss"].append(float(_indemnity(yt, c)))

    # ---------- agregación ----------------------------------------------
    tail = float(np.mean(pin_all))
    dist_score = float(np.mean(crps_all))
    es_score = float(np.mean(es_err)) if es_err else 0.0

    freq_rows, prem_bias = [], []
    for c in config.coverage_levels:
        d = cov[c]
        pf, of = float(np.mean(d["pred_freq"])), float(np.mean(d["obs_claim"]))
        obs_sev = np.nanmean(d["obs_sev"]) if np.isfinite(d["obs_sev"]).any() else np.nan
        pp, ol = float(np.mean(d["pred_prem"])), float(np.mean(d["obs_loss"]))
        under = max(0.0, ol - pp) * config.underpricing_multiplier
        over = max(0.0, pp - ol)
        prem_bias.append(under + over)
        freq_rows.append({"coverage": c, "pred_freq": pf, "obs_freq": of,
                          "pred_sev": float(np.mean(d["pred_sev"])),
                          "obs_sev": float(obs_sev) if np.isfinite(obs_sev) else np.nan,
                          "pred_premium": pp, "obs_loss_cost": ol})
    indemnity = float(np.mean(prem_bias)) + 0.5 * es_score

    # fold-level para SE (§48): pinball + prima del nivel central (70%)
    mid = config.coverage_levels[len(config.coverage_levels) // 2]
    fold_scores = [p + abs(pl - ol) for p, pl, ol in
                   zip(pin_all, cov[mid]["pred_prem"], cov[mid]["obs_loss"])]
    se = float(np.std(fold_scores, ddof=1) / np.sqrt(len(fold_scores)))

    # calibración de frecuencia observable (para gates): promedio en
    # coberturas con al menos 2 siniestros observados
    obs_pairs = [(r["pred_freq"], r["obs_freq"]) for r in freq_rows
                 if r["obs_freq"] * len(val_years) >= 2]
    freq_ratio = (float(np.mean([p / max(o, _EPS) for p, o in obs_pairs]))
                  if obs_pairs else float("nan"))

    return {"tail": tail, "distribution": dist_score, "indemnity": indemnity,
            "coverage_table": freq_rows, "fold_scores": fold_scores,
            "score_se": se, "freq_ratio": freq_ratio,
            "n_folds": len(val_years), "val_years": val_years.tolist()}
