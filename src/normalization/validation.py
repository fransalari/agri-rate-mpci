"""Validación rolling-origin expanding-window y componentes del score.

Anti-leakage (spec §7): para predecir el año t, TODO (fit, pesos,
shocks, cuantiles) usa exclusivamente años < t. El año de referencia
de los pesos es el último año de entrenamiento, nunca t.
"""

from __future__ import annotations

import numpy as np

from .trend_models import build_model

_EPS = 1e-9


def half_life_weights(train_years: np.ndarray, ref_year: float,
                      half_life: float) -> np.ndarray:
    """w_t = 2^(-(T - t)/h) (spec §19). h=inf → pesos iguales."""
    t = np.asarray(train_years, dtype=float)
    if not np.isfinite(half_life):
        return np.ones_like(t)
    return np.power(2.0, -(ref_year - t) / half_life)


def effective_sample_size(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / max((w ** 2).sum(), _EPS))


def weighted_quantile(x: np.ndarray, q, w: np.ndarray) -> np.ndarray:
    """Cuantil ponderado (interpolación sobre CDF de pesos)."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    order = np.argsort(x)
    x, w = x[order], w[order]
    cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(np.atleast_1d(q), cdf, x)


def pinball_loss(y_true: float, q_pred: float, q: float) -> float:
    diff = y_true - q_pred
    return float(max(q * diff, (q - 1) * diff))


def weighted_crps(sample: np.ndarray, w: np.ndarray, y: float) -> float:
    """CRPS empírico ponderado: E|X−y| − ½E|X−X'| (spec §29)."""
    x = np.asarray(sample, dtype=float)
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    term1 = float(np.sum(w * np.abs(x - y)))
    diff = np.abs(x[:, None] - x[None, :])
    term2 = 0.5 * float(w @ diff @ w)
    return term1 - term2


# --------------------------------------------------------------- folds
def validation_years(all_years: np.ndarray, min_training: int,
                     max_validation: int) -> np.ndarray:
    """Ventana común de validación (spec §18): los últimos K años con
    dato, dejando al menos `min_training` años previos en la serie
    completa. Todos los candidatos se evalúan sobre ESTA misma ventana."""
    ys = np.sort(np.asarray(all_years))
    eligible = ys[min_training:]
    return eligible[-max_validation:] if len(eligible) else np.array([], dtype=int)


def run_cv(years: np.ndarray, yields: np.ndarray, *,
           model_name: str, mode: str, start_year: int, half_life: float,
           val_years: np.ndarray, config) -> dict | None:
    """CV rolling-origin para una configuración. None si es inelegible."""
    years = np.asarray(years)
    y = np.asarray(yields, dtype=float)
    folds = []

    for t in val_years:
        mask = (years >= start_year) & (years < t)
        ty, vy = years[mask], y[mask]
        if len(ty) < config.minimum_training_years:
            return None  # inelegible: comparación justa exige entrenar en todos los folds
        w = half_life_weights(ty, ref_year=float(ty.max()), half_life=half_life)
        try:
            model = build_model(model_name, config).fit(ty, vy, w)
        except Exception:
            raise RuntimeError(f"fit falló en fold {t}")
        trend_t = float(model.predict(np.array([t]))[0])
        trend_train = model.predict(ty)

        if mode == "multiplicative":
            shocks = vy / np.maximum(trend_train, _EPS)
            pred_sample = trend_t * shocks
        else:
            shocks = vy - trend_train
            pred_sample = trend_t + shocks
        pred_sample = np.maximum(pred_sample, 0.0)

        sel = y[years == t]
        if len(sel) == 0:      # campaña ausente (series con huecos / HIV)
            continue
        y_t = float(sel[0])
        point = float(np.sum(w * pred_sample) / w.sum())
        q_preds = weighted_quantile(pred_sample, config.tail_quantiles, w)
        pin = np.mean([pinball_loss(y_t, qp, q)
                       for qp, q in zip(q_preds, config.tail_quantiles)])
        folds.append({
            "year": int(t),
            "abs_err": abs(y_t - point),
            "sq_err": (y_t - point) ** 2,
            "pinball": pin,
            "crps": weighted_crps(pred_sample, w, y_t),
            "y_true": y_t,
        })

    y_val = np.array([f["y_true"] for f in folds])
    scale = max(float(np.mean(y_val)), _EPS)
    n = len(folds)
    pred = float(np.sqrt(np.mean([f["sq_err"] for f in folds]))) / scale  # nRMSE
    tail = float(np.mean([f["pinball"] for f in folds])) / scale
    dist = float(np.mean([f["crps"] for f in folds])) / scale

    # score por fold (componentes que varían entre folds) para el SE (spec §33)
    w_ = config.score_weights
    fold_scores = [
        (w_.prediction * (np.sqrt(f["sq_err"]) / scale)
         + w_.tail * (f["pinball"] / scale)
         + w_.distribution * (f["crps"] / scale))
        for f in folds
    ]
    se = float(np.std(fold_scores, ddof=1) / np.sqrt(n)) if n > 1 else 0.0

    return {"prediction_error": pred, "tail_error": tail,
            "distribution_error": dist, "fold_scores": fold_scores,
            "score_se": se, "folds": folds}


# ------------------------------------------------- full-sample pieces
def fit_full(years, yields, *, model_name, mode, start_year, half_life,
             config, ref_year=None):
    """Fit final sobre toda la historia elegida (para normalizar/predecir)."""
    years = np.asarray(years)
    y = np.asarray(yields, dtype=float)
    mask = years >= start_year
    ty, vy = years[mask], y[mask]
    ref = float(ref_year if ref_year is not None else ty.max())
    w = half_life_weights(ty, ref_year=ref, half_life=half_life)
    model = build_model(model_name, config).fit(ty, vy, w)
    trend = model.predict(ty)
    if mode == "multiplicative":
        shocks = vy / np.maximum(trend, _EPS)
    else:
        shocks = vy - trend
    return {"model": model, "years": ty, "yields": vy, "weights": w,
            "trend": trend, "shocks": shocks}


def instability_penalty(years, yields, *, model_name, mode, start_year,
                        half_life, target_year, config, k: int = 3) -> float:
    """Sensibilidad (spec §31): cambio relativo del trend target al
    remover los k años de shock más extremo."""
    base = fit_full(years, yields, model_name=model_name, mode=mode,
                    start_year=start_year, half_life=half_life, config=config)
    t_target = float(base["model"].predict(np.array([target_year]))[0])
    order = np.argsort(-np.abs(base["shocks"] - np.median(base["shocks"])))
    changes = []
    for idx in order[:k]:
        yr = base["years"][idx]
        keep = base["years"] != yr
        if keep.sum() < config.minimum_training_years:
            continue
        try:
            alt = fit_full(base["years"][keep], base["yields"][keep],
                           model_name=model_name, mode=mode,
                           start_year=start_year, half_life=half_life,
                           config=config, ref_year=base["years"].max())
            t_alt = float(alt["model"].predict(np.array([target_year]))[0])
            changes.append(abs(t_alt - t_target) / max(t_target, _EPS))
        except Exception:
            changes.append(0.5)  # inestable si ni siquiera ajusta
    return float(np.mean(changes)) if changes else 0.0
