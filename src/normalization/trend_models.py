"""Modelos candidatos de tendencia tecnológica (spec §8–13).

Interfaz común YieldTrendModel:
    fit(years, yields, weights) → self
    predict(years) → trend
    detrend(...) / normalize(...) viven en el engine (dependen del modo).

Todos los ajustes son ponderados (half-life) escalando por sqrt(w):
para OLS equivale a WLS; para RLM (Huber) es M-estimación ponderada.
La tendencia se fuerza positiva (guardrail agronómico, spec §56).
"""

from __future__ import annotations

import warnings

import numpy as np
import statsmodels.api as sm

_EPS = 1e-9


class YieldTrendModel:
    name = "base"
    df = 2

    def __init__(self, **kwargs):
        self.params_ = None
        self.meta_ = {}

    # -- API ------------------------------------------------------------
    def fit(self, years: np.ndarray, yields: np.ndarray,
            weights: np.ndarray | None = None) -> "YieldTrendModel":
        raise NotImplementedError

    def predict(self, years: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def information_criteria(self, years, yields, weights=None) -> dict:
        """AICc/BIC gaussianos sobre residuos ponderados (secundarios, §15)."""
        w = _norm_w(weights, len(yields))
        resid = yields - self.predict(years)
        n = len(yields)
        sigma2 = max(np.average(resid ** 2, weights=w), _EPS)
        ll = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
        k = self.df + 1  # + varianza
        aic = 2 * k - 2 * ll
        aicc = aic + (2 * k * (k + 1)) / max(n - k - 1, 1)
        bic = k * np.log(n) - 2 * ll
        return {"aic": aic, "aicc": aicc, "bic": bic}

    def metadata(self) -> dict:
        return {"model": self.name, "df": self.df, **self.meta_}


# ------------------------------------------------------------------ helpers
def _norm_w(weights, n):
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    return w / w.sum() * n


def _design(years: np.ndarray, deg: int) -> np.ndarray:
    t = np.asarray(years, dtype=float)
    t0 = t - t.mean()
    cols = [np.ones_like(t0)] + [t0 ** d for d in range(1, deg + 1)]
    return np.column_stack(cols)


class _LinearBase(YieldTrendModel):
    deg = 1

    def fit(self, years, yields, weights=None):
        w = _norm_w(weights, len(yields))
        self._t_mean = float(np.mean(years))
        X = _design(years, self.deg)
        sw = np.sqrt(w)
        self.params_ = np.linalg.lstsq(X * sw[:, None], yields * sw,
                                       rcond=None)[0]
        return self

    def predict(self, years):
        t0 = np.asarray(years, dtype=float) - self._t_mean
        X = np.column_stack([np.ones_like(t0)] +
                            [t0 ** d for d in range(1, self.deg + 1)])
        return np.maximum(X @ self.params_, _EPS)


class ConstantModel(_LinearBase):
    name, df, deg = "constant", 1, 0


class LinearModel(_LinearBase):
    name, df, deg = "linear", 2, 1


class QuadraticModel(_LinearBase):
    name, df, deg = "quadratic", 3, 2


class RobustLinearModel(YieldTrendModel):
    """Huber M-estimator: años extremos no arrastran la tendencia (spec §9)."""
    name, df = "robust_linear", 2

    def fit(self, years, yields, weights=None):
        w = _norm_w(weights, len(yields))
        self._t_mean = float(np.mean(years))
        t0 = np.asarray(years, dtype=float) - self._t_mean
        X = sm.add_constant(t0)
        sw = np.sqrt(w)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.RLM(yields * sw, X * sw[:, None],
                         M=sm.robust.norms.HuberT()).fit()
        self.params_ = res.params
        return self

    def predict(self, years):
        t0 = np.asarray(years, dtype=float) - self._t_mean
        return np.maximum(self.params_[0] + self.params_[1] * t0, _EPS)


class LogLinearModel(YieldTrendModel):
    """log(Y) = a + b·t → progreso tecnológico porcentual (spec §10)."""
    name, df = "log_linear", 2

    def fit(self, years, yields, weights=None):
        y = np.asarray(yields, dtype=float)
        mask = y > 0
        if mask.sum() < 3:
            raise ValueError("log_linear requiere yields > 0")
        w = _norm_w(weights, len(y))[mask]
        self._t_mean = float(np.mean(np.asarray(years)[mask]))
        t0 = np.asarray(years, dtype=float)[mask] - self._t_mean
        X = np.column_stack([np.ones_like(t0), t0])
        sw = np.sqrt(w)
        self.params_ = np.linalg.lstsq(X * sw[:, None], np.log(y[mask]) * sw,
                                       rcond=None)[0]
        # corrección de sesgo lognormal (media, no mediana)
        resid = np.log(y[mask]) - X @ self.params_
        self._smear = float(np.exp(0.5 * np.average(resid ** 2, weights=w)))
        return self

    def predict(self, years):
        t0 = np.asarray(years, dtype=float) - self._t_mean
        return np.exp(self.params_[0] + self.params_[1] * t0) * self._smear

    @property
    def annual_growth(self) -> float:
        return float(np.exp(self.params_[1]) - 1)


class PiecewiseLinearModel(YieldTrendModel):
    """Lineal con un único breakpoint continuo (hinge), penalizado (spec §12)."""
    name, df = "piecewise", 4

    def __init__(self, min_segment: int = 8, **kwargs):
        super().__init__()
        self.min_segment = min_segment

    def fit(self, years, yields, weights=None):
        t = np.asarray(years, dtype=float)
        y = np.asarray(yields, dtype=float)
        w = _norm_w(weights, len(y))
        sw = np.sqrt(w)
        order = np.argsort(t)
        t_sorted = t[order]
        best = (np.inf, None, None)
        # candidatos de breakpoint respetando segmentos mínimos
        for k in t_sorted[self.min_segment - 1:len(t_sorted) - self.min_segment + 1]:
            X = np.column_stack([np.ones_like(t), t - t.mean(),
                                 np.maximum(0.0, t - k)])
            beta, res, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
            sse = float(((X @ beta - y) ** 2 * w).sum())
            if sse < best[0]:
                best = (sse, beta, k)
        if best[1] is None:  # serie corta: degradar a lineal
            lm = LinearModel().fit(years, yields, weights)
            self._fallback = lm
            self.meta_["fallback"] = "linear"
            return self
        self._fallback = None
        self._t_mean = float(t.mean())
        self.params_, self.breakpoint_ = best[1], float(best[2])
        self.meta_["breakpoint"] = self.breakpoint_
        return self

    def predict(self, years):
        if self._fallback is not None:
            return self._fallback.predict(years)
        t = np.asarray(years, dtype=float)
        X = np.column_stack([np.ones_like(t), t - self._t_mean,
                             np.maximum(0.0, t - self.breakpoint_)])
        return np.maximum(X @ self.params_, _EPS)


class SplineModel(YieldTrendModel):
    """Spline cúbico natural restringido (df bajo, ridge suave; spec §13).

    Base natural: lineal fuera del rango observado → extrapolación segura.
    """
    name, df = "spline", 5

    def __init__(self, spline_df: int = 4, ridge: float = 1e-4, **kwargs):
        super().__init__()
        self.spline_df = spline_df
        self.ridge = ridge

    def _basis(self, t: np.ndarray) -> np.ndarray:
        # base cúbica natural con nudos en cuantiles del training
        k = self._knots
        def d(z, knot):
            num = (np.maximum(0, z - knot) ** 3
                   - np.maximum(0, z - k[-1]) ** 3)
            return num / (k[-1] - knot)
        cols = [np.ones_like(t), t]
        for kn in k[:-2]:
            cols.append(d(t, kn) - d(t, k[-2]))
        return np.column_stack(cols)

    def fit(self, years, yields, weights=None):
        t = np.asarray(years, dtype=float)
        y = np.asarray(yields, dtype=float)
        w = _norm_w(weights, len(y))
        qs = np.linspace(0, 1, self.spline_df)
        self._knots = np.unique(np.quantile(t, qs))
        if len(self._knots) < 3:
            self._fallback = LinearModel().fit(years, yields, weights)
            return self
        self._fallback = None
        X = self._basis(t)
        sw = np.sqrt(w)
        Xw, yw = X * sw[:, None], y * sw
        # ridge: estabilidad + límite a la flexibilidad
        A = Xw.T @ Xw + self.ridge * np.eye(X.shape[1])
        self.params_ = np.linalg.solve(A, Xw.T @ yw)
        return self

    def predict(self, years):
        if getattr(self, "_fallback", None) is not None:
            return self._fallback.predict(years)
        t = np.asarray(years, dtype=float)
        return np.maximum(self._basis(t) @ self.params_, _EPS)


MODEL_REGISTRY: dict[str, type[YieldTrendModel]] = {
    "constant": ConstantModel,
    "linear": LinearModel,
    "robust_linear": RobustLinearModel,
    "log_linear": LogLinearModel,
    "quadratic": QuadraticModel,
    "piecewise": PiecewiseLinearModel,
    "spline": SplineModel,
}


def build_model(name: str, config) -> YieldTrendModel:
    cls = MODEL_REGISTRY[name]
    return cls(min_segment=config.piecewise_min_segment,
               spline_df=config.spline_df)
