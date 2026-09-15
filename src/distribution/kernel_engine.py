"""KERNEL YIELD RISK ENGINE (spec kernel §2-§18, §34-35, §63).

Un único KernelRiskModel calibrado (LOG_KERNEL ponderado sobre shocks
normalizados positivos) consumido por TRES motores numéricos:

  - DETERMINISTIC : anclas históricas × nodos de probabilidad → escenarios
                    explícitos, trazables, sin ruido Monte Carlo;
  - MONTE_CARLO   : muestreo aleatorio de la MISMA distribución;
  - INTEGRATION   : fórmula cerrada — cada componente del log-kernel es
                    lognormal, luego E[max(G−R,0)] es una mezcla de puts
                    lognormales (partial expectation analítica).

Los tres deben converger (test §51). Kernel = distribución; el motor
numérico solo la consume (§31). Log-kernel garantiza R > 0 (§7); la
masa en cero se modela como mezcla hurdle explícita (§8).

Principio: DO NOT LET SMOOTHING REMOVE CREDIBLE HISTORICAL DOWNSIDE RISK.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

_EPS = 1e-12
KERNEL_VERSION = "0.1.0"


def _silverman_log(z: np.ndarray, w: np.ndarray) -> float:
    n_eff = w.sum() ** 2 / max((w ** 2).sum(), _EPS)
    mu = np.average(z, weights=w)
    sd = np.sqrt(np.average((z - mu) ** 2, weights=w))
    iqr = np.subtract(*np.percentile(z, [75, 25]))
    a = min(sd, iqr / 1.349) if iqr > 0 else sd
    return 1.06 * max(a, _EPS) * n_eff ** (-0.2)


@dataclass
class KernelRiskModel:
    """LOG_KERNEL ponderado con masa en cero. Calibración única."""
    anchors_shock: np.ndarray = None      # shocks relativos R_i > 0
    anchor_years: np.ndarray = None
    weights: np.ndarray = None            # normalizados a 1
    bandwidth: float = 0.0                # en escala log
    bandwidth_method: str = "SILVERMAN"
    bandwidth_factor: float = 1.0
    p_zero: float = 0.0
    flags: list = field(default_factory=list)

    # ------------------------------------------------------------------
    @classmethod
    def fit(cls, shocks, years=None, weights=None, weighted=True,
            p_zero: float = 0.0, bandwidth_mode: str = "AUTO",
            factors=(0.5, 0.75, 1.0, 1.25, 1.5)) -> "KernelRiskModel":
        r = np.asarray(shocks, float)
        m = r > 0
        r = r[m]
        yrs = (np.asarray(years)[m] if years is not None
               else np.arange(len(r)))
        w = (np.asarray(weights, float)[m] if (weighted and weights is not None)
             else np.ones(len(r)))
        w = w / w.sum()
        z = np.log(r)
        h0 = _silverman_log(z, w)

        flags = []
        if bandwidth_mode == "SILVERMAN":
            h, method, factor = h0, "SILVERMAN", 1.0
        else:
            # AUTO (§11, §42-43): grilla × Silverman por LOO-likelihood
            # ponderada + preservación de cola (§13): penaliza fuerte si
            # el suavizado sube P10 o comprime la SD downside.
            best, best_score = None, -np.inf
            emp_p10 = float(np.quantile(r, 0.10))
            emp_dsd = float(np.sqrt(np.average(
                np.minimum(r - 1.0, 0.0) ** 2, weights=w)))
            for f in factors:
                h_c = h0 * f
                loo = cls._loo_loglik(z, w, h_c)
                q = np.exp(z[:, None] + h_c * norm.ppf(
                    np.linspace(0.025, 0.975, 20))[None, :]).ravel()
                qw = np.repeat(w / 20, 20)
                p10 = cls._wq(q, qw, 0.10)
                dsd = float(np.sqrt(np.average(
                    np.minimum(q - 1.0, 0.0) ** 2, weights=qw)))
                pen = (4.0 * max(0.0, (p10 - emp_p10) / max(emp_p10, _EPS) - 0.10)
                       + 4.0 * max(0.0, 1.0 - dsd / max(emp_dsd, _EPS) - 0.10))
                score = loo / len(z) - pen
                if score > best_score:
                    best, best_score, best_f = h_c, score, f
            h, method, factor = best, "TAIL_AWARE_LOO", best_f
            if factor >= factors[-1]:
                flags.append("KERNEL_OVER_SMOOTHED_CANDIDATE")
            if factor <= factors[0]:
                flags.append("KERNEL_UNDER_SMOOTHED_CANDIDATE")

        return cls(anchors_shock=r, anchor_years=yrs, weights=w,
                   bandwidth=float(h), bandwidth_method=method,
                   bandwidth_factor=float(factor), p_zero=float(p_zero),
                   flags=flags)

    # ------------------------------------------------------------------
    @staticmethod
    def _loo_loglik(z, w, h):
        d = z[:, None] - z[None, :]
        K = np.exp(-0.5 * (d / h) ** 2) / (h * np.sqrt(2 * np.pi))
        P = np.tile(w, (len(z), 1))
        np.fill_diagonal(K, 0.0)
        np.fill_diagonal(P, 0.0)
        dens = (K * P).sum(1) / np.maximum(P.sum(1), _EPS)
        return float(np.sum(np.log(np.maximum(dens, 1e-300))))

    @staticmethod
    def _wq(x, w, q):
        o = np.argsort(x)
        cw = np.cumsum(w[o])
        return float(x[o][np.searchsorted(cw, q * cw[-1])])

    # ==================================================================
    # MOTOR 1 — DETERMINISTIC (§4, §17, §34)
    # ==================================================================
    def deterministic_scenarios(self, n_nodes: int = 20) -> pd.DataFrame:
        """Anclas × nodos: escenarios explícitos con peso correcto.
        Nodos midpoint p_j=(j−½)/M ⇒ cuadratura válida, peso 1/M c/u."""
        p = (np.arange(n_nodes) + 0.5) / n_nodes
        eps = norm.ppf(p)                                  # perturbación
        z = np.log(self.anchors_shock)
        shock = np.exp(z[:, None] + self.bandwidth * eps[None, :])
        base = np.repeat(self.anchors_shock, n_nodes)
        df = pd.DataFrame({
            "scenario_id": np.arange(shock.size),
            "historical_anchor_year": np.repeat(self.anchor_years, n_nodes),
            "base_relative_shock": base,
            "kernel_quantile": np.tile(p, len(z)),
            "perturbation": shock.ravel() - base,
            "simulated_relative_shock": shock.ravel(),
            "weight": np.repeat(self.weights / n_nodes, n_nodes)
                      * (1.0 - self.p_zero),
        })
        if self.p_zero > 0:                                # §8: hurdle
            zero = pd.DataFrame({
                "scenario_id": [-1], "historical_anchor_year": [-1],
                "base_relative_shock": [0.0], "kernel_quantile": [np.nan],
                "perturbation": [0.0], "simulated_relative_shock": [0.0],
                "weight": [self.p_zero]})
            df = pd.concat([zero, df], ignore_index=True)
        assert abs(df["weight"].sum() - 1.0) < 1e-9
        return df

    def expected_loss_deterministic(self, guarantee: float,
                                    n_nodes: int = 20) -> dict:
        sc = self.deterministic_scenarios(n_nodes)
        r, w = sc["simulated_relative_shock"].to_numpy(), sc["weight"].to_numpy()
        loss = np.maximum(guarantee - r, 0.0) / guarantee
        claim = r < guarantee
        el = float((loss * w).sum())
        freq = float(w[claim].sum())
        return {"expected_loss": el, "claim_probability": freq,
                "severity": el / max(freq, _EPS),
                "scenario_count": len(sc), "engine": "KERNEL_DETERMINISTIC"}

    # ==================================================================
    # MOTOR 2 — MONTE CARLO (§5): la MISMA distribución, muestreada
    # ==================================================================
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        idx = rng.choice(len(self.anchors_shock), size=n, replace=True,
                         p=self.weights)
        z = np.log(self.anchors_shock[idx]) + self.bandwidth * rng.normal(size=n)
        r = np.exp(z)                                      # > 0 siempre (§7)
        if self.p_zero > 0:
            r = np.where(rng.uniform(size=n) < self.p_zero, 0.0, r)
        return r

    def expected_loss_mc(self, guarantee: float, n: int,
                         rng: np.random.Generator) -> dict:
        r = self.sample(n, rng)
        loss = np.maximum(guarantee - r, 0.0) / guarantee
        return {"expected_loss": float(loss.mean()),
                "claim_probability": float((r < guarantee).mean()),
                "severity": float(loss[loss > 0].mean()) if (loss > 0).any() else 0.0,
                "engine": "KERNEL_MONTE_CARLO", "simulations": n}

    # ==================================================================
    # MOTOR 3 — INTEGRACIÓN ANALÍTICA (§18)
    # ==================================================================
    def expected_loss_integration(self, guarantee: float) -> dict:
        """Cada componente es lognormal(z_i, h²) ⇒
        E[max(G−R,0)] = Σ w_i [G·Φ(d_i) − e^{z_i+h²/2}·Φ(d_i − h)]
        con d_i = (ln G − z_i)/h  (partial expectation exacta)."""
        h = max(self.bandwidth, _EPS)
        z = np.log(self.anchors_shock)
        d = (np.log(guarantee) - z) / h
        put = guarantee * norm.cdf(d) - np.exp(z + h * h / 2) * norm.cdf(d - h)
        el_pos = float(np.sum(self.weights * np.maximum(put, 0.0))) / guarantee
        freq_pos = float(np.sum(self.weights * norm.cdf(d)))
        el = (1 - self.p_zero) * el_pos + self.p_zero * 1.0   # cero ⇒ pérdida total
        freq = (1 - self.p_zero) * freq_pos + self.p_zero
        return {"expected_loss": el, "claim_probability": freq,
                "severity": el / max(freq, _EPS),
                "engine": "KERNEL_INTEGRATION"}

    # ------------------------------------------------------------------
    def governance(self) -> dict:
        return {"kernel_version": KERNEL_VERSION,
                "type": "WEIGHTED_LOG_KERNEL",
                "bandwidth": self.bandwidth,
                "bandwidth_method": self.bandwidth_method,
                "bandwidth_factor_silverman": self.bandwidth_factor,
                "historical_anchors": int(len(self.anchors_shock)),
                "effective_sample": float(self.weights.sum() ** 2
                                          / (self.weights ** 2).sum()),
                "p_zero": self.p_zero, "flags": self.flags}


# ======================================================================
# Adaptador a la interfaz de candidatos del Distribution Engine (§26-28)
# ======================================================================
from .candidates import BaseCandidate  # noqa: E402


class LogKernelCandidate(BaseCandidate):
    """log_kernel / weighted_log_kernel como candidatos del motor: compiten
    bajo los mismos gates y score asegurador que paramétricas y KDE."""
    n_params = 1

    def __init__(self, name="log_kernel", weighted=False):
        super().__init__(positive_support=True)
        self.name = name
        self.weighted = weighted

    def fit(self, x, weights=None):
        self.model = KernelRiskModel.fit(
            x, weights=weights, weighted=self.weighted,
            p_zero=0.0, bandwidth_mode="AUTO")
        self.bandwidth = self.model.bandwidth
        return self

    def sample(self, n, rng):
        return self.model.sample(n, rng)
