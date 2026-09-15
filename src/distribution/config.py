"""Configuración central del Yield Risk Distribution Engine (spec §23, §37, §71-72).

Principio rector: DO NOT LET STATISTICAL AUTOMATION REMOVE THE RISK
WE ARE TRYING TO INSURE.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RiskScoreWeights:
    """Pesos del RiskScore (spec §23). Lower = better."""
    tail: float = 0.30
    indemnity: float = 0.25
    distribution: float = 0.15
    volatility: float = 0.15
    stability: float = 0.10
    complexity: float = 0.05


@dataclass
class RiskDistributionConfig:
    execution_mode: str = "FAST"                 # FAST | FULL
    selection_mode: str = "AUTO"                 # AUTO | MANUAL | BENCHMARK

    # --- validación asegurador (spec §21-22) ---------------------------
    coverage_levels: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90)
    tail_quantiles: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)
    es_alphas: tuple[float, ...] = (0.05, 0.10)
    max_validation_years: int = 12
    min_training_shocks: int = 15

    # --- score / penalidades -------------------------------------------
    weights: RiskScoreWeights = field(default_factory=RiskScoreWeights)
    underpricing_multiplier: float = 2.0         # §45: subestimar cuesta doble
    underdispersion_multiplier: float = 2.0      # §20: comprimir cola cuesta doble

    # --- gates duros (§19, §51) ----------------------------------------
    gate_downside_sd_ratio: float = 0.85         # sim/obs < 0.85 → rechazo
    gate_claim_freq_ratio: float = 0.60          # pred/obs < 0.60 → rechazo
    gate_p10_ratio: float = 1.15                 # P10_sim > 1.15×P10_emp → rechazo

    # --- volatilidad (§17-18) ------------------------------------------
    vol_models: tuple[str, ...] = ("constant", "ewma", "asymmetric",
                                   "asymmetric_ewma")
    ewma_halflife: float = 12.0                  # campañas
    vol_drift_alpha: float = 0.05                # significancia para vol dinámica
    vol_asym_band: tuple[float, float] = (0.80, 1.25)  # down/up SD fuera ⇒ asimétrica
    min_side_obs: int = 6                        # obs mínimas por lado

    # --- distribución (§25-28) -----------------------------------------
    bandwidth_factors: tuple[float, ...] = (0.6, 0.8, 1.0, 1.3)  # × Silverman
    n_sims: int = 20_000
    ensemble_stability_threshold: float = 0.60   # §49

    # --- zero mass (§33-39) --------------------------------------------
    zero_prior_by_level: dict = field(default_factory=lambda: {
        "FIELD": 0.020, "FARM": 0.010, "PORTFOLIO": 0.004,
        "MUNICIPALITY": 0.002, "DEPARTMENT": 0.0008, "PROVINCE": 0.0002,
        "REGION": 0.0001})
    zero_prior_strength: float = 25.0            # pseudo-observaciones
    zero_countable: tuple[str, ...] = ("TRUE_TOTAL_LOSS",
                                       "PLANTED_NOT_HARVESTED",
                                       "UNKNOWN_ZERO")
    # multiplicador de volatilidad por nivel de agregación (§38 MVP):
    # los datos fuente son departamentales; niveles más finos son más
    # volátiles. Priors configurables, documentados como supuesto.
    vol_multiplier_by_level: dict = field(default_factory=lambda: {
        "FIELD": 1.60, "FARM": 1.35, "PORTFOLIO": 1.15,
        "MUNICIPALITY": 1.08, "DEPARTMENT": 1.00, "PROVINCE": 0.85,
        "REGION": 0.75})

    # --- misc -----------------------------------------------------------
    bootstrap_iterations: int = 200              # FULL; FAST usa 60
    stability_high: float = 0.75
    stability_medium: float = 0.50
    random_seed: int = 42

    @property
    def boots(self) -> int:
        return self.bootstrap_iterations if self.execution_mode == "FULL" else 60

    def candidate_distributions(self, mode: str) -> tuple[str, ...]:
        """Candidatos según soporte (spec §25): en multiplicativo los shocks
        son positivos (soporte [0,∞)); en aditivo pueden ser negativos."""
        if mode == "multiplicative":
            base = ("lognormal", "gamma", "weibull", "normal", "student_t",
                    "empirical", "kde", "weighted_kde",
                    "log_kernel", "weighted_log_kernel")
        else:
            base = ("normal", "student_t", "empirical", "kde", "weighted_kde")
        if self.execution_mode == "FAST":
            base = tuple(d for d in base if d != "student_t")
        return base

    # complejidad relativa (§23, peso 0.05): parámetros efectivos / 10
    DIST_COMPLEXITY = {"normal": 0.2, "lognormal": 0.2, "gamma": 0.2,
                       "weibull": 0.2, "student_t": 0.3, "empirical": 0.4,
                       "kde": 0.5, "weighted_kde": 0.6,
                       "log_kernel": 0.5, "weighted_log_kernel": 0.6,
                       "ensemble": 0.7}
