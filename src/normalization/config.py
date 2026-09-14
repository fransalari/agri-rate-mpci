"""Configuración central del Yield Risk Normalization Engine.

Todos los parámetros, pesos y umbrales viven acá — sin magic numbers
dispersos (spec §44).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScoreWeights:
    """Pesos del ActuarialValidationScore (spec §26). Lower score = better."""
    prediction: float = 0.30
    tail: float = 0.25
    distribution: float = 0.15
    residual: float = 0.10
    complexity: float = 0.10
    instability: float = 0.10


@dataclass
class YieldNormalizationConfig:
    # --- selección ---------------------------------------------------
    selection_mode: str = "AUTO"            # AUTO | MANUAL | BENCHMARK
    execution_mode: str = "FAST"            # FAST | FULL

    candidate_models: tuple[str, ...] = (
        "constant", "linear", "robust_linear", "log_linear",
        "quadratic", "piecewise", "spline",
    )
    candidate_detrending_modes: tuple[str, ...] = ("multiplicative", "additive")
    candidate_half_lives: tuple[float, ...] = (10, 15, 20, 25, 30, 40, float("inf"))
    candidate_start_year_step: int = 5

    # --- validación rolling-origin (spec §6) --------------------------
    minimum_training_years: int = 10
    max_validation_years: int = 12          # ventana común de validación
    forecast_horizon: int = 1

    # --- muestra ------------------------------------------------------
    minimum_series_length: int = 8
    minimum_effective_sample_size: float = 20.0
    n_eff_favorable: float = 30.0

    # --- score --------------------------------------------------------
    score_weights: ScoreWeights = field(default_factory=ScoreWeights)
    tail_quantiles: tuple[float, ...] = (0.10, 0.20, 0.30)

    # --- complejidad (spec §14): df efectivos por modelo --------------
    model_df: dict = field(default_factory=lambda: {
        "constant": 1, "linear": 2, "robust_linear": 2, "log_linear": 2,
        "quadratic": 3, "piecewise": 4, "spline": 5, "loess": 6,
    })
    # penalidad = df / complexity_scale → proporcional a parámetros
    # por serie "típica" (~20 obs útiles). Centralizado y defendible.
    complexity_scale: float = 20.0

    # --- guardrails agronómicos (spec §56) ----------------------------
    max_annual_tech_growth: float = 0.08    # >8 %/año sostenido → warning
    max_extrapolation_years: int = 3        # target muy lejos del dato → warning
    spline_df: int = 4                      # spline restringido
    piecewise_min_segment: int = 8          # obs mínimas por segmento

    # --- structural breaks (spec §23) ---------------------------------
    structural_break_enabled: bool = True
    break_alpha: float = 0.05

    # --- HIV / bootstrap ----------------------------------------------
    historical_information_value_enabled: bool = True   # solo FULL
    hiv_block_years: int = 5
    bootstrap_iterations_fast: int = 80
    bootstrap_iterations_full: int = 400

    # --- confianza ----------------------------------------------------
    stability_high: float = 0.75
    stability_medium: float = 0.50

    random_seed: int = 42

    # ------------------------------------------------------------------
    @property
    def bootstrap_iterations(self) -> int:
        return (self.bootstrap_iterations_full if self.execution_mode == "FULL"
                else self.bootstrap_iterations_fast)

    def effective_candidates(self) -> tuple[tuple[str, ...], tuple[float, ...]]:
        """Grid reducido en FAST para UI interactiva (spec §45)."""
        if self.execution_mode == "FULL":
            return self.candidate_models, self.candidate_half_lives
        models = tuple(m for m in self.candidate_models if m != "quadratic")
        half_lives = (15, 25, float("inf"))
        return models, half_lives

    def complexity_penalty(self, model: str) -> float:
        return self.model_df.get(model, 6) / self.complexity_scale
