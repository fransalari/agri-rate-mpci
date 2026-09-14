"""Esquemas de entrada/salida del engine (spec §40–41, §61)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import pandas as pd

ENGINE_VERSION = "0.1.0"


@dataclass
class ModelConfiguration:
    trend_model: str
    detrending_mode: str
    start_year: int
    half_life_years: float

    def label(self) -> str:
        hl = "∞" if self.half_life_years == float("inf") else f"{self.half_life_years:g}"
        return (f"{self.trend_model} · {self.detrending_mode[:4]} · "
                f"desde {self.start_year} · h={hl}")


@dataclass
class CandidateResult:
    config: ModelConfiguration
    status: str = "OK"                 # OK | INELIGIBLE | FAILED | REJECTED
    reason: str = ""
    score: float = float("nan")
    score_se: float = float("nan")
    prediction_error: float = float("nan")
    tail_error: float = float("nan")
    distribution_error: float = float("nan")
    residual_penalty: float = float("nan")
    complexity_penalty: float = float("nan")
    instability_penalty: float = float("nan")
    fold_scores: list = field(default_factory=list)
    aicc: float = float("nan")
    bic: float = float("nan")
    n_eff: float = float("nan")
    n_train: int = 0


@dataclass
class StructuralBreakReport:
    detected: bool = False
    year: int | None = None
    test: str = ""
    statistic: float = float("nan")
    p_value: float = float("nan")
    magnitude: float = float("nan")
    confidence: str = "NONE"           # NONE | WEAK | MODERATE | STRONG
    suggested_action: str = "KEEP"     # KEEP | DOWNWEIGHT | REVIEW | HARD_CUTOFF_CANDIDATE
    note: str = ""


@dataclass
class NormalizationResult:
    status: str                                    # OK | INSUFFICIENT_DATA
    selection_mode: str
    execution_mode: str
    recommended: ModelConfiguration | None = None
    numerical_champion: ModelConfiguration | None = None
    numerical_champion_score: float = float("nan")

    expected_yield: float = float("nan")
    expected_yield_ci: tuple[float, float] = (float("nan"), float("nan"))
    target_year: int | None = None

    normalized_history: pd.DataFrame | None = None  # year/observed/trend/shock/normalized/weight
    model_ranking: pd.DataFrame | None = None
    candidates: list = field(default_factory=list)

    validation_score: float = float("nan")
    score_se: float = float("nan")
    tail_score: float = float("nan")
    aicc: float = float("nan")
    bic: float = float("nan")

    raw_years: int = 0
    effective_sample_size: float = float("nan")
    sample_status: str = ""                        # FAVORABLE | MODERATE | LOW

    structural_break: StructuralBreakReport = field(default_factory=StructuralBreakReport)
    data_quality: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    historical_information_value: pd.DataFrame | None = None

    selection_confidence: str = ""                 # HIGH | MEDIUM | LOW
    selection_stability_pct: float = float("nan")
    selection_reason: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    governance: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Resumen serializable (sin DataFrames) para logging/API."""
        out = {
            "status": self.status,
            "recommended": asdict(self.recommended) if self.recommended else None,
            "numerical_champion": (asdict(self.numerical_champion)
                                   if self.numerical_champion else None),
            "expected_yield": self.expected_yield,
            "expected_yield_ci": self.expected_yield_ci,
            "target_year": self.target_year,
            "validation_score": self.validation_score,
            "score_se": self.score_se,
            "tail_score": self.tail_score,
            "effective_sample_size": self.effective_sample_size,
            "selection_confidence": self.selection_confidence,
            "selection_stability_pct": self.selection_stability_pct,
            "selection_reason": self.selection_reason,
            "warnings": self.warnings,
            "governance": self.governance,
        }
        return out


def governance_metadata(series: pd.DataFrame, config, selected, champion,
                        reasons: list) -> dict:
    """Metadata de reproducibilidad (spec §61)."""
    payload = series.to_csv(index=False).encode()
    return {
        "engine_version": ENGINE_VERSION,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_hash": hashlib.sha256(payload).hexdigest()[:16],
        "random_seed": config.random_seed,
        "execution_mode": config.execution_mode,
        "selection_mode": config.selection_mode,
        "selected_model": asdict(selected) if selected else None,
        "numerical_champion": asdict(champion) if champion else None,
        "selection_reason": reasons,
        "config": json.loads(json.dumps(
            {k: v for k, v in asdict(config).items()},
            default=str)),
    }
