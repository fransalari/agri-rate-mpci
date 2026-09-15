"""Esquemas de salida del Yield Risk Distribution Engine (spec §54-55, §70)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

ENGINE_VERSION = "0.1.0"


@dataclass
class DistCandidateResult:
    name: str
    status: str = "OK"          # OK | REJECTED_RISK_UNDERDISPERSION | FAILED
    reason: str = ""
    risk_score: float = float("nan")
    score_se: float = float("nan")
    tail: float = float("nan")
    indemnity: float = float("nan")
    distribution: float = float("nan")
    volatility_pen: float = float("nan")
    stability_pen: float = float("nan")
    complexity: float = float("nan")
    aicc: float = float("nan")
    downside_sd_ratio: float = float("nan")
    freq_ratio: float = float("nan")
    p10_sim: float = float("nan")
    fold_scores: list = field(default_factory=list)
    coverage_table: list = field(default_factory=list)


@dataclass
class RiskModelResult:
    status: str                                  # OK | INSUFFICIENT_DATA
    selection_mode: str = "AUTO"
    execution_mode: str = "FAST"

    # trend (heredado del motor de normalización)
    trend: dict = field(default_factory=dict)
    volatility: dict = field(default_factory=dict)
    distribution: dict = field(default_factory=dict)
    zero_mass: dict = field(default_factory=dict)
    sample: dict = field(default_factory=dict)
    tail: dict = field(default_factory=dict)     # P05/P10/P20/P30/ES en kg/ha
    insurance_validation: dict = field(default_factory=dict)
    selection: dict = field(default_factory=dict)

    normalized_history: pd.DataFrame | None = None
    ranking: pd.DataFrame | None = None
    candidates: list = field(default_factory=list)
    simulated_yields: object = None              # np.ndarray (muestra final)
    claim_curve: pd.DataFrame | None = None      # cobertura vs prima pura
    warnings: list = field(default_factory=list)
    governance: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)


def dist_governance(serie: pd.DataFrame, config, result: RiskModelResult) -> dict:
    payload = serie.to_csv(index=False).encode()
    rejected = [{"name": c.name, "status": c.status, "reason": c.reason}
                for c in result.candidates if c.status != "OK"]
    return {
        "engine_version": ENGINE_VERSION,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_hash": hashlib.sha256(payload).hexdigest()[:16],
        "random_seed": config.random_seed,
        "execution_mode": config.execution_mode,
        "aggregation_level": result.zero_mass.get("aggregation_level"),
        "zero_method": result.zero_mass.get("estimation_method"),
        "selected": result.distribution.get("recommended"),
        "aic_champion": result.distribution.get("numerical_aic_champion"),
        "rejected_models": rejected,
        "selection_reason": result.selection.get("reason", []),
        "validation_years": result.insurance_validation.get("val_years"),
        "warnings": result.warnings,
    }
