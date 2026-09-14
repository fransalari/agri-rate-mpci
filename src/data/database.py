"""Capa de acceso a datos: Parquet + DuckDB, con fallback al sample."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "estimaciones.parquet"
SAMPLE = ROOT / "data" / "sample" / "chaco_girasol.csv"

SCHEMA = [
    "cultivo", "anio", "campania", "provincia", "departamento",
    "rendimiento_kgxha", "superficie_sembrada_ha",
    "superficie_cosechada_ha", "produccion_tm",
]


def load(parquet: Path | str | None = None) -> pd.DataFrame:
    """Carga el dataset principal; si no existe, usa el sample del repo."""
    path = Path(parquet) if parquet else PARQUET
    if path.exists():
        df = pd.read_parquet(path)
        source = str(path)
    else:
        df = pd.read_csv(SAMPLE)
        source = f"{SAMPLE} (dataset de ejemplo)"
    df = _normalize(df)
    df.attrs["source"] = source
    return df


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip().str.lower()
        .str.replace(" ", "_").str.replace(".", "", regex=False)
    )
    keep = [c for c in SCHEMA if c in df.columns]
    df = df[keep]
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce").astype("Int64")
    for c in ("rendimiento_kgxha", "superficie_sembrada_ha",
              "superficie_cosechada_ha", "produccion_tm"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("cultivo", "provincia", "departamento"):
        df[c] = df[c].astype(str).str.strip()
    return df.dropna(subset=["anio"]).reset_index(drop=True)


def query(sql: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """SQL sobre el dataset con DuckDB (tabla: estimaciones)."""
    import duckdb  # import perezoso: solo lo usa la app, no el pipeline de datos

    con = duckdb.connect()
    con.register("estimaciones", df if df is not None else load())
    try:
        return con.execute(sql).df()
    finally:
        con.close()


def series(df: pd.DataFrame, cultivo: str, departamento: str,
           desde: int | None = None, hasta: int | None = None) -> pd.DataFrame:
    """Serie histórica de rinde para un cultivo/departamento."""
    out = df[
        (df["cultivo"].str.casefold() == cultivo.casefold())
        & (df["departamento"].str.casefold() == departamento.casefold())
    ].copy()
    if desde is not None:
        out = out[out["anio"] >= desde]
    if hasta is not None:
        out = out[out["anio"] <= hasta]
    out = out.sort_values("anio").reset_index(drop=True)
    # rinde 0 con superficie cosechada 0 = pérdida total de área;
    # se mantiene (es señal de riesgo), pero rinde NaN se descarta.
    return out.dropna(subset=["rendimiento_kgxha"])
