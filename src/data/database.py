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
    """Normaliza cualquiera de los dos esquemas de MAGyP al esquema interno."""
    df = df.copy()

    def clean(name: str) -> str:
        s = unicodedata.normalize("NFKD", str(name))
        s = "".join(c for c in s if not unicodedata.combining(c))  # sin tildes
        s = s.lower().strip()
        for junk in ("(ha)", "(tn)", "(tm)", "(kg/ha)", "(kgxha)", "."):
            s = s.replace(junk, "")
        return s.strip().replace(" ", "_")

    df.columns = [clean(c) for c in df.columns]

    # mapeo por contenido: cubre "sup_sembrada", "superficie_sembrada_ha", etc.
    def find(*tokens, exclude=("id",)):
        for col in df.columns:
            if all(t in col for t in tokens) and not any(col.startswith(e) for e in exclude):
                return col
        return None

    rename = {}
    for target, tokens in {
        "cultivo": ("cultivo",),
        "campania": ("campa",),
        "provincia": ("provincia",),
        "departamento": ("departamento",),
        "rendimiento_kgxha": ("rendimiento",),
        "superficie_sembrada_ha": ("sembrada",),
        "superficie_cosechada_ha": ("cosechada",),
        "produccion_tm": ("produccion",),
    }.items():
        col = find(*tokens)
        if col and col != target:
            rename[col] = target
    df = df.rename(columns=rename)

    missing = [c for c in ("cultivo", "campania", "provincia", "departamento",
                           "rendimiento_kgxha") if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas no encontradas: {missing}. "
                         f"Columnas del CSV: {list(df.columns)}")

    # anio: si no viene, se deriva de la campaña ("2024/25" → 2024)
    if "anio" not in df.columns:
        df["anio"] = df["campania"].astype(str).str.extract(r"(\d{4})")[0]

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
