"""
Conector MAGyP — Estimaciones Agrícolas
=======================================

Fuente identificada en el HAR (datosestimaciones.magyp.gob.ar):

1. DATASET COMPLETO (recomendado para agro-rate):
   POST https://datosestimaciones.magyp.gob.ar/reportes.php?reporte=Estimaciones
   Body (form-urlencoded): Dataset=Dataset
   → Respuesta: Content-Disposition: attachment; filename=Estimaciones.csv
   Contiene TODA la serie histórica (todas las campañas desde 1969/70,
   todos los cultivos, nivel departamento).

2. ENDPOINTS AUXILIARES (para poblar filtros de la app):
   POST /ajax/getProvincias.php            → [{"id":"22","descr":"CHACO"}, ...]
   POST /ajax/getDepartamentos.php         → [{"id":"22-105","dpto":"9 DE JULIO - CHACO"}, ...]
   (id = codProvincia-codDepartamento, códigos INDEC)

3. DESCARGA FILTRADA (mismo endpoint, campos del formulario):
   cultivo[]     → id numérico (25=Soja total, 32=Maíz, 14=Girasol, 2=Algodón, ...)
   variable[]    → "Sup. Sembrada" | "Sup. Cosechada" | "Producción" | "Rendimiento"
   desde / hasta → índice de campaña (0 ≈ 1966/67 ... 59 = 2025/26)
   agregacion    → "Total País" | "Total Provincia" | "Total Departamentos"
   provincia[]   → id INDEC (22 = Chaco)
   departamento[]→ "22-105", ...
   submit        → "Descargar"

Para agro-rate conviene la opción 1: un solo request, dataset completo,
y filtramos localmente con pandas/DuckDB. Es lo más reproducible.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://datosestimaciones.magyp.gob.ar"
DATASET_URL = f"{BASE_URL}/reportes.php?reporte=Estimaciones"

# ids de cultivo del formulario (por si se usa la descarga filtrada)
CULTIVOS = {
    "soja": 25, "soja 1ra": 34, "soja 2da": 35,
    "maiz": 32, "girasol": 14, "algodon": 2,
    "trigo": 28, "sorgo": 26, "cebada": 33, "arroz": 4,
}

HEADERS = {
    "User-Agent": "agro-rate/0.1 (analisis actuarial; contacto del repo en GitHub)",
    "Referer": DATASET_URL,
}


def download_full_dataset(dest: Path | str = "data/raw/estimaciones_magyp.csv",
                          timeout: int = 300) -> Path:
    """Descarga el dataset completo de Estimaciones Agrícolas (CSV)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    resp = requests.post(
        DATASET_URL,
        data={"Dataset": "Dataset"},
        headers=HEADERS,
        timeout=timeout,
    )
    resp.raise_for_status()

    # El servidor responde application/csv con filename=Estimaciones.csv
    dest.write_bytes(resp.content)
    return dest


def load_dataset(path: Path | str = "data/raw/estimaciones_magyp.csv") -> pd.DataFrame:
    """Lee y normaliza el CSV de MAGyP.

    Esquema esperado (mismo que la hoja 'Yields per department' del
    workbook MPCI): cultivo, anio, campania, provincia, departamento,
    rendimiento_kgxha, superficie_sembrada_ha, superficie_cosechada_ha,
    produccion_tm.
    """
    # El portal exporta con ';' y encoding latin-1 en algunas versiones;
    # probamos ambas combinaciones.
    for sep, enc in ((";", "utf-8"), (";", "latin-1"), (",", "utf-8"), (",", "latin-1")):
        try:
            df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
            if df.shape[1] > 3:
                break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    else:
        raise ValueError("No se pudo parsear el CSV de MAGyP")

    df.columns = (
        df.columns.str.strip().str.lower()
        .str.replace(" ", "_").str.replace(".", "", regex=False)
    )
    return df


def filter_data(df: pd.DataFrame,
                provincia: str | None = "Chaco",
                cultivos: list[str] | None = None,
                campania_desde: str | None = None) -> pd.DataFrame:
    """Filtra por provincia / cultivos / campaña inicial."""
    out = df.copy()
    if provincia:
        out = out[out["provincia"].str.strip().str.casefold() == provincia.casefold()]
    if cultivos:
        wanted = {c.casefold() for c in cultivos}
        out = out[out["cultivo"].str.strip().str.casefold().isin(wanted)]
    if campania_desde and "campania" in out.columns:
        out = out[out["campania"] >= campania_desde]
    return out.reset_index(drop=True)


def to_parquet(df: pd.DataFrame, dest: Path | str = "data/estimaciones.parquet") -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    return dest


if __name__ == "__main__":
    csv_path = download_full_dataset()
    print(f"Descargado: {csv_path} ({csv_path.stat().st_size/1e6:.1f} MB)")
    df = load_dataset(csv_path)
    print(df.head())
    print(f"{len(df):,} filas | columnas: {list(df.columns)}")
    chaco = filter_data(df, provincia="Chaco",
                        cultivos=["Soja total", "Maíz", "Girasol"])
    print(f"Chaco soja/maíz/girasol: {len(chaco):,} filas")
    to_parquet(chaco, "data/chaco.parquet")
