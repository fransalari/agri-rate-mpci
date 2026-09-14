"""
Conector CEDEI Chaco — Estadísticas históricas de cultivos 1959–2024
====================================================================

Hallazgo del HAR (cedei.produccion.chaco.gov.ar):

La página de CEDEI NO sirve los datos desde el sitio provincial: embebe
un dashboard de **Tableau Public**:

    workbook: campaashistorico  ("campañas historico")
    vista:    EstimacionesAgricolasAo1059-2024
              ("Estimaciones Agricolas Año 1959 - 2024")

El bootstrapSession del HAR solo contiene los datos de la vista visible
(4 cultivos de una campaña), pero al ser Tableau Public el dataset
completo se puede bajar sin sesión:

1. CSV de la vista (datos resumidos, lo más simple):
   GET https://public.tableau.com/views/campaashistorico/EstimacionesAgricolasAo1059-2024.csv

2. Workbook completo con el extract embebido (.twbx → .hyper):
   GET https://public.tableau.com/workbooks/campaashistorico.twb
   (se descarga un twbx: es un ZIP; adentro viene el extract .hyper con
   toda la serie 1959–2024, legible con la librería `tableauhyperapi`
   o con `pantab`)

Rol en agro-rate: fuente de VALIDACIÓN/EXTENSIÓN para Chaco.
MAGyP arranca en 1969/70; CEDEI declara serie desde 1959 → puede
extender ~10 campañas la serie histórica para burning cost.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

WORKBOOK = "campaashistorico"
VIEW = "EstimacionesAgricolasAo1059-2024"

VIEW_CSV_URL = f"https://public.tableau.com/views/{WORKBOOK}/{VIEW}.csv"
WORKBOOK_URL = f"https://public.tableau.com/workbooks/{WORKBOOK}.twb"

HEADERS = {"User-Agent": "agro-rate/0.1"}


def download_view_csv(dest: Path | str = "data/raw/cedei_view.csv",
                      timeout: int = 120) -> Path:
    """Baja el CSV resumido de la vista pública de Tableau."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(VIEW_CSV_URL, headers=HEADERS, timeout=timeout,
                        params={":showVizHome": "no"})
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def download_workbook(dest: Path | str = "data/raw/cedei_workbook.twbx",
                      timeout: int = 300) -> Path:
    """Baja el workbook completo (twbx) con el extract de datos embebido."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(WORKBOOK_URL, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def extract_hyper(twbx_path: Path | str,
                  out_dir: Path | str = "data/raw/cedei_extract") -> list[Path]:
    """Extrae los archivos .hyper del twbx (es un ZIP)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(twbx_path) as zf:
        for name in zf.namelist():
            if name.endswith(".hyper"):
                target = out_dir / Path(name).name
                target.write_bytes(zf.read(name))
                extracted.append(target)
    return extracted


def read_hyper(hyper_path: Path | str) -> pd.DataFrame:
    """Lee un extract .hyper a DataFrame (requiere `pip install pantab`)."""
    import pantab  # import perezoso: dependencia opcional

    frames = pantab.frames_from_hyper(str(hyper_path))
    # frames: dict {TableName: DataFrame}; normalmente hay una sola tabla
    key = next(iter(frames))
    return frames[key]


if __name__ == "__main__":
    p = download_view_csv()
    print(f"Vista CSV: {p} ({p.stat().st_size/1e3:.0f} KB)")
    print(pd.read_csv(p).head())

    twbx = download_workbook()
    hypers = extract_hyper(twbx)
    print("Extracts:", hypers)
    if hypers:
        df = read_hyper(hypers[0])
        print(df.head(), df.shape)
