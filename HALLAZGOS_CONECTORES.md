# agro-rate — Hallazgos de los HAR y decisión de arquitectura de datos

## 1. MAGyP (datosestimaciones.magyp.gob.ar)

**Endpoint del dataset completo** (el botón verde "Dataset" de la página):

```
POST https://datosestimaciones.magyp.gob.ar/reportes.php?reporte=Estimaciones
Content-Type: application/x-www-form-urlencoded
Body: Dataset=Dataset
→ 200 OK, Content-Disposition: attachment; filename=Estimaciones.csv
```

Un solo request devuelve toda la serie histórica: todas las campañas
(índices 0→59, hasta 2025/26), 41 cultivos, nivel departamento, con las
4 variables (sup. sembrada, sup. cosechada, producción, rendimiento).

**Endpoints auxiliares** (útiles para poblar selects de la app):

| Endpoint | Devuelve |
|---|---|
| `POST /ajax/getProvincias.php` | `[{"id":"22","descr":"CHACO"}, ...]` (códigos INDEC) |
| `POST /ajax/getDepartamentos.php` | `[{"id":"22-105","dpto":"9 DE JULIO - CHACO"}, ...]` |
| `POST /ajax/getTableData.php` | HTML paginado de la tabla (no sirve para datos) |

**Descarga filtrada** (mismo `reportes.php`, campos del form): `cultivo[]`
(25=Soja total, 32=Maíz, 14=Girasol, 2=Algodón), `variable[]`, `desde`/`hasta`
(índice de campaña), `agregacion` (País/Provincia/Departamentos),
`provincia[]`, `departamento[]`, `submit=Descargar`.

**Decisión**: usar el dataset completo y filtrar localmente (DuckDB/Parquet).
Un request mensual desde GitHub Actions, sin lógica de formulario frágil.

## 2. CEDEI Chaco (cedei.produccion.chaco.gov.ar)

La página provincial **no sirve datos propios**: embebe un dashboard de
**Tableau Public**:

- Workbook: `campaashistorico` — Vista: `EstimacionesAgricolasAo1059-2024`
  ("Estimaciones Agrícolas Año 1959–2024")
- El HTML solo trae los datos de la vista visible (4 cultivos de una campaña,
  dentro del `bootstrapSession`).

Al ser Tableau Public, el dataset completo se baja sin sesión:

```
# CSV resumido de la vista
GET https://public.tableau.com/views/campaashistorico/EstimacionesAgricolasAo1059-2024.csv

# Workbook completo con extract embebido (.twbx = ZIP con .hyper adentro)
GET https://public.tableau.com/workbooks/campaashistorico.twb
```

**Rol**: validación cruzada de MAGyP para Chaco y posible extensión de la
serie ~10 campañas hacia atrás (1959 vs 1969). No la usaría como fuente
primaria: es un dashboard mantenido a mano, sin garantía de esquema estable.

## 3. Conexión con el workbook MPCI (`MPCI_Rate_making_Alt_1.xlsb`)

La hoja **"Yields per department"** tiene exactamente el esquema del export
de MAGyP (`cultivo, anio, campania, provincia, departamento,
rendimiento_kgxha, superficie_sembrada_ha, superficie_cosechada_ha,
produccion_tm`) — es un pegado manual de esa misma fuente. El conector
`magyp.py` reemplaza ese paso manual.

El resto del workbook mapea 1:1 a los módulos planificados:

| Hoja(s) del xlsb | Módulo agro-rate |
|---|---|
| Yields per department | `src/data/magyp.py` |
| 18 hojas por departamento (Area Yield, Min Farm Yield, distribución de has por rango, trigger) | `analytics/` + `pricing/yield_insurance.py` (área-yield con dispersión intra-departamento) |
| Results (Trigger, Rate, Prima vs Tasa Re/Prima Re, L/C, PML) | `pricing/burning_cost.py` + comparación con tasas de reaseguro |
| For Agro Cat (Losses/Premium/Loss Ratio 1969→, EPI) | módulo Loss Ratio + input para stop-loss agregado |

Es decir: la metodología que la app debe reproducir en V0.1–V0.3 ya está
especificada en el workbook; el conector cierra el ciclo de datos.

## 4. Pendiente de verificación (correr localmente)

El sandbox donde se analizaron los HAR no tiene salida de red hacia
`magyp.gob.ar` ni `public.tableau.com`, así que los dos scripts están
escritos contra la evidencia del HAR pero **sin ejecutar la descarga
real**. Verificar localmente:

```bash
pip install requests pandas pyarrow
python -m src.data.magyp      # baja Estimaciones.csv y genera chaco.parquet
python -m src.data.cedei      # baja la vista CSV + workbook twbx
```

Posibles ajustes menores al correrlo: separador/encoding del CSV de MAGyP
(el loader ya prueba `;`/`,` × `utf-8`/`latin-1`) y el nombre exacto de la
vista de Tableau si CEDEI la renombra.
