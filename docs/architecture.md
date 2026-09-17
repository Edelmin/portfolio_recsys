# Arquitectura del proyecto

Este documento describe la arquitectura del proyecto **Portfolio Recsys Comparison**:
una herramienta de apoyo a la construcción de carteras de inversión que combina
análisis financiero, aprendizaje automático y sistemas de recomendación.

Alcance de este documento: describe la **estructura del sistema** (capas de datos,
fases lógicas, pipelines, formatos y decisiones de diseño). El detalle funcional de
subsistemas concretos vive en documentos dedicados de `docs/`:
- `docs/portfolio-markowitz.md`: pipeline de carteras y backtesting Markowitz.
- `docs/reporting-domains.md`: reporting, observabilidad y validación de datos.
- `docs/releases.md`: versionado y publicación de releases.

---

## 1. Objetivo del sistema

Proporcionar una herramienta software que apoye la construcción de carteras de
inversión combinando análisis financiero, aprendizaje automático y sistemas de
recomendación. El sistema evalúa y compara distintas estrategias algorítmicas
(optimización de Markowitz, GNNs y modelos recurrentes RNN/GRU) entre sí.

El sistema permite:
- Extraer y procesar datos financieros desde múltiples fuentes
- Normalizar monedas a EUR
- Consolidar datos en un dataset unificado
- Preparar ventanas temporales para entrenamiento de modelos
- Entrenar modelos predictivos y construir carteras con distintas estrategias
- Evaluar y comparar el rendimiento de las carteras mediante backtesting

---

## 2. Capas de datos

```
data/
├── 01_raw/              ← Datos crudos originales (HTML, Excel, API responses)
├── 02_intermediate/     ← Datos extraídos, parseados, transformados (subcarpetas por etapa)
├── 03_processed/        ← Datos listos para consumir/entrenar modelos
├── 04_models/           ← Modelos entrenados (pesos .pt, artefactos HPO)
├── 05_model_outputs/    ← Predicciones y carteras generadas
└── 06_reporting/        ← Reportes, gráficas, dashboards
```

- **01_raw**: fuentes tal cual se obtienen (HTML, Excel, respuestas de API).
- **02_intermediate**: todo el procesamiento intermedio (parseo, muestreo, normalización,
  limpieza, joins). Puede tener subcarpetas internas con nombres descriptivos por etapa.
- **03_processed**: dataset definitivo, listo para alimentar modelos (features preprocesadas,
  ventanas temporales, etc.).
- **04_models**: modelos entrenados (pesos `.pt`, estudios Optuna y artefactos de HPO). No
  catalogado en Kedro; se accede por ruta.
- **05_model_outputs**: predicciones de los modelos y recomendaciones de cartera.
- **06_reporting**: visualizaciones, dashboards, reportes.

---

## 3. Fases lógicas del proyecto

El proyecto se organiza en **fases lógicas** (A–I) que reflejan el flujo real de
trabajo. Es importante distinguir entre la organización de carpetas en código (por
tipo de operación) y el orden real de ejecución (determinado por dependencias de datos
y decisiones manuales).

> Las carpetas `_1_ingest`, `_2_transform`, etc. clasifican los pipelines por
> **tipo de operación**, no por orden cronológico. El orden real de ejecución
> lo determina el DAG de Kedro a partir de las dependencias de datos.
> Por ejemplo, `fetch_stock_prices` está en `_1_ingest` porque es una ingesta
> (descarga datos externos), pero se ejecuta después de `stratified_random_sampling`
> (que está en `_2_transform`) porque necesita conocer los tickers de la muestra fijada.

| Fase | Nombre | Qué hace | Cuándo se ejecuta |
|------|--------|----------|-------------------|
| A | Ingesta de datos | Parsear HTML → universo de empresas por sector | Una vez, al inicio |
| B | Calibración de muestra | Muestreo estratificado + parseo de financials + validación de cobertura | Iterativo (2-3 ciclos manuales hasta fijar la muestra) |
| C | Alineación de datos | Extraer divisas de filenames, descargar precios y tipos de cambio | Una vez, post-fijación de muestra |
| D | Transformación | Normalizar divisas, validar precios, limpiar financials | Tras C |
| E | Consolidación | Join empresa + precios + financials en dataset unificado | Tras D |
| F | Preparación | Ventanas temporales para modelo (multi-horizonte) | Tras E, una vez por horizonte |
| G | Generación de modelos predictivos | Entrenamiento y validación de modelos | Tras F, una vez por horizonte |
| H | Selección de activos | Generación de la muestra de activos a optimizar | Tras G, una vez por horizonte |
| I | Optimización de carteras | Construcción de carteras propuestas (backtest Markowitz) | Tras H, una vez por horizonte |

> El detalle funcional de las fases H e I (construcción y evaluación de carteras) se
> documenta en `docs/portfolio-markowitz.md`.

---

## 4. Flujo general del proyecto


```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                      ARQUITECTURA GENERAL DEL SISTEMA                         ║
╚═══════════════════════════════════════════════════════════════════════════════╝

  ┌─────────┐   ┌───────────────────────────────────────────────────────────────┐
  │         │   │  Fase A — Ingesta de datos                                    │
  │         │   │  Parseo HTML → universo de empresas por sector                │
  │         │   └───────────────────────────────┬───────────────────────────────┘
  │         │                                   │
  │         │                                   ▼
  │         │   ┌───────────────────────────────────────────────────────────────┐
  │         │   │  Fase B — Calibración de muestra (iterativa)                  │
  │         │   │  Muestreo estratificado → parseo de estados financieros       │
  │ ETAPA 1 │   │  → validación de cobertura                                    │◀─┐
  │         │   └───────────────────────────────┬───────────────────────────────┘  │
  │         │                                   │                                  │
  │         │                                   ▼                                  │
  │         │                        ╱───────────────────╲          No            │
  │         │                       ╱ ¿Cobertura suficiente?╲───────────────────────┘
  │         │                       ╲                       ╱
  │         │                        ╲─────────┬───────────╱
  │         │                          Sí      │      Sí
  │         │                 ┌────────────────┴────────────────┐
  │         │                 ▼                                 ▼
  │         │   ┌──────────────────────────┐   ┌──────────────────────────────┐
  │         │   │  Fase C — Alineación de   │   │  Fase C — Alineación de      │
  │         │   │  datos                    │   │  datos                       │
  │         │   │  Extraer divisas ·        │   │  Descargar precios y         │
  │         │   │  Enriquecer sectores      │   │  tipos de cambio             │
  │         │   └─────────────┬─────────────┘   └───────────────┬──────────────┘
  │         │                 └────────────────┬────────────────┘
  │         │                                  ▼
  │         │   ┌───────────────────────────────────────────────────────────────┐
  │         │   │  Fase D — Transformación                                      │
  │         │   │  Normalizar divisas · Validar precios · Limpiar estados fin.  │
  │         │   └───────────────────────────────┬───────────────────────────────┘
  │         │                                   ▼
  │         │   ┌───────────────────────────────────────────────────────────────┐
  │         │   │  Fase E — Consolidación                                       │
  │         │   │  Join empresa + precios + estados financieros → unificado     │
  │         │   └───────────────────────────────┬───────────────────────────────┘
  │         │                                   ▼
  │         │   ┌───────────────────────────────────────────────────────────────┐
  │         │   │  Fase F — Preparación para modelo                             │
  │         │   │  Feature engineering · Split temporal · Escalado · Ventanas   │
  └─────────┘   └───────────────────────────────┬───────────────────────────────┘
                                                │
  ┌─────────┐   ┌───────────────────────────────────────────────────────────────┐
  │ ETAPA 2 │   │  Fase G — Generación de modelos predictivos                   │
  │         │   │  Entrenamiento y validación                                   │
  └─────────┘   └───────────────────────────────┬───────────────────────────────┘
                                                │
  ┌─────────┐   ┌───────────────────────────────────────────────────────────────┐
  │ ETAPA 3 │   │  Fase H — Selección de activos prometedores                   │
  │         │   │  Generación de la muestra de activos a optimizar              │
  └─────────┘   └───────────────────────────────┬───────────────────────────────┘
                                                │
  ┌─────────┐   ┌───────────────────────────────────────────────────────────────┐
  │ ETAPA 4 │   │  Fase I — Optimización de carteras                            │
  │         │   │  Generación de carteras propuestas                            │
  └─────────┘   └───────────────────────────────────────────────────────────────┘
```

---

## 5. Ciclo de calibración de la muestra (Fase B)

La Fase B es el único punto del flujo con un ciclo iterativo manual: se parte de un
muestreo estratificado del universo de empresas, se parsean sus estados financieros y
se valida la cobertura de datos (Income Statement, Cash Flow, Ratios y años de
historia). Si la cobertura es insuficiente, se reajusta la muestra y se repite; cuando
es suficiente, la muestra se fija y se persiste. A partir de ese punto el pipeline es
determinista (DAG estático) y los pipelines posteriores operan sobre la muestra fijada
como input estable. Esta realimentación es la que refleja el rombo "¿Cobertura
suficiente?" del diagrama de la sección 4.

---

## 6. Relación carpetas (código) vs fases (flujo real)

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║       CARPETAS (organizacion de codigo) vs FASES (orden de ejecucion)         ║
╚═══════════════════════════════════════════════════════════════════════════════╝

  Carpeta en codigo                    Fase logica          Orden real
  ─────────────────────────────────    ──────────────────   ──────────
  _1_ingest/
    ├─ parse_financials_from_html  ──▶ Fase A (base)        1
    ├─ parse_financial_statements  ──▶ Fase B (calibracion) 2 ◀─── ciclo
    ├─ extract_currencies_from_fn  ──▶ Fase C (derivada)    3
    ├─ fetch_exchange_rates        ──▶ Fase C (derivada)    3
    └─ fetch_stock_prices          ──▶ Fase C (derivada)    3

  _2_transform/
    ├─ stratified_random_sampling  ──▶ Fase B (calibracion) 2 ◀─── ciclo
    ├─ clean_financial_statements  ──▶ Fase D (transform)   4
    ├─ normalize_currencies        ──▶ Fase D (transform)   4
    └─ validate_price_currencies   ──▶ Fase D (transform)   4

  _3_consolidate/
    └─ build_company_dataset       ──▶ Fase E (consolid.)   5

  _4_prepare/
    └─ build_temporal_windows      ──▶ Fase F (preparacion) 6


  Nota: El "orden real" asume ejecución post-fijación (DAG estático).
        La Fase B fue iterativa durante desarrollo; en reproducción es ①→②→③→④→⑤→⑥.
```

---

## 7. Flujo de datos entre fases

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    FLUJO DE DATOS ENTRE FASES                                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝

  Fase A                          Fase B                         Fase C
  ───────                         ──────                         ──────
  HTML TIKR                       SECTOR_{cap}_{sector}          SAMPLE fijada
      │                               │                              │
      ▼                               ▼                              ├──▶ filenames Excel
  ┌────────┐                    ┌────────────┐                       │      │
  │SECTOR_ │                    │  SAMPLE_   │                       │      ▼
  │{cap}_  │                    │  {cap}_    │                       │  currencies_{sector}
  │{sector}│                    │  {sector}  │                       │      │
  └────┬───┘                    └─────┬──────┘                       │      ▼
       │                              │                              │  exchange_rates
       │                              ▼                              │
       │                        ┌────────────┐                       ├──▶ tickers
       │                        │  FS_{sect} │                       │      │
       │                        │  (parquet) │                       │      ▼
       │                        └────────────┘                       │  stock_prices
       │                                                             │
       └──────────── input ──────▶ Fase B                            │
                                                                     │
  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

  Fase D                          Fase E                         Fase F
  ───────                         ──────                         ──────
  FS limpio                       Todos los anteriores           Dataset unificado
  Precios EUR                          │                              │
  Validaciones                         ▼                              ▼
      │                        ┌──────────────────┐          ┌──────────────────┐
      └──────────────────────▶ │ company_dataset  │────────▶ │ temporal_windows │
                               │ (02_intermediate)│          │ (03_processed)   │
                               └──────────────────┘          └──────────────────┘
```

---

## 8. Catálogo de pipelines

### Fase A: Ingesta base

| Pipeline | Entrada | Salida | Descripción |
|----------|---------|--------|-------------|
| `parse_financials_from_html` | HTML descargado manualmente (TIKR) | Parquet por sector | Parsea tablas de empresas: símbolo, industria, sector, ubicación |

### Fase B: Calibración de muestra (ciclo iterativo)

| Pipeline | Entrada | Salida | Descripción |
|----------|---------|--------|-------------|
| `stratified_random_sampling` | Dataset de empresas por sector | Muestra representativa | Selección estadística (confianza 95%, error 5%) |
| `parse_financial_statements` | Excel multihojas (TIKR) | Parquet unificado | Lee Income Statement, Cash Flow, Ratios y unifica (engine=calamine) |

### Fase C: Ingesta derivada (post-fijación)

| Pipeline | Entrada | Salida | Descripción |
|----------|---------|--------|-------------|
| `extract_currencies_from_filenames` | Nombres de fichero Excel | Parquet con divisas | Extrae moneda y ticker del patrón del filename |
| `fetch_exchange_rates` | Lista de divisas | Parquet de tipos de cambio | Descarga pares XXX/EUR históricos de Yahoo Finance |
| `fetch_stock_prices` | Lista de tickers | Parquet de precios | Descarga precios diarios de Yahoo Finance (ajustados por splits/dividendos y reparados: `auto_adjust=True`, `repair=True`) |

### Fase D: Transformación

| Pipeline | Entrada | Salida | Descripción |
|----------|---------|--------|-------------|
| `clean_financial_statements` | Estados financieros unificados | Dataset limpio | Trata missing values, outliers, inconsistencias |
| `normalize_currencies` | Precios + tipos de cambio | Precios en EUR | Convierte todas las monedas a EUR y descarta activos con precios inválidos/penny stocks (ver sección 8.1) |
| `validate_price_currencies` | Precios + info empresas | Reporte de discrepancias | Verifica moneda publicada vs esperada |

### Fase E: Consolidación

| Pipeline | Entrada | Salida | Descripción |
|----------|---------|--------|-------------|
| `build_company_dataset` | Empresas + precios EUR + estados financieros | Dataset unificado | Join por ticker/fecha: empresa, precio, sector, métricas |

### Fase F: Preparación para modelo

| Pipeline | Entrada | Salida | Descripción |
|----------|---------|--------|-------------|
| `compute_model_features` | Dataset consolidado | Features escaladas + split | Feature engineering, split temporal, escalado robusto |
| `build_temporal_windows` | Features escaladas | Ventanas train/val/test | Partición temporal móvil (tamaño de ventana configurable, p. ej. 60 sesiones), sin data leakage |

### Fases G–I: Modelado y carteras

| Pipeline | Entrada | Salida | Descripción |
|----------|---------|--------|-------------|
| (entrenamiento) | Ventanas train/val/test | Modelos entrenados + predicciones | Fase G: entrenamiento y validación del modelo recurrente por horizonte |
| `report_portfolio` | Predicciones + precios EUR | Informe Markowitz + JSON | Fases H–I: selección de activos y backtest de carteras optimizadas por horizonte de holding |

> Estas fases se ejecutan una vez por horizonte de predicción (1d, 1w, 2w, 1m, 2m, 3m,
> 6m, 1y): el modelo se entrena por horizonte y las predicciones se persisten antes del
> backtest. El detalle completo del entrenamiento, la selección de activos y la
> optimización de carteras se documenta en `docs/portfolio-markowitz.md`.

---

## 8.1. Control de calidad de las series de precios

Las series de precios de Yahoo Finance contienen, para un subconjunto pequeño de
activos, valores corruptos (saltos de gran magnitud por splits mal aplicados,
cotizaciones nulas y penny stocks poco fiables) que contaminan el target del modelo y
la matriz de covarianza del backtest. Como decisión de arquitectura, el control de
calidad se distribuye a lo largo del pipeline en cuatro medidas complementarias,
parametrizadas en `conf/base/parameters.yaml` (clave `price_quality`):

1. **Descarga ajustada y reparada** (`fetch_stock_prices`): ajuste por splits y
   dividendos y reparación de errores conocidos en el origen.
2. **Descarte de activos inválidos** (`normalize_currencies`): se excluye del universo
   de modelado la empresa completa cuando su cotización es nula/corrupta o es un penny
   stock. No afecta al benchmark.
3. **Recorte de retornos desorbitados** (`compute_model_features`): winsorización de
   los log-retornos a un límite que se escala con el horizonte.
4. **Cuarentena de ventanas anómalas** (`build_temporal_windows`): se descartan las
   ventanas de train/validation con retornos fuera de las vallas de Tukey de su sector;
   los umbrales se estiman solo con train+validation y las ventanas de test se conservan
   íntegras.

Cambiar cualquiera de estas medidas altera precios, target y features, por lo que
invalida los modelos y backtests aguas abajo y obliga a regenerar desde el punto
afectado. Los valores concretos de los parámetros y las notas de operación se
documentan en `docs/environment.md`.

---

## 9. Organización de carpetas en código

Las carpetas organizan el código por **tipo de operación** (no por orden de ejecución):

```
src/portfolio_recsys/pipelines/
├── _1_ingest/           ← Pipelines que traen datos desde fuentes externas
│   ├── parse_financials_from_html/
│   ├── parse_financial_statements/
│   ├── extract_currencies_from_filenames/
│   ├── fetch_exchange_rates/
│   └── fetch_stock_prices/
├── _2_transform/        ← Pipelines que limpian, normalizan, validan o muestrean
│   ├── stratified_random_sampling/
│   ├── clean_financial_statements/
│   ├── enrich_sectors_with_currencies/
│   ├── normalize_currencies/
│   └── validate_price_currencies/
├── _3_consolidate/      ← Pipelines que cruzan/unen datasets
│   └── build_company_dataset/
├── _4_prepare/          ← Pipelines que formatean para consumo del modelo
│   ├── compute_model_features/
│   ├── build_temporal_windows/
│   └── impute_missing_values/
└── _5_reporting/        ← Pipelines de evaluacion, reporting y carteras
    ├── report_sampling/         ← Informe de muestra (Markdown; PDF via CLI)
    ├── report_matching/         ← Discrepancias del matching sector-divisa
    ├── report_currencies/
    ├── report_data_quality/
    ├── report_exchange_rates/
    ├── report_normalization/
    ├── report_stock_prices/
    ├── report_company_dataset/
    ├── report_portfolio/        ← Backtest Markowitz
    └── observability/
```

---

## 10. Formato de datos

### Convención de formato

| Capa | Formato de almacenamiento |
|------|--------------------------|
| 01_raw | Original (HTML, Excel .xlsx) |
| 02_intermediate en adelante | **Parquet** vía `polars.EagerPolarsDataset` |

### Convención de lectura/escritura

- Fuentes raw en Excel: se leen con `polars.read_excel(engine="calamine")`
- Fuentes raw en HTML: se leen con `text.TextDataset` + BeautifulSoup
- Todos los datasets intermedios y superiores: Parquet, manipulados con `polars`
- El catálogo de Kedro serializa/deserializa de forma transparente

### Convención de nombres de datasets

Los identificadores de datasets usan **solo caracteres ASCII** (sin tildes):

| Prefijo | Patrón | Ejemplo |
|---------|--------|---------|
| `HTML_` | `HTML_{cap}_{sectorEs}` | `HTML_LargeCaps_BienesRaices` |
| `SECTOR_` | `SECTOR_{cap}_{sectorEn}` | `SECTOR_LargeCaps_RealEstate` |
| `SAMPLE_` | `SAMPLE_{cap}_{sectorEn}` | `SAMPLE_LargeCaps_Energy` |
| `FS_` | `FS_{sectorEs}` | `FS_Financiero` |
| `currencies_` | `currencies_{sectorEn}` | `currencies_Financials` |

Los `path:` en el catálogo SÍ pueden llevar tildes cuando apuntan a carpetas
reales del filesystem que las tienen.

---

## 11. Dataset Factories (catálogo Kedro)

Se usan dataset factories para evitar repetición. Un solo pattern en YAML
resuelve múltiples datasets automáticamente:

```yaml
# Ejemplo: un pattern resuelve 22 datasets (11 sectores x 2 capitalizaciones)
"SECTOR_{cap}_{sector}":
  type: polars.EagerPolarsDataset
  filepath: "${globals:path.intermediate}/${globals:step.financials_csv}/{cap}_{sector}.parquet"
```

Cuando un pipeline referencia `SECTOR_LargeCaps_RealEstate`, Kedro sustituye
`{cap}=LargeCaps` y `{sector}=RealEstate` y resuelve el filepath.

---

## 12. Decisiones de diseño

| Decisión | Justificación |
|----------|---------------|
| Kedro como orquestador | Reproducibilidad, catálogo explícito, trazabilidad |
| Polars sobre Pandas | Rendimiento, menor memoria, API expresiva |
| Parquet como formato | Compresión, tipado, lectura parcial por columnas |
| Dataset factories | Elimina repetición, escala sin tocar YAML |
| Carpetas por tipo de operación | Separación clara de responsabilidades; el DAG resuelve el orden |
| Fases lógicas documentadas | Reflejan el flujo real (incluyendo ciclos manuales) sin forzar linealidad |
| `calamine` para Excel | Más robusto y rápido que openpyxl con ficheros xlsx |
| uv como gestor de entorno | Lockfile reproducible, rápido, sin conda |
