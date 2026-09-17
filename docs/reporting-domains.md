# Reporting, Observabilidad y Validación de Datos

Este documento describe la estrategia de reporting, observabilidad y validación del proyecto Portfolio Recsys.

## Arquitectura

Los pipelines de reporting se ubican en `src/portfolio_recsys/pipelines/_5_reporting/` y sus outputs en `data/06_reporting/{dominio}/`. Son pipelines de solo lectura: consumen datasets materializados por otros pipelines pero no los transforman.

```
_5_reporting/
├── report_sampling/         # Muestra estratificada (Markdown; PDF via CLI pr-report-pdf)
├── report_currencies/       # Distribución de divisas
├── report_matching/         # Calidad del matching sector-divisa
├── report_data_quality/     # Completitud de estados financieros
├── report_stock_prices/     # Validación de precios descargados (dominio `prices`)
├── report_company_dataset/  # Validación del dataset consolidado por empresa
├── report_normalization/    # Conversión de precios a EUR y cobertura
├── report_exchange_rates/   # Validación de pares de divisa
├── report_portfolio/        # Rendimiento de carteras
└── observability/           # Métricas técnicas de ejecución
```

---

## Dominios de reporting

### Criterio de organización

Un dominio agrupa informes que se consultan juntos para tomar la misma decisión. Si un informe se entiende sin contexto del otro, van en dominios separados. Máximo 3-5 nodos por dominio.

### Tabla de dominios

| Dominio | Pregunta de decisión | Datasets consumidos | Formato output |
|---------|---------------------|---------------------|----------------|
| `sampling` | ¿Qué empresas descargar? ¿Es representativa la muestra? | `SAMPLE_*` | Markdown (`report_sampling`) + PDF (CLI `pr-report-pdf`) |
| `currencies` | ¿Qué divisas hay? ¿Cuántas empresas por moneda? | `currencies_*` | JSON |
| `matching` | ¿Se emparejaron bien sectores y divisas? | `ENRICHED_*` | Markdown (`report_matching`) |
| `data_quality` | ¿Están completos los estados financieros? ¿Cobertura, nulls? | Parquets FS | JSON (`report_data_quality`) |
| `prices` | ¿Se descargaron bien los precios? ¿Gaps? ¿Monedas? | `stock_prices_fetch_status` | JSON (`report_stock_prices`) |
| `eda` | ¿Cómo se distribuyen las features? ¿Correlaciones? | datasets consolidados | Notebooks `0.x` (no es pipeline) |
| `portfolio` | ¿Cómo rinden las carteras? Comparación estrategias | model outputs | Tablas, gráficas |

### Dominio `portfolio` (detalle)

El dominio `portfolio` es el más complejo. Implementa un backtest de carteras
Markowitz parametrizado por horizonte de predicción. Para cada horizonte:

- Se generan fechas de rebalanceo (cada `holding_period_sessions` fechas de señal).
- En cada rebalanceo se seleccionan los top-K tickers por predicción del modelo.
- Se optimizan pesos vía Markowitz (long-only y long-short).
- Se simula el rendimiento real hasta el siguiente rebalanceo.
- Se compara vs equiponderado (baseline) y los comparadores de mercado: S&P 500
  buy & hold y cartera Markowitz {S&P 500, activo seguro}.

**Configuraciones evaluadas**: top-K = {5, 10, 15, 20} × {long-only, long-short} + equiponderado, más los comparadores de mercado (S&P 500 buy & hold y S&P 500 + activo seguro)

**Métricas**: retorno total, retorno anualizado, volatilidad, Sharpe, max drawdown.

**Pipeline**: `report_portfolio` (ver `docs/portfolio-markowitz.md` para documentación completa).

**Datasets consumidos**: `prediction_table_test` (por defecto el split de test;
también admite `prediction_table` de validación), `stock_prices_eur`,
`benchmark_prices_eur`, `split_configuration`.

**Datasets producidos** (por combinación `{window}/{segmentation}/{horizon}`):
`portfolio_backtest_results` (JSON) y `portfolio_report` (Markdown), con variantes
por tipo de dataset (`_market`, `_enriched`).

---

## Análisis exploratorio (EDA)

El dominio `eda` NO se implementa como pipeline de Kedro: se realiza de forma
exploratoria en los notebooks `0.x` (iterativo y visual). Cubre:

- **Distribuciones**: histogramas y boxplots de métricas financieras por sector/cap.
- **Correlaciones**: matrices de correlación entre features numéricas. Útil para selección de features y detección de multicolinealidad.
- **Outliers**: detección por sector y capitalización.
- **Evolución temporal**: series temporales de variables clave agregadas.

Notebooks relevantes: `0.00 Análisis exploratorio.ipynb`,
`0.01 Analisis rentabilidad anomala por sector.ipynb`, y análisis afines
(`4.00 Inspect company dataset.ipynb`, `9.6 Data analysis.ipynb`).

---

## Observabilidad del pipeline

Métricas técnicas capturadas automáticamente durante la ejecución:

| Métrica | Descripción | Mecanismo |
|---------|-------------|-----------|
| Tiempos de ejecución | Duración por nodo y del run completo | Hooks `before_node_run` / `after_node_run` |
| Nodos fallidos | Estado de error por nodo y del run | Hooks `on_node_error` / `on_pipeline_error` |
| Datasets generados | Outputs del pipeline que se materializan en disco | Hook `after_pipeline_run` |
| Trazabilidad | Timestamp, parámetros del run, commit y estado de git | `RunAudit` (recopila git commit/branch/dirty) |
| Historial | Log estructurado de todas las ejecuciones | JSON por run + JSONL histórico en `06_reporting/observability/` |

### Implementación

La instrumentación vive en `src/portfolio_recsys/hooks.py` (`RunLoggingHooks`), que
delega en `src/portfolio_recsys/observability/audit.py` (`RunAudit`) sin modificar
los nodos. Cada ejecución escribe un JSON detallado por run en
`data/06_reporting/observability/runs/{run_id}.json` y añade una línea al histórico
`runs.jsonl`. Es tolerante a fallos: si la auditoría falla, no interrumpe el run.

---

## Validación de reglas de negocio (diseño propuesto, no implementado)

No existe todavía un módulo dedicado de validación de reglas de negocio. Las
comprobaciones de calidad actuales viven en los pipelines de reporting
(`report_data_quality`, `report_stock_prices`, `report_normalization`,
`report_matching`), que emiten métricas y discrepancias en sus informes.

Como línea de trabajo futura, se contempla un conjunto de *assertions* explícitas
sobre invariantes esperados, por ejemplo:

| Regla | Dataset | Condición |
|-------|---------|-----------|
| Sin tickers duplicados | `SECTOR_*`, `ENRICHED_*` | Ticker único por cap+sector |
| Cobertura temporal mínima | `currencies_*` | `period_end - period_start >= 5 años` |
| Códigos de divisa válidos | `currencies_*` | Solo ISO 4217 (3 letras mayúsculas) |
| Precios positivos | precios | `close > 0` |
| Sin gaps > 5 días hábiles | precios | No hay huecos mayores a 5 sesiones |
| Moneda consistente | `ENRICHED_*` vs precios | Moneda FS = moneda yfinance |
| Muestra representativa | `SAMPLE_*` | n >= fórmula de muestreo estadístico |

---

## Datasets materializados (fuentes disponibles para reporting)

| Dataset | Tipo | Ruta | Disponible para reporting |
|---------|------|------|--------------------------|
| `SECTOR_{cap}_{canonical}` | Polars Parquet | `02_intermediate/02_1. StockUniverse/` | Sí |
| `SAMPLE_{cap}_{canonical}` | Polars Parquet | `02_intermediate/02_2. StratifiedSampleSelection/` | Sí |
| `currencies_{canonical}` | Polars Parquet | `02_intermediate/currencies/` | Sí |
| `ENRICHED_{cap}_{canonical}` | Polars Parquet | `02_intermediate/enriched/` | Sí |
| Parquets FS (por hoja) | Polars Parquet | `02_intermediate/02_3_0. RawFinancialStatements_Parquet/{cap}/{sector}/` | Sí |
| `parse_fs_stats` | MemoryDataset | — | No (necesita persistirse para reporting) |
| `sample_report` | TextDataset | `06_reporting/stratified_sampling/` | Sí |
| `enrichment_matching_report` | TextDataset | `06_reporting/enrich_sectors/` | Sí |
| `prediction_table` (validación) | Polars Parquet | `05_model_outputs/predictions/{window}/{seg}/{horizon}/` | Sí |
| `prediction_table_test` (test) | Polars Parquet | `05_model_outputs/predictions/{window}/{seg}/{horizon}/` | Sí |
| `portfolio_backtest_results` | JSON | `06_reporting/portfolio/{window}/{seg}/{horizon}/` | Sí |
| `portfolio_report` | TextDataset | `06_reporting/portfolio/{window}/{seg}/{horizon}/` | Sí |

**Regla**: para que un pipeline de `_5_reporting/` pueda consumir un dataset, este debe tener filepath en el catálogo. Si es MemoryDataset, hay que darle filepath antes de poder usarlo en reporting.

---

## Estructura de carpetas de output

```
data/06_reporting/
├── stratified_sampling/     # Informe de muestra (Markdown) + PDF (via CLI)
├── currencies/              # Resumen de divisas (JSON)
├── enrich_sectors/          # Matching report (Markdown)
├── data_quality/            # Cobertura y nulls de estados financieros (JSON)
├── stock_prices/            # Validación de precios descargados (JSON)
├── portfolio/               # Rendimiento de carteras
└── observability/           # Logs de ejecución (JSON lines)
```
