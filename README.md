# Portfolio Recommendation System

Sistema de recomendación para la construcción de carteras de inversión,
desarrollado con **Kedro**. Combina análisis financiero, redes recurrentes
(RNN/GRU/LSTM) para la predicción de retornos y optimización de carteras de
Markowitz, evaluado mediante *backtesting* out-of-sample.

---

## Setup del entorno (con uv)

Este proyecto utiliza **uv** como gestor de entorno y dependencias. `uv` se
encarga de instalar el **Python 3.12** requerido y de crear el entorno virtual
(`.venv/`) a partir del `uv.lock`, de forma reproducible.

```bash
# 1. Instalar uv (ver https://docs.astral.sh/uv/getting-started/installation/)
#    Windows (PowerShell):
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
#    macOS / Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clonar el repositorio
git clone git@github.com:Edelmin/portfolio_recsys.git
cd portfolio-recsys

# 3. Crear el entorno e instalar TODAS las dependencias (incluye dev: pytest, ruff, matplotlib, jupyter)
uv sync --extra dev

# 4. Verificar que el entorno está sano (no debe proponer cambios)
uv sync --extra dev --dry-run        # -> "Would make no changes"

# 5. (Opcional) Registrar el kernel de Jupyter del proyecto
uv run pr-kernel
```

Con esto el repositorio queda listo para ejecutar cualquier CLI o pipeline.

### GPU vs CPU (PyTorch)

El entorno instala **PyTorch con CUDA 12.8** (`torch==2.7.1+cu128`), fijado en el
`uv.lock` desde el índice oficial de PyTorch. Esto es necesario para entrenar en
GPUs NVIDIA modernas (p. ej. RTX 40/50). Comprueba que la GPU se detecta:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Esperado en máquina con GPU CUDA 12.8+:  2.7.1+cu128 True
```

- **Con GPU NVIDIA (CUDA 12.8+)**: funciona sin cambios; el entrenamiento usa la GPU.
- **Sin GPU / solo CPU**: el entrenamiento (HPO, semillas) es inviable en la
  práctica, pero **toda la fase de datos y el backtest sí funcionan** (no usan
  torch en tiempo de ejecución). Si necesitas instalar torch en CPU, deberás
  ajustar la fuente `[tool.uv.sources]` de `pyproject.toml` al índice `cpu` de
  PyTorch y regenerar el lock (`uv lock`).

### Datos de entrada (no incluidos en el repo)

Los datos crudos (`data/`) no se versionan. Para arrancar desde la Fase A hay que
disponer de dos conjuntos de ficheros descargados manualmente del portal TIKR
y colocarlos en `data/01_raw/` con esta nomenclatura (los precios y tipos de
cambio, en cambio, se descargan solos vía Yahoo Finance en la Fase C):

**1. Universo de empresas (HTML), uno por capitalización y sector:**

```
data/01_raw/01.1 - HtmlSectorFiles/{Cap}_{Sector}.txt
```

- `{Cap}`: `LargeCaps` o `SmallCaps`.
- `{Sector}` (nombre en español, sin tildes): `BienesRaices`, `ConsumoDiscrecional`,
  `Energia`, `Financiero`, `Industria`, `MateriasPrimas`,
  `ProductosDePrimeraNecesidad`, `Salud`, `Servicios`, `ServiciosDeComunicacion`,
  `TecnologiasDeLaInformacion`.
- Cada `.txt` contiene el HTML de la tabla de empresas de ese sector/cap. Ejemplo:
  `LargeCaps_Financiero.txt`.

**2. Estados financieros (Excel), agrupados en una carpeta por sector:**

```
data/01_raw/01.2 - FinancialStatementHistorical/{Sector}/*.xlsx
```

- Un `.xlsx` por empresa, con el nombre:
  `{moneda} - {ticker} - Financials ({inicio} - {fin}).xlsx`
  (ejemplo: `USD - BRK.A - Financials (31.12.05 - 31.12.24).xlsx`).
- La carpeta de sector usa el nombre en español **con** tildes donde corresponda
  (ej. `Energía`, `BienesRaíces`, `ServiciosDeComunicación`); el resto sin tildes.

Los nombres de fichero y carpeta son los que esperan los catálogos de
`conf/base/catalog/01_raw/`, así que conviene respetarlos tal cual. Sin estos
ficheros el flujo no puede empezar desde la Fase A.

> **Windows / PowerShell**: al pasar listas separadas por comas a un CLI
> (`--horizons "1d,1w,2m"`, `--windows "w20,w60"`), entrecomilla el valor.

---

## Estructura del proyecto

```
src/portfolio_recsys/     # Código principal (pipelines Kedro + librería de modelos)
conf/                     # Configuración (parameters.yaml, catalog, globals)
data/                     # Capas de datos (01_raw → 06_reporting)
docs/                     # Documentación de arquitectura
reports/                  # Memoria TFM
```

Documentación ampliada en `docs/architecture.md`.

---

## Conceptos clave del flujo de modelado

La capa de modelado se organiza en **tres ejes de experimentación**:

- **window**: tamaño de la ventana temporal que ve la RNN (`w20`, `w60`, ...).
- **segmentation**: `unified` (un modelo global con el sector como *feature*) o
  `by_sector` (un modelo por sector).
- **horizon**: horizonte de rebalanceo/predicción (`1d`, `1w`, `2w`, `1m`, `2m`,
  `3m`, `6m`; `1y` se prepara pero no se entrena por defecto).

Los pipelines y datasets de modelado siguen la convención
`…__{segmentation}__{window}__{horizon}` (ver `docs/architecture.md`).

---

## Flujo de ejecución de principio a fin

El flujo va desde la obtención de datos hasta los modelos entrenados y las
carteras evaluadas. Las fases A–E se ejecutan una sola vez; a partir de F se
trabaja por combinación de ejes.

### Fase A — Ingesta base (una vez)

```bash
uv run kedro run --pipeline parse_financials_from_html
```

### Fase B — Calibración de la muestra (ciclo iterativo)

```bash
uv run kedro run --pipeline stratified_random_sampling
uv run pr-report-pdf                 # PDF de la muestra para revisión manual
uv run kedro run --pipeline parse_financial_statements
uv run kedro run --pipeline clean_financial_statements
```

### Fase C — Ingesta derivada

```bash
uv run kedro run --pipeline extract_currencies_from_filenames
uv run kedro run --pipeline enrich_sectors_with_currencies
uv run kedro run --pipeline fetch_exchange_rates
uv run kedro run --pipeline fetch_stock_prices
```

### Fase D — Transformación (normalización a EUR)

```bash
uv run kedro run --pipeline normalize_currencies
```

### Fase E — Consolidación

```bash
uv run kedro run --pipeline build_company_dataset
```

### Fase F — Preparación de datasets de modelado (multi-eje)

Genera features preprocesadas (`*_prepared`) y ventanas temporales (`*_window`)
por combinación de ejes. Sustituye a los antiguos pipelines `prepare_{horizon}`.

```bash
# Todo (ambas segmentaciones, todas las ventanas y horizontes)
uv run pr-prepare-datasets

# Solo unified, ventana w20, subconjunto de horizontes
uv run pr-prepare-datasets --segmentation unified --windows w20 --horizons "1d,1w,1m"

# Forzar regeneración (desactiva la reanudabilidad)
uv run pr-prepare-datasets --force
```

### Fase G — Entrenamiento con optimización de hiperparámetros

Optimización bayesiana (Optuna/TPE) por combinación de ejes; un estudio
independiente por conjunto de datos (market y enriched). Escribe los modelos en
`data/04_models/{window}/{segmentation}/{horizon}/hpo/`.

```bash
# Solo la fase de HPO (asume datasets ya preparados)
uv run pr-train-hpo --segmentation unified --windows w20 --horizons "1d,1w,1m"

# CLI MAESTRO: prepara datasets Y entrena con HPO en un solo comando
uv run pr-run-all --segmentation unified --windows w20
```

### Fase H — Predicciones de test, carteras y evaluación

```bash
# 1. Predicciones sobre TEST reentrenando la config ganadora con 5 semillas (ensemble)
uv run pr-predict-test --window w20 --segmentation unified

# 2. Backtest Markowitz por conjunto (market y enriched) y horizonte
uv run pr-backtest-all --window w20 --segmentation unified --dataset-type both --horizons "1d,1w,2w,1m,2m,3m,6m"

# 3. CSV plano de métricas (fuente de las tablas de la memoria)
uv run pr-backtest-csv --window w20 --segmentation unified
```

### Fase I — Explicabilidad y figuras de la memoria

```bash
# Explicabilidad: trazabilidad + descomposición de Markowitz + Integrated Gradients
uv run pr-explainability

# Figuras para la memoria (PDF en reports/Memoria/images/)
uv run pr-portfolio-figures --dataset-type market --window w20 --segmentation unified
uv run pr-portfolio-figures --dataset-type enriched --window w20 --segmentation unified
uv run pr-earlystopping-figures --window w20 --segmentation unified
uv run pr-explainability-figures
```

### Reporting de datos (ejecutar según necesidad)

```bash
uv run kedro run --pipeline report_currencies
uv run kedro run --pipeline report_exchange_rates
uv run kedro run --pipeline report_stock_prices
uv run kedro run --pipeline report_normalization
uv run kedro run --pipeline report_data_quality
```

---

## Referencia de CLIs

Todos se invocan con `uv run <cli> [opciones]`. Con el entorno sincronizado
(`uv sync --extra dev`), `uv run` no reinstala nada en cada invocación.

### Entorno y utilidades

| CLI | Para qué sirve |
|-----|----------------|
| `pr-kernel` | Registra el kernel de Jupyter del entorno del proyecto. |
| `pr-report-pdf` | Convierte el informe de muestra estratificada (Markdown) a PDF (requiere Node.js/npx). |

### Preparación y entrenamiento (flujo actual, multi-eje)

| CLI | Para qué sirve |
|-----|----------------|
| `pr-prepare-datasets` | Genera `*_prepared` y `*_window` por (window, segmentation, horizon). Reanudable; `--force` regenera. |
| `pr-train-hpo` | Optimización de hiperparámetros (Optuna/TPE) por eje; un estudio por dataset. Escribe en `data/04_models/.../hpo/`. |
| `pr-run-all` | CLI maestro: encadena `pr-prepare-datasets` + `pr-train-hpo` con los mismos ejes. `--only prepare|train`. |
| `pr-predict-test` | Reentrena la config ganadora con 5 semillas y genera `prediction_table_test_{market,enriched}` sobre TEST. |

### Carteras, evaluación y figuras

| CLI | Para qué sirve |
|-----|----------------|
| `pr-backtest-all` | Backtest Markowitz por (window, segmentation, horizon). `--dataset-type {market,enriched,both}`. |
| `pr-backtest-csv` | Vuelca las métricas de los backtests a `data/06_reporting/portfolio/{window}/{seg}/backtest_metrics.csv`. |
| `pr-portfolio-figures` | Figuras de carteras (equity, Sharpe, estrategias) para la memoria. |
| `pr-backtest-aggregate-figures` | Figuras de análisis agregado de backtests (heatmaps de diferencial vs S&P 500, boxplots, marginales) en Sharpe/Sortino. |
| `pr-earlystopping-figures` | Curvas de aprendizaje (validación arriba, entrenamiento abajo) por horizonte y conjunto. |
| `pr-explainability` | Genera trazabilidad, descomposición de Markowitz e Integrated Gradients por combinación. |
| `pr-explainability-figures` | Figuras de explicabilidad para la memoria a partir de los artefactos anteriores. |
| `pr-price-eval` | Reconstruye precios EUR predichos vs reales y calcula métricas por horizonte. |

---

## Outputs (layout multi-eje)

```
data/03_processed/model_features/{window}/{segmentation}/{horizon}/   # *_prepared, *_window
data/04_models/{window}/{segmentation}/{horizon}/hpo/                 # estudios Optuna + best_model.pt
data/04_models/{window}/{segmentation}/{horizon}/test_seeds/          # checkpoints y history por semilla
data/05_model_outputs/predictions/{window}/{segmentation}/{horizon}/  # prediction_table_test
data/06_reporting/portfolio/{window}/{segmentation}/{horizon}/        # backtest_results + informe
data/06_reporting/explainability/{window}/{segmentation}/{horizon}/   # trazabilidad, IG
```

---

## Comandos habituales

```bash
uv run kedro run --pipeline X       # Ejecutar un pipeline concreto
uv run kedro viz                    # Visualizar el DAG
uv run pytest                       # Tests
uv run ruff check .                 # Lint
```

---

## Documentación adicional

- `docs/architecture.md` — Arquitectura general y fases del proyecto
- `docs/portfolio-markowitz.md` — Pipeline de carteras Markowitz (detalle)
- `docs/reporting-domains.md` — Dominios de reporting y observabilidad
- `docs/environment.md` — Configuración del entorno

