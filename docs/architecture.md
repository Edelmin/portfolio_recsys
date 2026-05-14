# 🏗 Architecture

Este documento describe la arquitectura del proyecto **Portfolio Recsys Comparison**, incluyendo:

- Organización del código
- Flujo de datos
- Diseño de pipelines
- Estrategia de modelado
- Evaluación
- Reproducibilidad
- Decisiones de diseño

---

# 1️⃣ Objetivo del sistema

El objetivo del proyecto es:

> Comparar distintos sistemas de recomendación aplicados a la construcción de carteras de inversión.

El sistema permite:

- Extraer y procesar datos financieros
- Generar features relevantes
- Entrenar múltiples modelos de recomendación
- Evaluar su rendimiento bajo métricas técnicas y financieras
- Versionar resultados de manera reproducible

---

# 2️⃣ Visión general del flujo de datos

El flujo lógico del sistema es el siguiente:

```
Raw Data (HTML / fuentes externas)
        ↓
Data Engineering (Parsing + Limpieza)
        ↓
Feature Engineering
        ↓
Model Training
        ↓
Recommendation Generation
        ↓
Evaluation & Reporting
```

---

# 3️⃣ Estructura del repositorio

```
portfolio-recsys-comparison/
│
├── conf/
│   ├── base/
│   ├── local/
│   └── logging.yml
│
├── data/
│   ├── 01_raw/
│   ├── 02_intermediate/
│   ├── 03_primary/
│   ├── 04_feature/
│   ├── 05_model_input/
│   ├── 06_models/
│   ├── 07_model_output/
│   └── 08_reporting/
│
├── src/
│   └── portfolio_recsys/
│       ├── pipelines/
│       ├── tools/
│       ├── hooks.py
│       └── settings.py
│
├── docs/
│   ├── development.md
│   └── architecture.md
│
├── pyproject.toml
├── uv.lock
└── README.md
```

---

# 4️⃣ Organización de pipelines (Kedro)

Los pipelines están organizados por responsabilidad:

## 📦 data_engineering
- Parsing de HTML
- Normalización de datos
- Limpieza y tipado
- Control de encoding

## 📦 feature_engineering
- Construcción de variables financieras
- Normalización
- Enriquecimiento con metadata (sector, tamaño, etc.)

## 📦 modeling
- Entrenamiento de modelos
- Serialización en `data/06_models`

## 📦 inference
- Generación de recomendaciones Top-K
- Exportación en `07_model_output`

## 📦 evaluation
- Cálculo de métricas técnicas
- Métricas financieras (Sharpe, retorno acumulado, etc.)
- Comparación entre modelos

---

# 5️⃣ Contratos de datos

Cada etapa produce artefactos con estructura definida:

### Identificadores estándar
- `asset_id`
- `date`
- `sector`
- `market_cap`
- `price`
- `return`

### Reglas
- Fechas en formato datetime (sin timezone)
- Decimales consistentes
- Encoding UTF-8 normalizado
- No espacios no separables (`U+202F`, `NBSP`)

---

# 6️⃣ Modelos de recomendación

El proyecto contempla múltiples familias:

## 🔹 Baselines
- Popularidad
- Reglas heurísticas

## 🔹 Collaborative Filtering
- Matrix Factorization (ALS)
- kNN

## 🔹 Content-Based
- Similitud por características financieras
- Embeddings

## 🔹 Híbridos
- Combinación ponderada
- Learning-to-rank (si aplica)

Todos los modelos deben producir una salida estandarizada:

```
user_id | asset_id | score | rank
```

---

# 7️⃣ Protocolo de evaluación

Se utiliza evaluación offline con:

## Métricas técnicas
- Precision@K
- Recall@K
- MAP@K
- NDCG@K

## Métricas financieras
- Retorno esperado
- Sharpe ratio
- Volatilidad
- Max drawdown
- Diversificación
- Turnover

## Separación temporal
- Split train/test basado en tiempo
- Prevención de data leakage

---

# 8️⃣ Gestión de dependencias y entorno

El proyecto utiliza:

- `uv` como gestor de entorno
- `uv.lock` para reproducibilidad exacta
- Python recomendado: 3.11

Comandos clave:

```
uv sync --extra dev
uv run kedro run
```

No se utiliza conda.

---

# 9️⃣ Versionado y releases

Se utiliza:

- Semantic Versioning (MAJOR.MINOR.PATCH)
- Conventional Commits
- Release Please para automatización

Al hacer merge a `main`:

- Se actualiza `CHANGELOG.md`
- Se genera nueva versión
- Se crea tag `vX.Y.Z`

No se crean tags manualmente.

---

# 🔟 Logging y observabilidad

Configuración en:

```
conf/logging.yml
```

Los logs se almacenan en:

```
logs/
```

No se versionan.

---

# 11️⃣ Decisiones de diseño

## 🔹 Uso de Kedro
- Modularidad
- Separación clara de responsabilidades
- Data catalog explícito
- Reproducibilidad

## 🔹 Uso de uv
- Entornos ligeros
- Lockfile reproducible
- Instalación rápida

## 🔹 Separación stricta
- Notebooks solo exploración
- Código productivo en `src/`
- Nodes sin acceso directo al filesystem

---

# 12️⃣ Reproducibilidad

Para reconstruir completamente el entorno:

```
uv sync --extra dev
uv run kedro run
```

La combinación de:

- `pyproject.toml`
- `uv.lock`
- Versionado semántico
- CHANGELOG automático

garantiza trazabilidad completa del proyecto.

---

# 🎯 Principios del sistema

- Modular
- Reproducible
- Versionado automáticamente
- Separación clara entre desarrollo y ejecución
- Orientado a benchmarking comparativo