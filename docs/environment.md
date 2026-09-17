# Entorno, configuración y ejecución

Este documento reúne todo lo relativo al **entorno de trabajo, la configuración y la
ejecución** del proyecto Portfolio Recsys. La arquitectura del sistema (capas de datos,
fases lógicas, pipelines y decisiones de diseño) se documenta en `docs/architecture.md`.

---

## 1. Configuración

```
conf/
├── base/           ← Configuracion del proyecto (versionada)
│   ├── catalog/    ← Definiciones de datasets (factories)
│   ├── globals.yml ← Nombres de subcarpetas/etapas
│   └── parameters.yaml
└── local/          ← Overrides de maquina (no versionados)
    └── globals.yml ← Rutas base al filesystem
```

**Separación**: `base/globals.yml` tiene la semántica del proyecto (nombres de etapas).
`local/globals.yml` tiene las rutas concretas al disco. Kedro mergea ambos.

---

## 2. Ejecución

```bash
# Entorno completo (uv instala Python 3.12 y todas las dependencias desde uv.lock)
uv sync --extra dev

# Comprobar que torch detecta la GPU (torch se instala con CUDA 12.8: torch==2.7.1+cu128)
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Pipeline completo (post-fijacion, DAG estatico)
uv run kedro run

# Pipeline individual
uv run kedro run --pipeline parse_financials_from_html
uv run kedro run --pipeline stratified_random_sampling
uv run kedro run --pipeline fetch_exchange_rates

# Visualizar DAG
uv run kedro viz
```

---

## 3. Reproducibilidad

La combinación de los siguientes elementos garantiza que cualquier resultado puede
reproducirse:

- `pyproject.toml` + `uv.lock` (dependencias exactas)
- Kedro catalog (trazabilidad de datos)
- Semillas fijas en muestreo y entrenamiento
- Conventional commits + release-please (versionado)
- Muestra fijada y persistida tras el ciclo de calibración

---

## 4. Parámetros de calidad de precios (`price_quality`)

El control de calidad de las series de precios (descrito a nivel de arquitectura en
`docs/architecture.md`, se parametriza en `conf/base/parameters.yaml` bajo
la clave `price_quality`:

| Parámetro | Defecto | Efecto |
|-----------|---------|--------|
| `drop_nonpositive_prices` | `true` | Descarta activos con cotización nula/corrupta o penny stocks (Medida 2) |
| `min_price_eur` | `0.01` | Precio mínimo válido en EUR |
| `min_median_price_eur` | `1.0` | Mediana mínima; por debajo = penny stock |
| `clip_returns` | `true` | Activa el recorte de log-retornos (Medida 3) |
| `max_abs_log_return` | `0.40` | Límite diario de log-retorno (se escala por horizonte) |
| `filter_anomalous_windows` | `true` | Activa la cuarentena de ventanas anómalas (Medida 4) |
| `anomaly_tukey_k` | `5.0` | Factor `k` de las vallas de Tukey sectoriales |

> Nota de operación: cambiar cualquiera de estos parámetros invalida los modelos y
> backtests aguas abajo. Como Kedro no reejecuta un nodo cuyo output ya existe, hay que
> forzar la regeneración (`--force`) desde el punto afectado: `fetch_stock_prices`
> (Medida 1) o `normalize_currencies` / `compute_model_features` (Medidas 2 y 3), y
> reentrenar y regenerar predicciones y backtests.
