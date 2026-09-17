# Pipeline de Carteras Markowitz

Este documento describe el pipeline `report_portfolio`, que implementa la
construcción y evaluación de carteras de inversión mediante optimización
de Markowitz con backtesting sobre la muestra de validación.

---

## 1. Objetivo

Evaluar la calidad de las predicciones del modelo recurrente como señal
para la construcción de carteras. Se comparan distintas configuraciones
(número de activos, tipo de restricciones) contra benchmarks de referencia.

---

## 2. Horizontes de predicción

El sistema entrena un modelo independiente por cada horizonte de holding.
Cada modelo predice el log-retorno acumulado a su horizonte.

| Horizonte | Sesiones | Target del modelo |
|-----------|----------|-------------------|
| 1 día     | 1        | `log(close[t+2] / close[t+1])` |
| 1 semana  | 5        | `log(close[t+6] / close[t+1])` |
| 2 semanas | 10       | `log(close[t+11] / close[t+1])` |
| 1 mes     | 21       | `log(close[t+22] / close[t+1])` |
| 2 meses   | 42       | `log(close[t+43] / close[t+1])` |
| 3 meses   | 63       | `log(close[t+64] / close[t+1])` |
| 6 meses   | 126      | `log(close[t+127] / close[t+1])` |

El periodo de rebalanceo de la cartera coincide con el horizonte de holding
del modelo: al vencer cada predicción, se rebalancea.

---

## 3. Split temporal

Se usa un split único para todos los horizontes, dimensionado para soportar
hasta el horizonte de 1 año:

| Split      | Rango de fechas         | Uso |
|------------|------------------------|-----|
| Train      | inicio — 2019-12-31    | Ajuste de pesos de los modelos |
| Validation | 2020-01-01 — 2021-12-31 | Selección de arquitectura/hiperparámetros y backtest exploratorio |
| Test       | 2022-01-01 en adelante  | Backtest de evaluación out-of-sample |

---

## 3.1. Los dos backtests: validación vs test

El backtest de carteras puede ejecutarse sobre dos periodos, con objetivos
distintos. El periodo se controla con el parámetro
`report_portfolio.evaluation_split` (por defecto `test`).

### Backtest de validación (`evaluation_split: validation`)

- **Periodo**: 2020-2021.
- **Objetivo**: backtest **exploratorio**, durante el desarrollo. Es el periodo
  que se usa también para seleccionar la arquitectura (Bi-LSTM/GRU), el early
  stopping y los hiperparámetros.
- **Sesgo**: optimista. El modelo se ha *elegido* mirando este periodo, por lo
  que su rendimiento aquí está sesgado al alza (sesgo de selección). Además
  2020-2021 fue una fase marcadamente alcista tras el COVID.
- **Uso**: análisis y depuración, NO como evidencia final de rendimiento.

### Backtest de test (`evaluation_split: test`) — por defecto

- **Periodo**: 2022 en adelante (incluye el mercado bajista de 2022).
- **Objetivo**: evaluación **imparcial** del rendimiento de las carteras. 
- **Uso**: es la evidencia de rendimiento que se reporta en la memoria.

### Cómo se generan las predicciones de cada split

Las predicciones de test las genera `pr-predict-test`, que reutiliza 
los checkpoints ya entrenados (no reentrena) y ejecuta inferencia sobre el 
split test, guardando `prediction_table_test.parquet`. El pipeline `report_portfolio` 
consume por defecto la tabla de test.

```bash
# 1. Generar predicciones sobre test (reutiliza checkpoints, no reentrena)
uv run pr-predict-test                      # todos los horizontes
uv run pr-predict-test --horizons 1d,3m

# 2. Backtest de carteras sobre test (por defecto) y sus informes
uv run pr-backtest-all

# Backtest exploratorio sobre validacion (override del parametro):
uv run kedro run --pipeline report_portfolio_3m \\
  --params "report_portfolio.evaluation_split=validation"
```

---

## 4. Formulación matemática

El backtest evalúa cuatro estrategias de optimización distintas, todas sobre
la misma selección top-K y en escala de periodo de holding. Se configuran en
`markowitz_portfolio.strategies`. Cada una se evalúa en variante long-only
(`w>=0`) y long-short (`||w||_1 <= L`).

### 4.1. max_return (enfoque original: topar el riesgo)

```
maximizar    muᵀw
sujeto a     wᵀΣw <= sigma_max²,  sum(w)=1,  0 <= w <= w_max   (long-only)
```

Maximiza retorno con un techo de riesgo. **No diversifica por sí solo**: empuja
hacia los activos de mayor `mu` hasta agotar el presupuesto de riesgo.

### 4.2. min_variance (mínima varianza: bajo riesgo)

```
minimizar    wᵀΣw
sujeto a     sum(w)=1,  0 <= w <= w_max
```

No usa `mu` en la ponderación ni el tope `sigma_max`: reparte el peso para
minimizar el riesgo conjunto, por lo que **diversifica de forma natural**. La
señal del modelo interviene solo en la selección top-K previa. Al no depender de
`sigma_max`, evita la infactibilidad a horizontes largos.

### 4.3. mean_variance (Markowitz canónico con aversión al riesgo)

```
maximizar    muᵀw - lambda·wᵀΣw
sujeto a     sum(w)=1,  0 <= w <= w_max
```

`lambda` (aversión al riesgo) es el dial retorno↔diversificación: bajo => cerca
de max_return; alto => cerca de min_variance. Como `mu` y `Sigma` están en la
misma escala de periodo, un mismo `lambda` es comparable entre horizontes. Se
evalúa un barrido `risk_aversion_lambdas` (default `[1,2,3,5,10]`).

### 4.4. target_return (dual del original)

```
minimizar    wᵀΣw
sujeto a     muᵀw >= r_objetivo,  sum(w)=1,  0 <= w <= w_max
```

Fija un retorno esperado mínimo (`target_annual_return`, convertido a periodo) y
minimiza el riesgo para alcanzarlo. Si es infactible en un rebalanceo, recurre a
`min_variance` como fallback.

Donde en todas las formulaciones:
- `w`: vector de pesos de la cartera
- `mu`: log-retornos esperados del periodo (del modelo)
- `Sigma`: covarianza de log-retornos, escalada al periodo de holding
- `sigma_max`: volatilidad anual objetivo (default 20%), convertida a periodo
- `w_max`: límite de concentración por activo (default 0.30)
- `L`: leverage máximo para long-short (default 1.5)
- `lambda`: aversión al riesgo (solo mean_variance)

La optimización se resuelve con `cvxpy`, usando OSQP como solver principal
y SCS como fallback.

---

## 5. Flujo del backtest

```
Para cada horizonte de predicción:
  │
  ├─ 1. Generar fechas de rebalanceo
  │     - Se toman las signal_dates del split de validación
  │     - Se espacian cada holding_period_sessions fechas
  │
  ├─ 2. En cada fecha de rebalanceo:
  │     │
  │     ├─ 2a. Seleccionar top-K tickers
  │     │      Ordenar por predicted_log_return descendente
  │     │      Tomar los K mejores (K = 5, 10, 15, 20)
  │     │
  │     ├─ 2b. Estimar matriz de covarianza
  │     │      Tomar retornos log-diarios de los K tickers
  │     │      Ventana histórica: últimos 252 días hábiles
  │     │      Regularización: Sigma += I * 1e-8
  │     │
  │     ├─ 2c. Optimizar pesos (Markowitz)
  │     │      Variante long-only: w >= 0
  │     │      Variante long-short: ||w||_1 <= leverage
  │     │      Si infactible: fallback a pesos iguales (1/K)
  │     │
  │     └─ 2d. Calcular retorno realizado
  │            r_periodo = sum(w_i * (P_exit_i / P_entry_i - 1))
  │            Equity *= (1 + r_periodo)
  │
  ├─ 3. Calcular benchmarks
  │     │
  │     ├─ Equiponderado: mismos K del modelo, pesos 1/K
  │     │
  │     ├─ S&P 500 (buy & hold): mantener el índice durante todo el periodo
  │     │
  │     └─ S&P 500 + activo seguro: cartera Markowitz de dos activos
  │                                 {S&P 500, activo seguro}
  │
  └─ 4. Generar métricas e informe
```

---

## 6. Configuraciones evaluadas

Para cada horizonte se evalúan 4 * top_k combinaciones:

| Dimensión | Valores |
|-----------|---------|
| Top-K (número de activos) | 5, 10, 15, 20 |
| Estrategia de optimización | max_return, min_variance, mean_variance (barrido de lambda), target_return |
| Variante | long-only, long-short |
| Benchmarks | equiponderado (por top-K), S&P 500 buy & hold, S&P 500 + activo seguro (Markowitz) |

Cada estrategia de optimización se evalúa por top-K y variante (long-only /
long-short). A ello se añaden, por top-K, la cartera equiponderada, y como
comparadores de mercado (independientes del top-K) el S&P 500 buy & hold y la
cartera Markowitz {S&P 500, activo seguro}.

---

## 7. Métricas de evaluación

| Métrica | Fórmula | Interpretación |
|---------|---------|----------------|
| Retorno total | `equity_final / equity_inicial - 1` | Rendimiento bruto del periodo |
| Retorno anualizado | `(1 + R_total)^(252/N_sesiones) - 1` | Rendimiento comparable entre horizontes |
| Volatilidad anualizada | `std(r_periodo) * sqrt(252/holding)` | Riesgo realizado |
| Sharpe ratio | `R_anualizado / Vol_anualizada` | Retorno ajustado por riesgo |
| Max drawdown | `max((peak - valley) / peak)` | Peor caída desde máximo |

---

## 8. Parámetros del pipeline

Definidos en `conf/base/parameters.yaml` bajo la sección `markowitz_portfolio`:

```yaml
markowitz_portfolio:
  excluded_tickers: [IDEA]             # Tickers apartados del universo ponderable
  top_k_configs: [5, 10, 15, 20]       # Activos por cartera
  max_risk: 0.20                        # Riesgo max del periodo de holding
  covariance_lookback_sessions: 252     # Ventana para estimar Sigma
  allow_short_selling: true             # Generar variante long-short
  max_leverage: 1.5                     # Leverage para long-short
  min_price_history_sessions: 60        # Minimo de datos para covarianza
```

El horizonte activo se selecciona con `report_portfolio.horizon_key` y
`report_portfolio.horizon_config`.

### 8.1. Exclusión manual de tickers (`excluded_tickers`)

Lista de tickers que se apartan del universo ponderable del backtest. Su
predicción se calcula igual y permanece en `prediction_table` (para analizarla por
separado), pero el ticker no entra en la selección top-K ni en la optimización de
Markowitz. El filtro se aplica en `run_markowitz_backtest` justo después de filtrar
por split, de modo que la exclusión afecta por igual a todas las estrategias
—modelo y equiponderada—, preservando un universo común y una comparación
homogénea.

A diferencia del descarte de penny stocks de `price_quality` (Medida 2, que actúa en
`normalize_currencies` y requiere regenerar el dataset), `excluded_tickers` es un
parámetro del backtest: no requiere regenerar datasets ni reentrenar, basta con
relanzar `pr-backtest-all`. Es útil para apartar un activo concreto de forma inmediata
sin rehacer el pipeline.

Valor por defecto: `[IDEA]`. IDEA es un penny stock con saltos de retorno desorbitados
(más del 35 000 % a 1 mes) que, de no excluirse, inflan la cota de predicción perfecta
y contaminan las carteras. Se aparta aquí de forma provisional hasta la regeneración
del pipeline con el filtro de calidad (Medida 2), que lo eliminaría ya en la capa de
precios.

---

## 9. Datasets

### Inputs

| Dataset | Tipo | Ubicación |
|---------|------|-----------|
| `prediction_table` | Polars Parquet | `05_model_outputs/predictions/prediction_table.parquet` |
| `stock_prices_eur` | Polars Parquet | `02_intermediate/normalized_prices/` |
| `split_configuration` | JSON | `03_processed/model_features/split_configuration.json` |

### Outputs

| Dataset | Tipo | Ubicación |
|---------|------|-----------|
| `portfolio_backtest_results` | JSON | `06_reporting/portfolio/backtest_results.json` |
| `portfolio_report` | Markdown | `06_reporting/portfolio/portfolio_report.md` |

---

## 10. Ejecución

```bash
# Horizonte por defecto (1 día):
uv run kedro run --pipeline report_portfolio

# Horizonte específico (3 meses):
uv run kedro run --pipeline report_portfolio \
  --params "report_portfolio.horizon_key=3m,report_portfolio.horizon_config.holding_period_sessions=63,report_portfolio.horizon_config.label=3 meses"

# Horizonte de 1 año:
uv run kedro run --pipeline report_portfolio \
  --params "report_portfolio.horizon_key=1y,report_portfolio.horizon_config.holding_period_sessions=252,report_portfolio.horizon_config.label=1 ano"
```

---

## 11. Prerrequisitos

Antes de ejecutar `report_portfolio`, debe existir el dataset `prediction_table`.
Este se genera durante el entrenamiento del modelo y debe persistirse manualmente:

```python
# En el notebook o script de entrenamiento:
from portfolio_recsys.models.prediction import build_prediction_table

prediction_table = build_prediction_table(
    model=trained_model,
    dataloader=val_dataloader,
    window_index=window_index,
    preprocessor=preprocessor,
    device=device,
)

# Persistir para que el pipeline lo consuma:
prediction_table.write_parquet(
    "data/05_model_outputs/predictions/prediction_table.parquet"
)
```

---

## 12. Arquitectura del código

```
src/portfolio_recsys/pipelines/_5_reporting/report_portfolio/
├── __init__.py       # Docstring del modulo
├── pipeline.py       # Definicion del DAG (2 nodos)
└── nodes.py          # Logica de negocio:
                      #   - run_markowitz_backtest()     [nodo principal]
                      #   - generate_portfolio_report()  [nodo de informe]
                      #   - _solve_markowitz_long_only() [helper cvxpy]
                      #   - _solve_markowitz_long_short()[helper cvxpy]
                      #   - _estimate_covariance()       [helper numpy]
                      #   - _compute_period_return()     [helper]
                      #   - _compute_portfolio_metrics() [helper]
```

---

## 13. Decisiones de diseño

| Decisión | Justificación |
|----------|---------------|
| Riesgo como restricción (no objetivo) | Permite al usuario fijar su tolerancia al riesgo y maximizar retorno dentro de ese límite |
| Long-only + Long-short | Compara el escenario realista (long-only) con el teóricamente óptimo (long-short) |
| Equiponderado como baseline | Baseline mínimo: mismos K del modelo pero sin optimización de pesos (aísla el valor de la ponderación) |
| S&P 500 (buy & hold) como referencia de mercado | ¿Bate la cartera al índice de referencia manteniéndolo pasivamente? |
| S&P 500 + activo seguro (Markowitz) como referencia | Comparador de mercado con gestión de riesgo: cartera de dos activos optimizada |
| Fallback a pesos iguales si infactible | Evita perder periodos de backtest por problemas numéricos |
| Rebalanceo = horizonte de holding | Coherencia: se rebalancea cuando vence la predicción anterior |
