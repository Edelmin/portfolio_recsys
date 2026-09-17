"""Trazabilidad de la cadena de decision de inversion.

Reconstruye, para todas las configuraciones del modelo y todos sus rebalanceos,
el recorrido de cada activo a lo largo de las etapas interpretables por diseno
del sistema:

    retorno predicho -> posicion en la ordenacion -> inclusion en el Top-K
    -> peso asignado por la optimizacion de Markowitz.

Trabaja exclusivamente sobre los artefactos ya generados por el backtesting
(``backtest_results_{market,enriched}.json``), sin recalcular nada. Es la
materializacion del discurso de "explicabilidad como trazabilidad": las etapas
de seleccion y asignacion son transparentes y su resultado puede reconstruirse
enlazando los datos que el propio sistema registra en cada rebalanceo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl


def load_backtest_results(path: str | Path) -> dict[str, Any]:
    """Carga un fichero de resultados de backtesting (JSON)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el fichero de backtesting: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _find_rebalance_record(
    config: dict[str, Any], rebalance_date: str
) -> dict[str, Any]:
    """Localiza el detalle de un rebalanceo por fecha dentro de una config."""
    details = config.get("rebalance_details", [])
    for record in details:
        if str(record.get("date")) == str(rebalance_date):
            return record
    raise ValueError(
        f"No se encontro el rebalanceo con fecha '{rebalance_date}'. "
        f"Fechas disponibles (muestra): {[str(r.get('date')) for r in details[:5]]}..."
    )


# Configuraciones que corresponden a carteras del modelo (con seleccion Top-K y
# pesos). Se excluyen los benchmarks (S&P 500), el oracle (prediccion perfecta,
# descartado) y la equiponderada (no registra detalle por rebalanceo).
def model_config_names(results: dict[str, Any]) -> list[str]:
    """Devuelve las configs del modelo aptas para trazabilidad/descomposicion.

    Filtra las que tienen ``rebalance_details`` con tickers y pesos, excluyendo
    ``oracle``, ``equal_weight`` y los ``benchmark_*``.
    """
    names = []
    for name, cfg in results.get("configs", {}).items():
        if name.startswith("benchmark_") or "oracle" in name or "equal_weight" in name:
            continue
        details = cfg.get("rebalance_details", [])
        if details and details[0].get("tickers"):
            names.append(name)
    return names


def build_full_traceability(results: dict[str, Any]) -> pl.DataFrame:
    """Trazabilidad exhaustiva de una combinacion: todas las configs del modelo
    y todos sus rebalanceos, apilados en un unico DataFrame.

    Columns: config, rebalance_date, next_date, optimization_status, rank,
    ticker, predicted_return, weight, weight_pct.
    """
    frames: list[pl.DataFrame] = []
    for config_name in model_config_names(results):
        cfg = results["configs"][config_name]
        for record in cfg.get("rebalance_details", []):
            tickers = list(record.get("tickers", []))
            weights = list(record.get("weights", []))
            predicted = list(record.get("predicted_returns", []))
            if not tickers:
                continue
            rows = [
                {
                    "ticker": t,
                    "predicted_return": float(p),
                    "weight": float(w),
                }
                for t, w, p in zip(tickers, weights, predicted)
            ]
            frame = (
                pl.DataFrame(rows)
                .sort("predicted_return", descending=True)
                .with_row_index("rank", offset=1)
                .with_columns(
                    pl.lit(config_name).alias("config"),
                    pl.lit(str(record.get("date"))).alias("rebalance_date"),
                    pl.lit(str(record.get("next_date"))).alias("next_date"),
                    pl.lit(str(record.get("optimization_status"))).alias(
                        "optimization_status"
                    ),
                    (pl.col("weight") * 100.0).alias("weight_pct"),
                )
            )
            frames.append(frame)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed").select([
        "config", "rebalance_date", "next_date", "optimization_status",
        "rank", "ticker", "predicted_return", "weight", "weight_pct",
    ])
