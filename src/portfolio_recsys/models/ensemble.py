"""Ensemble de semillas.

Construye predicciones ensemble como la media de las predicciones de cada
semilla individual (y su desviacion estandar entre semillas). El
entrenamiento multi-semilla lo realizan los CLIs de entrenamiento
(pr-train-hpo / pr-predict-test), que luego pasan aqui los resultados.
"""

from __future__ import annotations

from typing import Any

import polars as pl


def build_seed_ensemble_table(
    seed_results: list[dict[str, Any]],
    seed_predictions: list[pl.DataFrame],
) -> tuple[pl.DataFrame, list[str]]:
    """Construye predicciones ensemble como media de las semillas individuales.

    Args:
        seed_results: Lista de diccionarios de metricas por semilla.
        seed_predictions: Lista de DataFrames de predicciones por semilla.

    Returns:
        Tupla (DataFrame con prediccion ensemble, lista de columnas por semilla).
    """
    if len(seed_predictions) == 0:
        raise ValueError("Se necesita al menos una semilla.")

    # Usar la primera como base
    base = seed_predictions[0].select([
        "sample_id",
        "signal_date",
        "ticker",
        "trade_entry_date",
        "trade_exit_date",
        "split",
        "target_log_return",
    ])

    # Adjuntar predicciones de cada semilla
    seed_prediction_columns: list[str] = []
    for i, (result, predictions) in enumerate(zip(seed_results, seed_predictions)):
        seed = result["seed"]
        col_name = f"predicted_seed_{seed}"
        seed_prediction_columns.append(col_name)

        seed_col = predictions.select([
            "sample_id",
            pl.col("predicted_log_return").alias(col_name),
        ])

        base = base.join(seed_col, on="sample_id", how="left")

    # Calcular ensemble (media)
    ensemble_expr = (
        pl.mean_horizontal(*[pl.col(c) for c in seed_prediction_columns])
    )

    base = base.with_columns(
        ensemble_expr.alias("predicted_log_return"),
    )

    # Desviacion estandar horizontal (no existe pl.std_horizontal)
    n_seeds = len(seed_prediction_columns)
    mean_expr = pl.mean_horizontal(*[pl.col(c) for c in seed_prediction_columns])
    variance_expr = pl.sum_horizontal(
        *[(pl.col(c) - mean_expr).pow(2) for c in seed_prediction_columns]
    ) / n_seeds
    std_expr = variance_expr.sqrt()

    base = base.with_columns(
        std_expr.alias("prediction_across_seeds_std"),
    )

    return base, seed_prediction_columns
