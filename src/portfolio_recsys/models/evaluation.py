"""Metricas de evaluacion de predicciones financieras.

Implementa las metricas clave para evaluar modelos de prediccion
de retornos en el contexto de recomendacion de carteras:

- Metricas de regresion: MAE, RMSE, correlacion, directional accuracy.
- Information Coefficient (IC): correlacion de Spearman diaria entre
  ranking predicho y ranking realizado.
- Quantile spread: retorno medio del cuantil superior vs inferior
  (long-short spread).
- Evaluacion integrada: combina todas las metricas en un solo paso.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl


def calculate_regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calcula metricas estandar de regresion.

    Args:
        actual: Array con valores reales (log-returns).
        predicted: Array con predicciones (log-returns).

    Returns:
        Diccionario con: number_of_samples, mae_log_return, rmse_log_return,
        directional_accuracy, pearson_correlation, prediction_mean/std,
        actual_mean/std.

    Raises:
        ValueError: Si las formas no coinciden o no hay observaciones validas.
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    if actual.shape != predicted.shape:
        raise ValueError("actual y predicted deben tener la misma forma.")

    valid_mask = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid_mask]
    predicted = predicted[valid_mask]

    if len(actual) == 0:
        raise ValueError("No hay observaciones validas.")

    errors = predicted - actual
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(np.square(errors)))

    if np.std(actual) > 0 and np.std(predicted) > 0:
        correlation = np.corrcoef(predicted, actual)[0, 1]
    else:
        correlation = np.nan

    directional_accuracy = np.mean(
        (predicted > 0) == (actual > 0)
    )

    return {
        "number_of_samples": int(len(actual)),
        "mae_log_return": float(mae),
        "rmse_log_return": float(rmse),
        "directional_accuracy": float(directional_accuracy),
        "pearson_correlation": float(correlation),
        "prediction_mean": float(np.mean(predicted)),
        "prediction_std": float(np.std(predicted)),
        "actual_mean": float(np.mean(actual)),
        "actual_std": float(np.std(actual)),
    }


def calculate_daily_ic(
    prediction_table: pl.DataFrame,
    minimum_assets_per_date: int = 5,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Calcula el Information Coefficient (Spearman) diario.

    Para cada fecha de senal, calcula la correlacion de rango entre
    las predicciones ordenadas y los retornos realizados ordenados.

    Args:
        prediction_table: DataFrame con columnas signal_date,
            predicted_log_return, target_log_return.
        minimum_assets_per_date: Minimo de activos por fecha para
            que el IC sea significativo.

    Returns:
        Tupla (DataFrame diario con IC, diccionario resumen).
    """
    ranked_predictions = prediction_table.with_columns(
        pl.col("predicted_log_return")
        .rank(method="average")
        .over("signal_date")
        .alias("predicted_rank"),
        pl.col("target_log_return")
        .rank(method="average")
        .over("signal_date")
        .alias("realized_rank"),
        pl.len().over("signal_date").alias("number_of_assets"),
    )

    daily_ic = (
        ranked_predictions.group_by("signal_date")
        .agg(
            pl.col("number_of_assets").first().alias("number_of_assets"),
            pl.corr("predicted_rank", "realized_rank").alias("spearman_ic"),
        )
        .filter(pl.col("number_of_assets") >= minimum_assets_per_date)
        .filter(pl.col("spearman_ic").is_not_null() & pl.col("spearman_ic").is_finite())
        .sort("signal_date")
    )

    if daily_ic.height == 0:
        return daily_ic, {
            "number_of_ic_dates": 0,
            "mean_ic": None,
            "median_ic": None,
            "std_ic": None,
            "positive_ic_fraction": None,
            "ic_information_ratio": None,
        }

    mean_ic = daily_ic.select(pl.col("spearman_ic").mean()).item()
    median_ic = daily_ic.select(pl.col("spearman_ic").median()).item()
    std_ic = daily_ic.select(pl.col("spearman_ic").std()).item()
    positive_fraction = daily_ic.select((pl.col("spearman_ic") > 0).mean()).item()

    if std_ic is not None and math.isfinite(float(std_ic)) and float(std_ic) > 0:
        ic_information_ratio = float(mean_ic) / float(std_ic)
    else:
        ic_information_ratio = None

    summary = {
        "number_of_ic_dates": int(daily_ic.height),
        "mean_ic": float(mean_ic),
        "median_ic": float(median_ic),
        "std_ic": float(std_ic) if std_ic is not None else None,
        "positive_ic_fraction": float(positive_fraction),
        "ic_information_ratio": ic_information_ratio,
    }

    return daily_ic, summary


def calculate_quantile_spread(
    prediction_table: pl.DataFrame,
    quantile_fraction: float = 0.20,
    minimum_assets_per_date: int = 5,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Calcula el spread de retornos entre cuantiles extremos (long-short).

    Divide las predicciones en cuantil superior (top) e inferior (bottom)
    y calcula la diferencia de retornos medios realizados.

    Args:
        prediction_table: DataFrame con signal_date, predicted_log_return,
            target_log_return.
        quantile_fraction: Fraccion del total en cada extremo (default 20%).
        minimum_assets_per_date: Activos minimos por fecha.

    Returns:
        Tupla (DataFrame diario con spreads, diccionario resumen).
    """
    if not 0 < quantile_fraction < 0.5:
        raise ValueError("quantile_fraction debe estar entre 0 y 0.5.")

    ranked = (
        prediction_table.with_columns(
            pl.col("predicted_log_return")
            .rank(method="average")
            .over("signal_date")
            .alias("__predicted_rank"),
            pl.len().over("signal_date").alias("__number_of_assets"),
        )
        .with_columns(
            ((pl.col("__predicted_rank") - 0.5) / pl.col("__number_of_assets")).alias(
                "__predicted_percentile"
            )
        )
        .filter(pl.col("__number_of_assets") >= minimum_assets_per_date)
    )

    daily_spreads = (
        ranked.group_by("signal_date")
        .agg(
            pl.col("target_log_return")
            .filter(pl.col("__predicted_percentile") >= 1.0 - quantile_fraction)
            .mean()
            .alias("top_return"),
            pl.col("target_log_return")
            .filter(pl.col("__predicted_percentile") <= quantile_fraction)
            .mean()
            .alias("bottom_return"),
            pl.len().alias("number_of_assets"),
            (pl.col("__predicted_percentile") >= 1.0 - quantile_fraction)
            .sum()
            .alias("top_positions"),
            (pl.col("__predicted_percentile") <= quantile_fraction)
            .sum()
            .alias("bottom_positions"),
        )
        .filter(pl.col("top_return").is_not_null() & pl.col("bottom_return").is_not_null())
        .with_columns(
            (pl.col("top_return") - pl.col("bottom_return")).alias("long_short_spread")
        )
        .sort("signal_date")
    )

    if daily_spreads.height == 0:
        return daily_spreads, {
            "number_of_spread_dates": 0,
            "mean_top_return": None,
            "mean_bottom_return": None,
            "mean_long_short_spread": None,
            "spread_std": None,
            "positive_spread_fraction": None,
        }

    summary_row = daily_spreads.select(
        pl.len().alias("number_of_spread_dates"),
        pl.col("top_return").mean().alias("mean_top_return"),
        pl.col("bottom_return").mean().alias("mean_bottom_return"),
        pl.col("long_short_spread").mean().alias("mean_long_short_spread"),
        pl.col("long_short_spread").std().alias("spread_std"),
        (pl.col("long_short_spread") > 0).mean().alias("positive_spread_fraction"),
    ).row(0, named=True)

    summary = {
        key: (
            int(value) if key == "number_of_spread_dates"
            else (float(value) if value is not None else None)
        )
        for key, value in summary_row.items()
    }

    return daily_spreads, summary


def evaluate_validation_predictions(
    prediction_table: pl.DataFrame,
    minimum_assets_per_date: int = 5,
    quantile_fraction: float = 0.20,
) -> tuple[dict[str, Any], pl.DataFrame, pl.DataFrame]:
    """Evaluacion integrada de predicciones de validacion.

    Combina metricas de regresion, IC diario y quantile spread en un
    unico punto de entrada.

    Args:
        prediction_table: DataFrame con signal_date, ticker,
            predicted_log_return, target_log_return.
        minimum_assets_per_date: Activos minimos por fecha.
        quantile_fraction: Fraccion del cuantil extremo.

    Returns:
        Tupla (metricas_dict, daily_ic_df, daily_spreads_df).
    """
    actual = prediction_table.get_column("target_log_return").to_numpy().astype(np.float64)
    predicted = (
        prediction_table.get_column("predicted_log_return").to_numpy().astype(np.float64)
    )

    regression_metrics = calculate_regression_metrics(actual=actual, predicted=predicted)

    daily_ic, ic_summary = calculate_daily_ic(
        prediction_table=prediction_table,
        minimum_assets_per_date=minimum_assets_per_date,
    )

    daily_spreads, spread_summary = calculate_quantile_spread(
        prediction_table=prediction_table,
        quantile_fraction=quantile_fraction,
        minimum_assets_per_date=minimum_assets_per_date,
    )

    metrics = {**regression_metrics, **ic_summary, **spread_summary}

    actual_std = metrics["actual_std"]
    prediction_std = metrics["prediction_std"]
    if actual_std > 0:
        metrics["dispersion_ratio"] = prediction_std / actual_std
    else:
        metrics["dispersion_ratio"] = None

    return metrics, daily_ic, daily_spreads
