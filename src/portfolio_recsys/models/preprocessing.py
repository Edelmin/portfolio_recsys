"""Preprocesador robusto para features financieras.

Implementa un escalado robusto basado en mediana y rango intercuartilico (IQR)
que es resistente a outliers, tipicos en datos financieros. Incluye:

- Sanitizacion de Inf/NaN → null.
- Estimacion de estadisticos robustos (mediana, IQR) solo sobre train.
- Escalado: (x - mediana) / IQR.
- Clipping a [-clip_value, +clip_value] para limitar extremos.
- Imputacion de nulls con 0 (valor neutral post-escalado).
- Escalado del target con mediana + IQR propio.
- Variables binarias: se mantienen sin escalar.
- Deteccion de variables constantes o con IQR=0 (fallback a constante 0).

El preprocesador se serializa como un dict[str, Any] para persistencia JSON.
"""

from __future__ import annotations

from typing import Any

import polars as pl


def sanitize_numeric_columns(
    dataframe: pl.DataFrame,
    columns: list[str],
) -> pl.DataFrame:
    """Reemplaza Inf y valores no finitos por null en columnas numericas.

    Args:
        dataframe: DataFrame de entrada.
        columns: Columnas a sanitizar.

    Returns:
        DataFrame con valores no finitos convertidos a null.
    """
    expressions = []
    for column_name in columns:
        numeric_expression = pl.col(column_name).cast(pl.Float64, strict=False)
        expressions.append(
            pl.when(
                numeric_expression.is_not_null() & numeric_expression.is_finite()
            )
            .then(numeric_expression)
            .otherwise(None)
            .alias(column_name)
        )
    return dataframe.with_columns(expressions)


def fit_robust_preprocessor(
    train_dataframe: pl.DataFrame,
    feature_columns: list[str],
    target_column: str,
    binary_feature_columns: list[str] | None = None,
    clip_value: float | None = 10.0,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Ajusta el preprocesador robusto sobre datos de entrenamiento.

    Calcula mediana e IQR para cada feature continua y para el target.
    Las features con IQR = 0 o sin datos se descartan (constant_fallback).

    Args:
        train_dataframe: DataFrame exclusivamente con filas de entrenamiento.
        feature_columns: Lista de nombres de features (continuas + binarias).
        target_column: Nombre de la columna objetivo.
        binary_feature_columns: Features que son binarias (0/1), no se escalan.
        clip_value: Limite de clipping post-escalado. None para desactivar.
        epsilon: Valor minimo de IQR para evitar division por cero.

    Returns:
        Diccionario con estadisticos, mapeo de columnas y metadata.

    Raises:
        ValueError: Si faltan columnas o el DataFrame esta vacio.
    """
    binary_feature_columns = binary_feature_columns or []

    unknown_binary = set(binary_feature_columns) - set(feature_columns)
    if unknown_binary:
        raise ValueError(
            f"Columnas binarias no estan en feature_columns: {unknown_binary}"
        )

    required_columns = set(feature_columns) | {target_column}
    missing_columns = required_columns - set(train_dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"Faltan columnas para ajustar el preprocesador: {sorted(missing_columns)}"
        )

    if train_dataframe.height == 0:
        raise ValueError("El conjunto de entrenamiento esta vacio.")

    if "split" in train_dataframe.columns:
        observed_splits = (
            train_dataframe.get_column("split").drop_nulls().unique().to_list()
        )
        if observed_splits != ["train"]:
            raise ValueError(
                "El preprocesador debe ajustarse solamente con filas de entrenamiento."
            )

    # Sanitizar
    numeric_columns = list(dict.fromkeys(feature_columns + [target_column]))
    clean_train = sanitize_numeric_columns(
        dataframe=train_dataframe, columns=numeric_columns
    )

    continuous_feature_columns = [
        c for c in feature_columns if c not in binary_feature_columns
    ]

    feature_statistics: dict[str, dict[str, float | str]] = {}
    usable_continuous_columns: list[str] = []
    dropped_feature_columns: list[str] = []

    for column_name in continuous_feature_columns:
        non_null_count = clean_train.select(pl.col(column_name).count()).item()

        if non_null_count == 0:
            dropped_feature_columns.append(column_name)
            continue

        median = clean_train.select(pl.col(column_name).median()).item()
        q1 = clean_train.select(
            pl.col(column_name).quantile(0.25, interpolation="linear")
        ).item()
        q3 = clean_train.select(
            pl.col(column_name).quantile(0.75, interpolation="linear")
        ).item()

        iqr = float(q3) - float(q1) if q1 is not None and q3 is not None else 0.0

        if iqr < epsilon:
            # Variable constante o casi constante
            feature_statistics[column_name] = {
                "median": float(median) if median is not None else 0.0,
                "iqr": 0.0,
                "scale_method": "constant_fallback",
            }
            dropped_feature_columns.append(column_name)
        else:
            feature_statistics[column_name] = {
                "median": float(median),
                "iqr": float(iqr),
                "scale_method": "robust_iqr",
            }
            usable_continuous_columns.append(column_name)

    # Estadisticos del target
    target_non_null = clean_train.select(pl.col(target_column).count()).item()
    if target_non_null == 0:
        raise ValueError(f"El target '{target_column}' no tiene valores validos.")

    target_median = float(clean_train.select(pl.col(target_column).median()).item())
    target_q1 = float(
        clean_train.select(
            pl.col(target_column).quantile(0.25, interpolation="linear")
        ).item()
    )
    target_q3 = float(
        clean_train.select(
            pl.col(target_column).quantile(0.75, interpolation="linear")
        ).item()
    )
    target_iqr = target_q3 - target_q1

    if target_iqr < epsilon:
        target_iqr = float(
            clean_train.select(pl.col(target_column).std()).item() or 1.0
        )

    # Columnas finales del modelo
    model_feature_columns = usable_continuous_columns + binary_feature_columns

    return {
        "feature_statistics": feature_statistics,
        "target": {
            "column": target_column,
            "mean": target_median,
            "iqr": target_iqr,
        },
        "model_feature_columns": model_feature_columns,
        "continuous_feature_columns": usable_continuous_columns,
        "binary_feature_columns": binary_feature_columns,
        "dropped_feature_columns": dropped_feature_columns,
        "clip_value": clip_value,
        "epsilon": epsilon,
        "target_scaled_column": "target_scaled",
    }


def transform_with_preprocessor(
    dataframe: pl.DataFrame,
    preprocessor: dict[str, Any],
) -> pl.DataFrame:
    """Aplica el preprocesador ajustado a un DataFrame.

    Operaciones:
    1. Sanitiza valores no finitos → null.
    2. Escala features continuas: (x - median) / iqr.
    3. Clip a [-clip_value, +clip_value].
    4. Imputa nulls con 0.
    5. Escala el target: (y - median) / iqr.

    Args:
        dataframe: DataFrame con todas las columnas requeridas.
        preprocessor: Diccionario devuelto por fit_robust_preprocessor.

    Returns:
        DataFrame con features y target escalados.
    """
    continuous_columns = preprocessor["continuous_feature_columns"]
    binary_columns = preprocessor["binary_feature_columns"]
    feature_stats = preprocessor["feature_statistics"]
    target_info = preprocessor["target"]
    clip_value = preprocessor["clip_value"]
    target_column = target_info["column"]

    # Sanitizar
    all_numeric = continuous_columns + [target_column]
    df = sanitize_numeric_columns(dataframe=dataframe, columns=all_numeric)

    # Escalar features continuas
    scale_expressions = []
    for col_name in continuous_columns:
        stats = feature_stats[col_name]
        median = stats["median"]
        iqr = stats["iqr"]

        scaled = (pl.col(col_name) - median) / iqr

        if clip_value is not None:
            scaled = scaled.clip(-clip_value, clip_value)

        # Imputar null → 0
        scaled = pl.when(scaled.is_not_null() & scaled.is_finite()).then(scaled).otherwise(0.0)

        scale_expressions.append(scaled.alias(col_name))

    if scale_expressions:
        df = df.with_columns(scale_expressions)

    # Binarias: null → 0
    if binary_columns:
        binary_expressions = [
            pl.col(c).fill_null(0).cast(pl.Float64).alias(c) for c in binary_columns
        ]
        df = df.with_columns(binary_expressions)

    # Escalar target
    target_median = target_info["mean"]
    target_iqr = target_info["iqr"]
    target_scaled_column = preprocessor["target_scaled_column"]

    df = df.with_columns(
        pl.when(pl.col(target_column).is_not_null() & pl.col(target_column).is_finite())
        .then((pl.col(target_column) - target_median) / target_iqr)
        .otherwise(None)
        .alias(target_scaled_column)
    )

    return df


def inverse_transform_target(
    scaled_values,
    preprocessor: dict[str, Any],
):
    """Deshace el escalado del target para recuperar log-returns originales.

    Args:
        scaled_values: Array NumPy o escalar con valores escalados.
        preprocessor: Diccionario del preprocesador.

    Returns:
        Valores en la escala original (log-return).
    """
    import numpy as np

    target_median = float(preprocessor["target"]["mean"])
    target_iqr = float(preprocessor["target"]["iqr"])

    return np.asarray(scaled_values, dtype=np.float64) * target_iqr + target_median
