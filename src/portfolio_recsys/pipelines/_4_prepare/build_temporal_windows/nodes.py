"""Nodos del pipeline build_temporal_windows.

Construye ventanas temporales deslizantes sobre los datasets preprocesados
para el entrenamiento de redes neuronales recurrentes (RNN/GRU/LSTM).

Cada ventana es una secuencia de SEQUENCE_LENGTH sesiones consecutivas
de trading. Se descarta cualquier ventana que cruce un gap excesivo
(suspension, datos faltantes).

La salida son indices Polars (ligeros, persistibles) y arrays NumPy
agrupados por ticker (almacenados temporalmente en memoria para los
DataLoaders de PyTorch).
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

from portfolio_recsys.models.config import WindowConfig
from portfolio_recsys.models.windows import build_temporal_window_store

logger = logging.getLogger(__name__)


def build_windows_from_prepared(
    prepared_df: pl.DataFrame,
    preprocessor: dict[str, Any],
    window_config: dict[str, Any],
) -> dict[str, Any]:
    """Construye el almacen de ventanas temporales para un dataset preprocesado.

    Args:
        prepared_df: DataFrame preprocesado (market o enriched) con features
            escaladas, split, sample_eligible, etc.
        preprocessor: Diccionario del preprocesador (contiene model_feature_columns).
        window_config: Parametros de WindowConfig como dict YAML.

    Returns:
        Diccionario window_store con feature_arrays, target_arrays,
        window_index y metadata.
    """
    win_config = WindowConfig.from_dict(window_config)
    feature_columns = preprocessor["model_feature_columns"]

    window_store = build_temporal_window_store(
        dataframe=prepared_df,
        feature_columns=feature_columns,
        target_scaled_column=preprocessor["target_scaled_column"],
        target_original_column="target_log_return",
        window_config=win_config,
    )

    window_index = window_store["window_index"]

    # Log resumen por split
    split_summary = (
        window_index.group_by("split")
        .agg(
            pl.len().alias("windows"),
            pl.col("ticker").n_unique().alias("tickers"),
        )
        .sort("split")
    )

    for row in split_summary.iter_rows(named=True):
        logger.info(
            "Split %s: %d ventanas, %d tickers",
            row["split"], row["windows"], row["tickers"],
        )

    return window_store


def extract_window_index(window_store: dict[str, Any]) -> pl.DataFrame:
    """Extrae el indice de ventanas del window_store para persistencia.

    Args:
        window_store: Diccionario generado por build_windows_from_prepared.

    Returns:
        DataFrame con metadata de cada ventana (sin arrays NumPy).
    """
    return window_store["window_index"]


def materialize_windows(window_store: dict[str, Any]) -> dict[str, Any]:
    """Materializa los tensores densos por split para persistir en disco.

    Devuelve el dataset de entrada DIRECTA al modelo (tensores ``[N, seq, feat]``
    por split, target, sample_ids y orden de features), listo para guardarse
    como ``.npz`` mediante ``NpzDataset``.
    """
    from portfolio_recsys.models.windows import materialize_persistable_windows

    payload = materialize_persistable_windows(window_store)
    n_by_split = {
        k.replace("_X", ""): int(v.shape[0])
        for k, v in payload.items() if k.endswith("_X")
    }
    logger.info(
        "Ventanas materializadas para persistir: %s (n_features=%d, seq_len=%d)",
        n_by_split, payload["n_features"], payload["sequence_length"],
    )
    return payload
