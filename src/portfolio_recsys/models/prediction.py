"""Inferencia y construccion de tablas de prediccion.

Funciones para ejecutar forward pass sobre DataLoaders de test/validacion
y construir DataFrames con predicciones, targets y metadata alineados.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

from portfolio_recsys.models.preprocessing import inverse_transform_target


def predict_scaled_targets(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Genera predicciones escaladas sobre un DataLoader.

    Args:
        model: Modelo en modo eval.
        dataloader: DataLoader (validacion o test).
        device: Dispositivo de computo.

    Returns:
        Tupla (predicted_scaled, actual_scaled, sample_ids) como arrays NumPy.

    Raises:
        RuntimeError: Si no se genera ninguna prediccion.
        FloatingPointError: Si las predicciones contienen NaN/Inf.
    """
    model.eval()

    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_sample_ids: list[np.ndarray] = []

    with torch.inference_mode():
        for batch_x, batch_y, batch_sample_ids in dataloader:
            batch_x = batch_x.to(device, non_blocking=True)
            predictions = model(batch_x)

            all_predictions.append(predictions.detach().cpu().numpy().astype(np.float64))
            all_targets.append(batch_y.detach().cpu().numpy().astype(np.float64))
            all_sample_ids.append(
                batch_sample_ids.detach().cpu().numpy().astype(np.int64)
            )

    if not all_predictions:
        raise RuntimeError("No se genero ninguna prediccion.")

    predicted_scaled = np.concatenate(all_predictions)
    actual_scaled = np.concatenate(all_targets)
    sample_ids = np.concatenate(all_sample_ids)

    if not (len(predicted_scaled) == len(actual_scaled) == len(sample_ids)):
        raise RuntimeError("Predicciones, objetivos e IDs tienen longitudes diferentes.")

    if not np.isfinite(predicted_scaled).all():
        raise FloatingPointError("Las predicciones contienen NaN o infinitos.")

    return predicted_scaled, actual_scaled, sample_ids


def get_sample_metadata(
    window_index: pl.DataFrame,
    sample_ids: list[int] | np.ndarray,
) -> pl.DataFrame:
    """Obtiene metadata alineada con los sample_ids de un DataLoader.

    Mantiene el orden de los sample_ids proporcionados mediante un
    join que preserva la posicion.

    Args:
        window_index: DataFrame de indice de ventanas (con sample_id).
        sample_ids: IDs de las muestras en el orden del DataLoader.

    Returns:
        DataFrame con metadata ordenado segun sample_ids.
    """
    ordered_ids = pl.DataFrame({"sample_id": np.asarray(sample_ids, dtype=np.int64)})

    metadata = ordered_ids.join(
        window_index,
        on="sample_id",
        how="left",
    )

    return metadata


def build_prediction_table(
    model: nn.Module,
    dataloader: DataLoader,
    window_index: pl.DataFrame,
    preprocessor: dict[str, Any],
    device: torch.device,
) -> pl.DataFrame:
    """Construye una tabla completa de predicciones con metadata.

    Ejecuta inferencia, invierte el escalado del target, y une con la
    metadata de ventanas para producir un DataFrame listo para evaluacion.

    Args:
        model: Modelo entrenado.
        dataloader: DataLoader de validacion/test.
        window_index: Indice de ventanas temporales.
        preprocessor: Diccionario del preprocesador.
        device: Dispositivo de computo.

    Returns:
        DataFrame con: signal_date, ticker, trade_entry_date, trade_exit_date,
        split, target_log_return, predicted_log_return, y columnas derivadas.
    """
    predicted_scaled, actual_scaled, sample_ids = predict_scaled_targets(
        model=model, dataloader=dataloader, device=device
    )

    predicted_log_returns = inverse_transform_target(
        scaled_values=predicted_scaled, preprocessor=preprocessor
    )
    reconstructed_actual_returns = inverse_transform_target(
        scaled_values=actual_scaled, preprocessor=preprocessor
    )

    metadata = get_sample_metadata(
        window_index=window_index, sample_ids=sample_ids
    )

    prediction_columns = pl.DataFrame({
        "sample_id": sample_ids,
        "target_scaled_actual": actual_scaled,
        "target_scaled_predicted": predicted_scaled,
        "predicted_log_return": predicted_log_returns,
        "reconstructed_actual_log_return": reconstructed_actual_returns,
    })

    prediction_table = (
        metadata.join(prediction_columns, on="sample_id", how="left")
        .with_columns(
            pl.col("predicted_log_return").exp().sub(1.0).alias("predicted_simple_return"),
            pl.col("target_log_return").exp().sub(1.0).alias("realized_simple_return"),
            (pl.col("predicted_log_return") - pl.col("target_log_return")).alias(
                "prediction_error_log_return"
            ),
            (pl.col("predicted_log_return") > 0).cast(pl.Int8).alias("predicted_positive"),
            (pl.col("target_log_return") > 0).cast(pl.Int8).alias("realized_positive"),
        )
        .sort(["signal_date", "ticker"])
    )

    # Verificacion de consistencia
    maximum_reconstruction_error = (
        prediction_table.select(
            (pl.col("reconstructed_actual_log_return") - pl.col("target_log_return"))
            .abs()
            .max()
        ).item()
    )

    if maximum_reconstruction_error is not None and maximum_reconstruction_error > 1e-5:
        raise RuntimeError(
            "El objetivo reconstruido no coincide con el original. "
            f"Error maximo: {maximum_reconstruction_error}"
        )

    return prediction_table
