"""Persistencia y restauracion de checkpoints de entrenamiento.

Cada checkpoint almacena: pesos del modelo, estado del optimizador,
estado del scheduler, epoca, metricas, configuracion y metadata.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from portfolio_recsys.models.config import RecurrentModelConfig, TrainingConfig


def save_training_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    validation_loss: float,
    model_config: RecurrentModelConfig,
    training_config: TrainingConfig,
    feature_columns: list[str],
    preprocessor: dict[str, Any],
) -> None:
    """Guarda un checkpoint completo del estado de entrenamiento.

    Args:
        checkpoint_path: Ruta del fichero .pt a guardar.
        model: Modelo con los mejores pesos.
        optimizer: Estado del optimizador.
        scheduler: Estado del scheduler.
        epoch: Epoca actual.
        validation_loss: Loss de validacion en esta epoca.
        model_config: Configuracion de la arquitectura.
        training_config: Configuracion de entrenamiento.
        feature_columns: Columnas de features usadas.
        preprocessor: Diccionario del preprocesador.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "validation_loss": validation_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "feature_columns": feature_columns,
        "input_size": len(feature_columns),
        "preprocessor": preprocessor,
    }

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint_file(
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """Carga un fichero de checkpoint desde disco.

    Args:
        checkpoint_path: Ruta del fichero .pt.
        device: Dispositivo donde mapear los tensores.

    Returns:
        Diccionario con todo el contenido del checkpoint.

    Raises:
        FileNotFoundError: Si el fichero no existe.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")

    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def restore_model_from_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """Restaura los pesos del modelo desde un checkpoint.

    Carga el state_dict del modelo y lo mueve al dispositivo indicado.

    Args:
        model: Modelo con la misma arquitectura que el guardado.
        checkpoint_path: Ruta del fichero .pt.
        device: Dispositivo destino.

    Returns:
        Diccionario completo del checkpoint (incluye metadata).
    """
    checkpoint = load_checkpoint_file(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return checkpoint
