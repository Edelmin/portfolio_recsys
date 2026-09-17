"""Bucle de entrenamiento para modelos recurrentes.

Implementa las funciones de una epoca (train/eval), gestion de early
stopping, ReduceLROnPlateau y la funcion principal fit_recurrent_model
que orquesta todo el proceso.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from portfolio_recsys.models.checkpointing import (
    restore_model_from_checkpoint,
    save_training_checkpoint,
)
from portfolio_recsys.models.config import RecurrentModelConfig, TrainingConfig


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_gradient_norm: float,
) -> dict[str, float]:
    """Ejecuta una epoca de entrenamiento.

    Args:
        model: Modelo a entrenar (se pone en mode train).
        dataloader: DataLoader del split de entrenamiento.
        optimizer: Optimizador (AdamW).
        criterion: Funcion de perdida.
        device: Dispositivo (cpu/cuda).
        max_gradient_norm: Norma maxima para gradient clipping.

    Returns:
        Metricas de la epoca: loss, mae_scaled, mean/max_gradient_norm.

    Raises:
        RuntimeError: Si el DataLoader no produce muestras.
        FloatingPointError: Si la loss es NaN/Inf.
    """
    model.train()

    total_loss = 0.0
    total_absolute_error = 0.0
    total_samples = 0
    gradient_norms: list[float] = []

    for batch_x, batch_y, batch_sample_ids in dataloader:
        del batch_sample_ids

        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(batch_x)

        if predictions.shape != batch_y.shape:
            raise RuntimeError(
                f"Forma prediccion {predictions.shape} != objetivo {batch_y.shape}."
            )

        loss = criterion(predictions, batch_y)

        if not torch.isfinite(loss):
            raise FloatingPointError("La perdida de entrenamiento es NaN o infinita.")

        loss.backward()

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(),
            max_norm=max_gradient_norm,
            error_if_nonfinite=True,
        )

        optimizer.step()

        batch_size = batch_x.shape[0]
        total_loss += loss.detach().item() * batch_size
        total_absolute_error += (
            torch.abs(predictions.detach() - batch_y).sum().item()
        )
        total_samples += batch_size
        gradient_norms.append(float(gradient_norm.detach().item()))

    if total_samples == 0:
        raise RuntimeError("El DataLoader de entrenamiento no produjo muestras.")

    return {
        "loss": total_loss / total_samples,
        "mae_scaled": total_absolute_error / total_samples,
        "mean_gradient_norm": float(np.mean(gradient_norms)),
        "max_gradient_norm_observed": float(np.max(gradient_norms)),
    }


def evaluate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evalua el modelo sobre un DataLoader (sin gradientes).

    Args:
        model: Modelo a evaluar (se pone en mode eval).
        dataloader: DataLoader del split de validacion/test.
        criterion: Funcion de perdida.
        device: Dispositivo (cpu/cuda).

    Returns:
        Metricas: loss, mae_scaled, rmse_scaled.
    """
    model.eval()

    total_loss = 0.0
    total_absolute_error = 0.0
    total_squared_error = 0.0
    total_samples = 0

    with torch.inference_mode():
        for batch_x, batch_y, _ in dataloader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)

            batch_size = batch_x.shape[0]
            errors = predictions - batch_y

            total_loss += loss.item() * batch_size
            total_absolute_error += torch.abs(errors).sum().item()
            total_squared_error += (errors**2).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("El DataLoader de validacion no produjo muestras.")

    mse = total_squared_error / total_samples

    return {
        "loss": total_loss / total_samples,
        "mae_scaled": total_absolute_error / total_samples,
        "rmse_scaled": math.sqrt(mse),
    }


def fit_recurrent_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    model_config: RecurrentModelConfig,
    training_config: TrainingConfig,
    feature_columns: list[str],
    preprocessor: dict[str, Any],
    device: torch.device,
    checkpoint_path: str | Path,
    epoch_callback: "Callable[[int, dict[str, float]], None] | None" = None,
) -> tuple[nn.Module, pl.DataFrame, dict[str, Any]]:
    """Entrena un modelo recurrente con early stopping y checkpointing.

    Orquesta el bucle completo: epocas, scheduler, early stopping,
    guardado del mejor checkpoint y restauracion al final.

    Args:
        model: Modelo RecurrentRegressor inicializado.
        train_loader: DataLoader de entrenamiento.
        validation_loader: DataLoader de validacion.
        model_config: Configuracion de la arquitectura.
        training_config: Hiperparametros de entrenamiento.
        feature_columns: Lista de nombres de features del modelo.
        preprocessor: Diccionario del preprocesador (para guardar en checkpoint).
        device: Dispositivo de computo.
        checkpoint_path: Ruta donde guardar el mejor checkpoint.
        epoch_callback: Callback opcional invocado al final de cada epoca con
            (epoch, validation_metrics). Permite reportar metricas intermedias a
            un optimizador de hiperparametros (Optuna) y abortar el trial
            lanzando una excepcion (pruning). Si es None, comportamiento estandar.

    Returns:
        Tupla (modelo restaurado al mejor checkpoint, historial, checkpoint_dict).

    Raises:
        RuntimeError: Si no se guardo ningun checkpoint.
    """
    criterion = nn.SmoothL1Loss(
        beta=training_config.smooth_l1_beta,
        reduction="mean",
    )

    optimizer = AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=training_config.scheduler_factor,
        patience=training_config.scheduler_patience,
        min_lr=training_config.minimum_learning_rate,
    )

    best_validation_loss = math.inf
    epochs_without_improvement = 0
    history_rows: list[dict[str, Any]] = []
    checkpoint_path = Path(checkpoint_path)

    for epoch in range(1, training_config.max_epochs + 1):
        learning_rate_before = float(optimizer.param_groups[0]["lr"])

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_gradient_norm=training_config.max_gradient_norm,
        )

        validation_metrics = evaluate_one_epoch(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=device,
        )

        validation_loss = validation_metrics["loss"]
        scheduler.step(validation_loss)
        learning_rate_after = float(optimizer.param_groups[0]["lr"])

        improvement = best_validation_loss - validation_loss
        has_improved = improvement > training_config.early_stopping_min_delta

        if has_improved:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            save_training_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                validation_loss=validation_loss,
                model_config=model_config,
                training_config=training_config,
                feature_columns=feature_columns,
                preprocessor=preprocessor,
            )
        else:
            epochs_without_improvement += 1

        history_rows.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "validation_loss": validation_loss,
            "train_mae_scaled": train_metrics["mae_scaled"],
            "validation_mae_scaled": validation_metrics["mae_scaled"],
            "validation_rmse_scaled": validation_metrics["rmse_scaled"],
            "mean_gradient_norm": train_metrics["mean_gradient_norm"],
            "max_gradient_norm_observed": train_metrics["max_gradient_norm_observed"],
            "learning_rate_before": learning_rate_before,
            "learning_rate_after": learning_rate_after,
            "improved": has_improved,
            "epochs_without_improvement": epochs_without_improvement,
        })

        # Callback de epoca (p.ej. reporte a Optuna + pruning). Se invoca DESPUES
        # de haber guardado el mejor checkpoint de esta epoca, de modo que si el
        # callback aborta el trial (pruning) ya existe un checkpoint restaurable.
        if epoch_callback is not None:
            epoch_callback(epoch, validation_metrics)

        if epochs_without_improvement >= training_config.early_stopping_patience:
            break

    if not checkpoint_path.exists():
        raise RuntimeError("No se guardo ningun checkpoint.")

    best_checkpoint = restore_model_from_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    history_df = pl.DataFrame(history_rows)

    return model, history_df, best_checkpoint
