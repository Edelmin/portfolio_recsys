"""Atribucion de importancia de variables de la red recurrente.

La etapa de prediccion es el unico componente opaco del sistema (una red
recurrente). Para explicarla se emplea Integrated Gradients (IG), una tecnica de
atribucion post-hoc que reparte la prediccion entre las variables de entrada
integrando el gradiente a lo largo de un camino desde una linea base hasta la
entrada real. IG respeta la estructura temporal: se obtiene una atribucion por
cada (paso temporal, variable) de la ventana de entrada, lo que permite ver
tanto qué variables pesan mas como qué sesiones de la ventana influyen mas.

Este modulo carga un checkpoint de una sola semilla (el ``best_model.pt`` del
HPO) y calcula las atribuciones sobre muestras concretas del conjunto de test.
Es de SOLO LECTURA: no reentrena ni modifica artefactos.

Requiere ``captum`` (dependencia declarada en el proyecto).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from portfolio_recsys.models.architectures import RecurrentRegressor
from portfolio_recsys.models.checkpointing import load_checkpoint_file
from portfolio_recsys.models.config import RecurrentModelConfig


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> tuple[RecurrentRegressor, dict[str, Any]]:
    """Carga y reconstruye el modelo desde un checkpoint de una semilla.

    Returns:
        Tupla (modelo en modo eval, checkpoint completo). El checkpoint incluye
        ``feature_columns``, ``input_size`` y ``preprocessor``.
    """
    device = device or torch.device("cpu")
    checkpoint = load_checkpoint_file(checkpoint_path, device)
    model_config = RecurrentModelConfig.from_dict(checkpoint["model_config"])
    model = RecurrentRegressor(
        input_size=int(checkpoint["input_size"]),
        config=model_config,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def integrated_gradients_for_window(
    model: RecurrentRegressor,
    window: np.ndarray,
    device: torch.device | None = None,
    n_steps: int = 64,
) -> np.ndarray:
    """Calcula las atribuciones de IG para una ventana de entrada.

    Args:
        model: Modelo recurrente en modo eval.
        window: Ventana de entrada escalada, shape ``[sequence_length, n_features]``.
        device: Dispositivo de computo.
        n_steps: Numero de pasos de la integral de Riemann de IG.

    Returns:
        Matriz de atribuciones de la misma forma que la ventana
        (``[sequence_length, n_features]``). El signo indica la direccion del
        efecto sobre el retorno predicho; la magnitud, su importancia.

    La linea base es el vector cero en el espacio escalado. Por el escalado
    robusto (mediana + IQR), el cero corresponde a la mediana de cada variable,
    interpretable como "empresa promedio".
    """
    from captum.attr import IntegratedGradients

    device = device or torch.device("cpu")
    x = torch.tensor(window, dtype=torch.float32, device=device).unsqueeze(0)
    x.requires_grad_(True)
    baseline = torch.zeros_like(x)

    ig = IntegratedGradients(model)
    attributions = ig.attribute(x, baselines=baseline, n_steps=n_steps)
    return attributions.squeeze(0).detach().cpu().numpy()


def global_feature_importance(
    model: RecurrentRegressor,
    windows: np.ndarray,
    feature_columns: list[str],
    device: torch.device | None = None,
    n_steps: int = 32,
) -> list[tuple[str, float]]:
    """Importancia global de variables promediando |IG| sobre varias muestras.

    Args:
        model: Modelo recurrente en modo eval.
        windows: Tensor de ventanas ``[n_muestras, sequence_length, n_features]``.
        feature_columns: Nombres de las variables.
        device: Dispositivo de computo.
        n_steps: Pasos de IG (menor que en el caso individual, por coste).

    Returns:
        Lista de (feature, importancia_media_absoluta) ordenada descendentemente.
        La importancia se promedia sobre muestras y pasos temporales usando el
        valor absoluto, de modo que refleje la relevancia con independencia del
        signo.
    """
    from captum.attr import IntegratedGradients

    device = device or torch.device("cpu")
    ig = IntegratedGradients(model)

    accumulated = np.zeros(len(feature_columns), dtype=np.float64)
    n = windows.shape[0]
    for i in range(n):
        x = torch.tensor(windows[i], dtype=torch.float32, device=device).unsqueeze(0)
        x.requires_grad_(True)
        baseline = torch.zeros_like(x)
        attr = ig.attribute(x, baselines=baseline, n_steps=n_steps)
        attr = attr.squeeze(0).detach().cpu().numpy()
        accumulated += np.abs(attr).sum(axis=0)

    accumulated /= max(n, 1)
    pairs = list(zip(feature_columns, accumulated.tolist()))
    pairs.sort(key=lambda kv: kv[1], reverse=True)
    return pairs
