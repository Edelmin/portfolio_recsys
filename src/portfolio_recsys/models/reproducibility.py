"""Utilidades de reproducibilidad.

Funciones para fijar semillas aleatorias y garantizar determinismo
en entrenamientos de modelos de deep learning.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_random_seed(seed: int = 42) -> None:
    """Fija las semillas aleatorias de Python, NumPy y PyTorch.

    Garantiza reproducibilidad en entrenamiento y evaluacion de modelos.
    Debe invocarse antes de crear el modelo y los DataLoaders.

    Args:
        seed: Valor entero para la semilla. Por defecto 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
