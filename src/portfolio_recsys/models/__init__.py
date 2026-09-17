"""Modulo de modelos de aprendizaje profundo.

Contiene las arquitecturas recurrentes (RNN, GRU, LSTM), configuraciones,
logica de entrenamiento, evaluacion y utilidades de prediccion para el
sistema de recomendacion de carteras.

Submodulos:
    config      — Dataclasses de configuracion (modelo, entrenamiento, features)
    architectures — Clases nn.Module (RecurrentRegressor)
    training    — Bucle de entrenamiento y evaluacion por epoca
    checkpointing — Persistencia y restauracion de checkpoints
    evaluation  — Metricas de prediccion (IC, spread, regresion)
    prediction  — Inferencia y construccion de tablas de prediccion
    ensemble    — Ensemble de semillas y comparacion de datasets
    preprocessing — Preprocesador robusto (mediana + MAD/IQR, clip)
    windows     — Ventanas temporales, Dataset PyTorch, DataLoaders
    reproducibility — Semillas y determinismo
"""

from portfolio_recsys.models.config import (
    RecurrentModelConfig,
    TrainingConfig,
    FeatureEngineeringConfig,
    TemporalSplitConfig,
    WindowConfig,
)
from portfolio_recsys.models.architectures import RecurrentRegressor
from portfolio_recsys.models.reproducibility import set_random_seed

__all__ = [
    "RecurrentModelConfig",
    "TrainingConfig",
    "FeatureEngineeringConfig",
    "TemporalSplitConfig",
    "WindowConfig",
    "RecurrentRegressor",
    "set_random_seed",
]
