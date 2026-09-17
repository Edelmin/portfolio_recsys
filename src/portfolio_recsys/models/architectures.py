"""Arquitecturas de redes neuronales recurrentes.

Define la clase RecurrentRegressor, un modelo generico que soporta
RNN vanilla, GRU y LSTM (unidireccional y bidireccional), con una
cabeza de regresion configurable para prediccion de retornos logaritmicos.

Cabezas disponibles (head_activation en RecurrentModelConfig):
    - 'relu': Dropout → Linear → ReLU → Dropout → Linear
    - 'silu': Dropout → Linear → SiLU → Dropout → Linear
    - 'leaky_relu': Dropout → Linear → LeakyReLU → Dropout → Linear
                    (con inicializacion Kaiming)
    - 'gelu': Dropout → Linear → GELU → Dropout → Linear
    - 'linear': Dropout → Linear (cabeza directa sin activacion oculta)

Flujo del modelo:
    Input [batch, sequence_length, features]
        → Capa(s) recurrente(s)
        → Extraccion del estado oculto final
        → Layer Normalization
        → Cabeza de regresion (configurable)
    Output [batch]
"""

from __future__ import annotations

import torch
from torch import nn

from portfolio_recsys.models.config import RecurrentModelConfig


# ─────────────────────────────────────────────────────────────────────────────
# Inicializaciones especializadas
# ─────────────────────────────────────────────────────────────────────────────


def initialize_leaky_relu_head(module: nn.Module) -> None:
    """Inicializacion Kaiming para capas lineales con LeakyReLU.

    Aplica kaiming_uniform_ con a=0.05 (negative_slope del LeakyReLU)
    para que la varianza de las activaciones se preserve correctamente.

    Args:
        module: Modulo nn.Module (se aplica solo a nn.Linear).
    """
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(
            module.weight,
            a=0.05,
            mode="fan_in",
            nonlinearity="leaky_relu",
        )
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ─────────────────────────────────────────────────────────────────────────────
# Construccion de la cabeza de regresion
# ─────────────────────────────────────────────────────────────────────────────


def _build_regression_head(
    input_size: int,
    config: RecurrentModelConfig,
) -> nn.Module:
    """Construye la cabeza de regresion segun head_activation.

    Args:
        input_size: Dimensiones de la representacion de entrada.
        config: Configuracion del modelo con head_activation,
                head_hidden_size, head_dropout.

    Returns:
        nn.Module (Sequential o Linear) que mapea [batch, input_size] → [batch, 1].
    """
    activation = config.head_activation

    if activation == "linear":
        # Cabeza directa: sin capa oculta ni activacion
        head = nn.Sequential(
            nn.Dropout(p=config.head_dropout),
            nn.Linear(in_features=input_size, out_features=1),
        )
        return head

    # Cabezas con capa oculta + activacion
    activation_layer: nn.Module
    if activation == "relu":
        activation_layer = nn.ReLU()
    elif activation == "silu":
        activation_layer = nn.SiLU()
    elif activation == "leaky_relu":
        activation_layer = nn.LeakyReLU(negative_slope=0.05)
    elif activation == "gelu":
        activation_layer = nn.GELU()
    else:
        raise ValueError(
            f"head_activation no reconocida: '{activation}'. "
            "Opciones: 'relu', 'silu', 'leaky_relu', 'gelu', 'linear'."
        )

    head = nn.Sequential(
        nn.Dropout(p=config.head_dropout),
        nn.Linear(
            in_features=input_size,
            out_features=config.head_hidden_size,
        ),
        activation_layer,
        nn.Dropout(p=config.head_dropout),
        nn.Linear(
            in_features=config.head_hidden_size,
            out_features=1,
        ),
    )

    # Inicializacion Kaiming para LeakyReLU
    if activation == "leaky_relu":
        head.apply(initialize_leaky_relu_head)

    return head


# ─────────────────────────────────────────────────────────────────────────────
# Modelo principal
# ─────────────────────────────────────────────────────────────────────────────


class RecurrentRegressor(nn.Module):
    """Modelo recurrente generico para regresion de secuencias temporales.

    Recibe una secuencia temporal de features financieras y produce una
    prediccion escalar (retorno logaritmico esperado).

    Entrada: [batch_size, sequence_length, input_size]
    Salida:  [batch_size]

    La cabeza de regresion es configurable via config.head_activation:
    - 'relu': MLP con ReLU (default, compatible con version anterior).
    - 'silu': MLP con SiLU (Swish), smooth y no monotona.
    - 'leaky_relu': MLP con LeakyReLU + inicializacion Kaiming.
    - 'gelu': MLP con GELU, popular en transformers.
    - 'linear': Capa lineal directa (sin hidden layer).

    Args:
        input_size: Numero de features de entrada por timestep.
        config: RecurrentModelConfig con la topologia de la red.
    """

    def __init__(
        self,
        input_size: int,
        config: RecurrentModelConfig,
    ) -> None:
        super().__init__()

        if input_size <= 0:
            raise ValueError("input_size debe ser positivo.")
        if config.hidden_size <= 0:
            raise ValueError("hidden_size debe ser positivo.")
        if config.num_layers <= 0:
            raise ValueError("num_layers debe ser positivo.")
        if not (0.0 <= config.recurrent_dropout < 1.0):
            raise ValueError(
                "recurrent_dropout debe estar entre 0 y 1."
            )
        if not (0.0 <= config.head_dropout < 1.0):
            raise ValueError(
                "head_dropout debe estar entre 0 y 1."
            )

        self.input_size = input_size
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.bidirectional = config.bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        # Dropout recurrente solo aplica con mas de una capa
        effective_recurrent_dropout = (
            config.recurrent_dropout if config.num_layers > 1 else 0.0
        )

        common_recurrent_arguments = {
            "input_size": input_size,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "batch_first": True,
            "dropout": effective_recurrent_dropout,
            "bidirectional": config.bidirectional,
        }

        if config.recurrent_type == "rnn":
            self.recurrent = nn.RNN(
                **common_recurrent_arguments,
                nonlinearity=config.rnn_nonlinearity,
            )
        elif config.recurrent_type == "gru":
            self.recurrent = nn.GRU(**common_recurrent_arguments)
        elif config.recurrent_type == "lstm":
            self.recurrent = nn.LSTM(**common_recurrent_arguments)
        else:
            raise ValueError(
                f"Tipo recurrente no reconocido: {config.recurrent_type}"
            )

        recurrent_representation_size = (
            config.hidden_size * self.num_directions
        )
        self.representation_size = recurrent_representation_size

        self.representation_normalization = nn.LayerNorm(
            recurrent_representation_size
        )

        # Cabeza de regresion configurable
        self.regression_head = _build_regression_head(
            input_size=recurrent_representation_size,
            config=config,
        )

    def extract_final_representation(
        self,
        hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        """Extrae la representacion final del estado oculto.

        Convierte h_n desde [num_layers * num_directions, batch, hidden]
        a [batch, hidden * num_directions] tomando solo la ultima capa.

        Args:
            hidden_state: Estado oculto de la capa recurrente.

        Returns:
            Tensor [batch_size, representation_size].
        """
        if hidden_state.ndim != 3:
            raise RuntimeError(
                "El estado oculto debe tener tres dimensiones."
            )

        batch_size = hidden_state.shape[1]

        hidden_state = hidden_state.reshape(
            self.num_layers,
            self.num_directions,
            batch_size,
            self.hidden_size,
        )

        # Ultima capa
        last_layer_hidden = hidden_state[-1]

        if self.bidirectional:
            forward_hidden = last_layer_hidden[0]
            backward_hidden = last_layer_hidden[1]
            final_representation = torch.cat(
                [forward_hidden, backward_hidden], dim=1
            )
        else:
            final_representation = last_layer_hidden[0]

        return final_representation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass del modelo.

        Args:
            x: Tensor de entrada [batch, sequence_length, input_size].

        Returns:
            Tensor de predicciones [batch_size].
        """
        if x.ndim != 3:
            raise ValueError(
                "La entrada debe tener forma [batch, sequence, features]. "
                f"Forma recibida: {tuple(x.shape)}"
            )
        if x.shape[2] != self.input_size:
            raise ValueError(
                f"Numero incorrecto de variables. "
                f"Esperadas: {self.input_size}. "
                f"Recibidas: {x.shape[2]}."
            )

        recurrent_output = self.recurrent(x)

        if self.config.recurrent_type == "lstm":
            _, (final_hidden_state, _) = recurrent_output
        else:
            _, final_hidden_state = recurrent_output

        representation = self.extract_final_representation(
            final_hidden_state
        )
        representation = self.representation_normalization(
            representation
        )
        prediction = self.regression_head(representation)

        # [batch_size, 1] → [batch_size]
        return prediction.squeeze(-1)
