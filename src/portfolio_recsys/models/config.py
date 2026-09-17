"""Configuraciones centralizadas para el pipeline de modelado.

Todas las dataclasses de este modulo son inmutables (frozen=True) y
serializables a JSON/YAML para su uso como parametros de Kedro.

Cada configuracion agrupa los hiperparametros de un aspecto concreto:
- RecurrentModelConfig: arquitectura de la red recurrente.
- TrainingConfig: bucle de entrenamiento (early stopping, scheduler, etc.).
- FeatureEngineeringConfig: feature engineering de mercado y fundamentales.
- TemporalSplitConfig: division temporal para backtesting.
- WindowConfig: ventanas temporales para la RNN.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal


# ─────────────────────────────────────────────────────────────────────────────
# Tipos compartidos
# ─────────────────────────────────────────────────────────────────────────────

RecurrentType = Literal["rnn", "gru", "lstm"]
"""Tipos de capa recurrente soportados."""

HeadActivationType = Literal["relu", "silu", "leaky_relu", "gelu", "linear"]
"""Tipos de activacion soportados para la cabeza de regresion."""


# ─────────────────────────────────────────────────────────────────────────────
# Configuracion de arquitectura
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RecurrentModelConfig:
    """Configuracion de la arquitectura recurrente.

    Define la topologia de la red: tipo de capa, dimensiones ocultas,
    numero de capas, bidireccionalidad, dropouts y cabeza de regresion.

    Attributes:
        recurrent_type: Tipo de capa recurrente ('rnn', 'gru', 'lstm').
        hidden_size: Dimensiones del estado oculto por direccion.
        num_layers: Numero de capas recurrentes apiladas.
        bidirectional: Si True, procesa la secuencia en ambas direcciones.
        recurrent_dropout: Dropout entre capas recurrentes (solo si num_layers > 1).
        head_dropout: Dropout en la cabeza de regresion.
        head_hidden_size: Neuronas en la capa oculta de la cabeza de regresion.
        head_activation: Activacion de la cabeza ('relu', 'silu', 'leaky_relu',
            'gelu', 'linear'). 'linear' = sin activacion (cabeza lineal directa).
        rnn_nonlinearity: Funcion de activacion para RNN vanilla ('tanh' o 'relu').
    """

    recurrent_type: RecurrentType

    hidden_size: int = 64
    num_layers: int = 2

    bidirectional: bool = False

    recurrent_dropout: float = 0.10
    head_dropout: float = 0.20

    head_hidden_size: int = 32
    head_activation: HeadActivationType = "relu"

    rnn_nonlinearity: Literal["tanh", "relu"] = "tanh"

    def to_dict(self) -> dict[str, Any]:
        """Serializa la configuracion a diccionario JSON-compatible."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecurrentModelConfig:
        """Reconstruye la configuracion desde un diccionario."""
        return cls(**data)


# ─────────────────────────────────────────────────────────────────────────────
# Configuracion de entrenamiento
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrainingConfig:
    """Hiperparametros del bucle de entrenamiento.

    Controla early stopping, scheduler de learning rate, gradient clipping
    y la funcion de perdida (Smooth L1 / Huber).

    Attributes:
        max_epochs: Numero maximo de epocas de entrenamiento.
        learning_rate: Learning rate inicial para AdamW.
        weight_decay: Regularizacion L2 para AdamW.
        max_gradient_norm: Limite de norma del gradiente (gradient clipping).
        early_stopping_patience: Epocas sin mejora antes de parar.
        early_stopping_min_delta: Mejora minima para considerar progreso.
        scheduler_patience: Epocas sin mejora antes de reducir LR.
        scheduler_factor: Factor multiplicativo al reducir LR.
        minimum_learning_rate: LR minimo del scheduler.
        smooth_l1_beta: Parametro beta de la perdida Smooth L1.
        batch_size: Tamano del lote (muestras por iteracion).
        random_seed: Semilla para reproducibilidad.
        num_workers: Workers del DataLoader (0 = main thread).
    """

    max_epochs: int = 50

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    max_gradient_norm: float = 1.0

    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-5

    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    minimum_learning_rate: float = 1e-6

    smooth_l1_beta: float = 1.0

    batch_size: int = 128
    random_seed: int = 42
    num_workers: int = 0

    # Loader rapido (ventanas pre-materializadas, sin workers). Elimina el cuello
    # de botella de carga de datos y aprovecha la GPU. store_on_gpu mantiene los
    # tensores en la GPU (usar solo si el split cabe en memoria de la GPU).
    use_fast_loader: bool = True
    store_on_gpu: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serializa la configuracion a diccionario JSON-compatible."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        """Reconstruye la configuracion desde un diccionario."""
        return cls(**data)


# ─────────────────────────────────────────────────────────────────────────────
# Configuracion de feature engineering
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    """Parametros para la generacion de features de mercado y fundamentales.

    Attributes:
        return_horizons: Diccionario {nombre_feature: num_sesiones} para retornos.
        fundamental_lag_days: Dias de retraso para disponibilidad de fundamentales.
            Simula el tiempo entre cierre fiscal y publicacion real (sin fecha
            de publicacion disponible, se usa un lag conservador).
        fundamental_log_mapping: Mapeo columna_fuente → nombre_feature (signed log1p).
        fundamental_ratio_mapping: Mapeo columna_fuente → nombre_feature (ratios directos).
        execution_lag_sessions: Sesiones entre senala y entrada al trade.
        holding_period_sessions: Sesiones que permanece abierta la posicion.
    """

    return_horizons: dict[str, int] = field(default_factory=lambda: {
        "market_return_1d": 1,
        "market_return_1w": 5,
        "market_return_2w": 10,
        "market_return_1m": 21,
        "market_return_2m": 42,
        "market_return_3m": 63,
        "market_return_6m": 126,
        "market_return_1y": 252,
        "market_return_2y": 504,
        "market_return_3y": 756,
    })

    # Lag entre el cierre fiscal y la disponibilidad del dato como feature.
    # Se fija a 0: se asume que cada estado financiero esta disponible en su
    # propia fecha de cierre real (no se alinea al 31/12 ni se interpola, por lo
    # que no hay informacion futura que retrasar). Valores >0 desplazarian la
    # fecha de disponibilidad hacia adelante (mas conservador).
    fundamental_lag_days: int = 0

    fundamental_log_mapping: dict[str, str] = field(default_factory=lambda: {
        "Total Revenues_eur": "fund_revenue_slog",
        "Cost of Goods Sold_eur": "fund_cogs_slog",
        "Gross Profit_eur": "fund_gross_profit_slog",
        "Total Operating Expenses_eur": "fund_opex_slog",
        "Operating Income_eur": "fund_operating_income_slog",
        "EBITDA_eur": "fund_ebitda_slog",
        "Net Income to Company_eur": "fund_net_income_slog",
        "Weighted Average Diluted Shares Outstanding": "fund_diluted_shares_slog",
        "Free Cash Flow_eur": "fund_free_cash_flow_slog",
    })

    fundamental_ratio_mapping: dict[str, str] = field(default_factory=lambda: {
        "gross_profit_ratio": "fund_gross_profit_ratio",
        "ebitda_ratio": "fund_ebitda_ratio",
        "operating_income_ratio": "fund_operating_income_ratio",
        "net_income_ratio": "fund_net_income_ratio",
    })

    execution_lag_sessions: int = 1
    holding_period_sessions: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Serializa la configuracion a diccionario JSON-compatible."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureEngineeringConfig:
        """Reconstruye la configuracion desde un diccionario."""
        return cls(**data)


# ─────────────────────────────────────────────────────────────────────────────
# Configuracion de split temporal
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TemporalSplitConfig:
    """Parametros de la division temporal para backtesting.

    La division se basa en las fechas de salida del trade (trade_exit_date)
    para evitar data leakage: todo trade cuya salida es posterior al corte
    de entrenamiento pertenece a validacion o test.

    Attributes:
        train_fraction: Fraccion del total de fechas para entrenamiento.
        validation_fraction: Fraccion para validacion.
        test_fraction: Fraccion para test.
        train_end_date: Fecha fija de fin de entrenamiento (override manual).
            Si es None, se calcula automaticamente a partir de las fracciones.
        validation_end_date: Fecha fija de fin de validacion (override manual).
            Si es None, se calcula automaticamente.
        test_end_date: Fecha fija de fin de test (override manual). Los trades
            cuya salida (trade_exit_date) sea posterior a esta fecha se marcan
            como 'excluded' y NO entran en ningun split. Sirve para acotar el
            periodo de evaluacion cuando la informacion mas reciente deja de ser
            fiable (p. ej. fundamentales sin cierre fiscal publicado). Si es
            None, el test no tiene tope superior (todo lo posterior a validacion).
    """

    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15

    train_end_date: str | None = "2021-12-31"
    validation_end_date: str | None = "2023-12-31"
    test_end_date: str | None = None

    def __post_init__(self) -> None:
        """Valida que las fracciones sumen 1.0."""
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            msg = f"Las fracciones deben sumar 1.0. Suma actual: {total}"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serializa la configuracion a diccionario JSON-compatible."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemporalSplitConfig:
        """Reconstruye la configuracion desde un diccionario."""
        return cls(**data)


# ─────────────────────────────────────────────────────────────────────────────
# Configuracion de ventanas temporales
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WindowConfig:
    """Parametros para la construccion de ventanas temporales.

    Attributes:
        sequence_length: Numero de sesiones de trading en cada ventana.
        max_calendar_gap_days: Maximo gap en dias de calendario permitido
            entre dos sesiones consecutivas dentro de la misma ventana.
            Si se supera, la ventana se descarta (evita cruzar suspensiones
            o huecos prolongados de datos).
        filter_anomalous_windows: Si True, descarta las ventanas de train y
            validation que contengan alguna fila marcada como retorno anomalo
            (columna anomaly_column). Las ventanas de test nunca se filtran.
            Por defecto False para no alterar el comportamiento de llamadas
            que no proporcionan la columna de anomalias.
        anomaly_column: Nombre de la columna booleana que marca las filas con
            retorno anomalo (calculada aguas arriba en compute_model_features).
    """

    sequence_length: int = 60
    max_calendar_gap_days: int = 10
    filter_anomalous_windows: bool = False
    anomaly_column: str = "is_anomalous_return"

    def to_dict(self) -> dict[str, Any]:
        """Serializa la configuracion a diccionario JSON-compatible."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WindowConfig:
        """Reconstruye la configuracion desde un diccionario."""
        return cls(**data)
