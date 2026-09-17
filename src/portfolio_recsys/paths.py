"""Fuente unica de verdad para las rutas de datos de modelado.

Este modulo centraliza la construccion de rutas y de identificadores de dataset
para las tres dimensiones de experimentacion del proyecto:

    - window       : tamaño de la ventana temporal (clave legible, ej: "w60", "w20").
                     Se corresponde con las entradas de
                     `compute_model_features.window_experiments` en parameters.yaml.
    - segmentation : forma de agrupar las empresas para entrenar:
                       * "unified"            -> un unico dataset con todos los sectores
                                                 (opcionalmente con one-hot de sector).
                       * "by_sector"/{sector} -> un dataset por sector (nombre canonico).
    - horizon      : horizonte de rebalanceo/prediccion (ej: "1d", "3m", "1y").

Layout en disco (uniforme para todas las capas):

    data/03_processed/model_features/{window}/{segmentation_path}/{horizon}/...
    data/04_models/{window}/{segmentation_path}/{horizon}/...
    data/05_model_outputs/predictions/{window}/{segmentation_path}/{horizon}/...
    data/06_reporting/portfolio/{window}/{segmentation_path}/{horizon}/...

donde `segmentation_path` es:
    - "unified"                para la segmentacion unificada.
    - "by_sector/{Canonical}"  para un sector concreto (ej: "by_sector/Energy").

IMPORTANTE:
    - Los MODELOS (.pt) NO se catalogan en Kedro; se leen/escriben por ruta usando
      este modulo (ver `models_dir`).
    - Los DATASETS de datos SI se catalogan; el catalogo usa el mismo layout via
      dataset factories, y el identificador de dataset se construye con
      `dataset_name()` para que ambos mundos (catalogo y disco) coincidan.

Convencion de identificadores de dataset (para el catalogo de Kedro):

    {base}__{segmentation_token}__{window}__{horizon}

    ej: market_prepared__unified__w60__3m
        market_prepared__sector_Energy__w20__3m

El token de segmentacion NO contiene barras (para ser un nombre de dataset
valido): "unified" o "sector_{Canonical}". El factory del catalogo lo traduce a
la subruta correspondiente.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from portfolio_recsys.pipelines.sector_mapping import SECTORS_WITH_FS

# ─────────────────────────────────────────────────────────────────────────────
# Constantes de capas de datos
# ─────────────────────────────────────────────────────────────────────────────

DATA_ROOT = Path("data")

PROCESSED_ROOT = DATA_ROOT / "03_processed" / "model_features"
MODELS_ROOT = DATA_ROOT / "04_models"
PREDICTIONS_ROOT = DATA_ROOT / "05_model_outputs" / "predictions"
PORTFOLIO_REPORT_ROOT = DATA_ROOT / "06_reporting" / "portfolio"

UNIFIED = "unified"
BY_SECTOR = "by_sector"

SegmentationKind = Literal["unified", "by_sector"]


# ─────────────────────────────────────────────────────────────────────────────
# Segmentacion
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Segmentation:
    """Describe una segmentacion de datos para modelado.

    Attributes:
        kind: "unified" (todos los sectores juntos) o "by_sector" (un sector).
        sector: Nombre canonico del sector cuando kind == "by_sector".
            Debe ser None cuando kind == "unified".
    """

    kind: SegmentationKind
    sector: str | None = None

    def __post_init__(self) -> None:
        if self.kind == BY_SECTOR:
            if self.sector is None:
                raise ValueError("segmentation 'by_sector' requiere 'sector'.")
            if self.sector not in SECTORS_WITH_FS:
                raise ValueError(
                    f"Sector desconocido: {self.sector!r}. "
                    f"Disponibles: {SECTORS_WITH_FS}"
                )
        elif self.kind == UNIFIED:
            if self.sector is not None:
                raise ValueError("segmentation 'unified' no admite 'sector'.")
        else:
            raise ValueError(
                f"kind de segmentacion no reconocido: {self.kind!r}. "
                f"Use '{UNIFIED}' o '{BY_SECTOR}'."
            )

    # --- Representaciones ---

    @property
    def path_fragment(self) -> Path:
        """Subruta en disco: 'unified' o 'by_sector/{Canonical}'."""
        if self.kind == UNIFIED:
            return Path(UNIFIED)
        return Path(BY_SECTOR) / self.sector  # type: ignore[arg-type]

    @property
    def dataset_token(self) -> str:
        """Token para el identificador de dataset (sin barras).

        'unified' o 'sector_{Canonical}'.
        """
        if self.kind == UNIFIED:
            return UNIFIED
        return f"sector_{self.sector}"

    @property
    def label(self) -> str:
        """Etiqueta legible para logs/informes."""
        if self.kind == UNIFIED:
            return "unificado (todos los sectores)"
        return f"sector {self.sector}"

    # --- Constructores ---

    @classmethod
    def unified(cls) -> "Segmentation":
        return cls(kind=UNIFIED)

    @classmethod
    def for_sector(cls, sector: str) -> "Segmentation":
        return cls(kind=BY_SECTOR, sector=sector)


# ─────────────────────────────────────────────────────────────────────────────
# Rutas por capa
# ─────────────────────────────────────────────────────────────────────────────


def _leaf(root: Path, window: str, segmentation: Segmentation, horizon: str) -> Path:
    """Construye {root}/{window}/{segmentation_path}/{horizon}."""
    return root / window / segmentation.path_fragment / horizon


def features_dir(window: str, segmentation: Segmentation, horizon: str) -> Path:
    """Directorio de datasets preprocesados (03_processed/model_features)."""
    return _leaf(PROCESSED_ROOT, window, segmentation, horizon)


def models_dir(window: str, segmentation: Segmentation, horizon: str) -> Path:
    """Directorio de modelos entrenados (.pt). NO catalogado en Kedro."""
    return _leaf(MODELS_ROOT, window, segmentation, horizon)


def predictions_dir(window: str, segmentation: Segmentation, horizon: str) -> Path:
    """Directorio de tablas de prediccion (05_model_outputs/predictions)."""
    return _leaf(PREDICTIONS_ROOT, window, segmentation, horizon)


def portfolio_report_dir(window: str, segmentation: Segmentation, horizon: str) -> Path:
    """Directorio de informes de carteras (06_reporting/portfolio)."""
    return _leaf(PORTFOLIO_REPORT_ROOT, window, segmentation, horizon)


# ─────────────────────────────────────────────────────────────────────────────
# Identificadores de dataset (para el catalogo de Kedro)
# ─────────────────────────────────────────────────────────────────────────────


def dataset_name(
    base: str, window: str, segmentation: Segmentation, horizon: str
) -> str:
    """Construye el identificador de dataset del catalogo.

    Formato: {base}__{segmentation_token}__{window}__{horizon}

    Args:
        base: Nombre base del dataset (ej: "market_prepared").
        window: Clave de ventana (ej: "w60").
        segmentation: Segmentation.
        horizon: Horizonte (ej: "3m").

    Returns:
        Identificador de dataset, ej: "market_prepared__unified__w60__3m".
    """
    return f"{base}__{segmentation.dataset_token}__{window}__{horizon}"
