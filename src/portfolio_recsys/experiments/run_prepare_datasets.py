"""Orquestador de preparacion de datasets de modelado (pr-prepare-datasets).

Genera los datasets listos para entrenamiento (features + ventanas temporales)
para las combinaciones de los tres ejes de experimentacion:

    - window       : tamaño de ventana temporal (ej: w60, w20).
    - segmentation : "unified" (todos los sectores, opcional one-hot) y/o
                     "by_sector" (un dataset por sector, escalado propio).
    - horizon      : horizonte de rebalanceo (1d, 1w, 2w, 1m, 2m, 3m, 6m, 1y).

Para cada combinacion ejecuta el pipeline de Kedro
`prepare__{seg_token}__{window}__{horizon}` (compute_model_features +
build_temporal_windows), que escribe en:

    data/03_processed/model_features/{window}/{segmentation}/{horizon}/

PRIORIDAD: cuando se piden ambas segmentaciones, se completa TODO 'unified'
antes de empezar 'by_sector' (unified es la linea base de referencia).

REANUDABLE: cada combinacion completada deja un marker `.prepared`. Las
combinaciones ya completadas se saltan. Usar --force para regenerar.

Uso:
    # Todo: ambas segmentaciones, todas las ventanas configuradas, todos los horizontes
    uv run pr-prepare-datasets

    # Solo la linea unificada, ventana w60, horizontes 2m y 3m
    uv run pr-prepare-datasets --segmentation unified --windows w60 --horizons 2m,3m

    # Solo por sector, un subconjunto de sectores
    uv run pr-prepare-datasets --segmentation by_sector --sectors Energy,Financials

    # Ambas (unified primero), forzando regeneracion
    uv run pr-prepare-datasets --segmentation both --force
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

from portfolio_recsys.paths import Segmentation, features_dir
from portfolio_recsys.pipelines.sector_mapping import SECTORS_WITH_FS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# En Windows, la consola por defecto usa cp1252 y el RichHandler de Kedro puede
# fallar al emitir caracteres no ASCII (acentos, flechas). Forzar UTF-8 en los
# streams de salida evita UnicodeEncodeError sin cambiar los mensajes.
try:  # pragma: no cover - depende de la plataforma
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:  # noqa: BLE001
    pass

ALL_HORIZONS = ["1d", "1w", "2w", "1m", "2m", "3m", "6m", "1y"]
PREPARED_MARKER = ".prepared"


# ─────────────────────────────────────────────────────────────────────────────
# Reanudabilidad
# ─────────────────────────────────────────────────────────────────────────────


def _marker_path(window: str, segmentation: Segmentation, horizon: str) -> Path:
    """Ruta del marker de completado de una combinacion."""
    return features_dir(window, segmentation, horizon) / PREPARED_MARKER


def _is_prepared(window: str, segmentation: Segmentation, horizon: str) -> bool:
    return _marker_path(window, segmentation, horizon).exists()


def _mark_prepared(
    window: str, segmentation: Segmentation, horizon: str, payload: dict[str, Any]
) -> None:
    marker = _marker_path(window, segmentation, horizon)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ejecucion de una combinacion
# ─────────────────────────────────────────────────────────────────────────────


def _pipeline_name(segmentation: Segmentation, window: str, horizon: str) -> str:
    return f"prepare__{segmentation.dataset_token}__{window}__{horizon}"


def _run_pipeline(project_path: Path, pipeline_name: str) -> None:
    """Ejecuta un pipeline de Kedro en una sesion aislada."""
    bootstrap_project(project_path)
    with KedroSession.create(project_path=project_path) as session:
        session.run(pipeline_name=pipeline_name)


def _prepare_one(
    project_path: Path,
    window: str,
    segmentation: Segmentation,
    horizon: str,
    force: bool,
) -> str:
    """Prepara una combinacion. Devuelve el estado: 'skipped', 'done', 'error'."""
    if not force and _is_prepared(window, segmentation, horizon):
        logger.info(
            "  [SKIP] %s | %s | %s (ya preparado)",
            window, segmentation.label, horizon,
        )
        return "skipped"

    pipeline_name = _pipeline_name(segmentation, window, horizon)
    logger.info(
        "  [RUN ] %s | %s | %s -> %s",
        window, segmentation.label, horizon, pipeline_name,
    )
    try:
        _run_pipeline(project_path, pipeline_name)
    except Exception:
        logger.exception(
            "  [FAIL] %s | %s | %s", window, segmentation.label, horizon
        )
        return "error"

    _mark_prepared(
        window,
        segmentation,
        horizon,
        {
            "window": window,
            "segmentation": segmentation.dataset_token,
            "horizon": horizon,
            "pipeline": pipeline_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    return "done"


# ─────────────────────────────────────────────────────────────────────────────
# Orquestacion
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_segmentations(
    segmentation_arg: str, sectors: list[str]
) -> list[Segmentation]:
    """Construye la lista ORDENADA de segmentaciones a procesar.

    'unified' siempre va primero (prioridad de completado).
    """
    result: list[Segmentation] = []
    if segmentation_arg in ("unified", "both"):
        result.append(Segmentation.unified())
    if segmentation_arg in ("by_sector", "both"):
        result.extend(Segmentation.for_sector(s) for s in sectors)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepara datasets de modelado por (window, segmentation, horizon). "
            "Reanudable; usa --force para regenerar."
        )
    )
    parser.add_argument(
        "--segmentation",
        choices=["unified", "by_sector", "both"],
        default="both",
        help="Segmentacion a preparar. 'both' hace unified primero (default).",
    )
    parser.add_argument(
        "--windows",
        type=str,
        default=None,
        help=(
            "Claves de ventana separadas por coma (ej: w60,w20). Por defecto, "
            "todas las definidas en parameters.yaml (window_experiments)."
        ),
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default=",".join(ALL_HORIZONS),
        help=f"Horizontes separados por coma. Por defecto todos: {','.join(ALL_HORIZONS)}.",
    )
    parser.add_argument(
        "--sectors",
        type=str,
        default=None,
        help=(
            "Sectores (canonicos) separados por coma para by_sector. Por "
            "defecto, todos los que tienen estados financieros."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenera aunque la combinacion ya este preparada.",
    )
    args = parser.parse_args(argv)

    project_path = Path.cwd()

    # Cargar ventanas disponibles desde parameters.yaml
    bootstrap_project(project_path)
    with KedroSession.create(project_path=project_path) as session:
        params = session.load_context().params
    window_experiments = params["compute_model_features"]["window_experiments"]

    all_windows = list(window_experiments.keys())
    windows = (
        [w.strip() for w in args.windows.split(",") if w.strip()]
        if args.windows
        else all_windows
    )
    horizons = [h.strip() for h in args.horizons.split(",") if h.strip()]
    sectors = (
        [s.strip() for s in args.sectors.split(",") if s.strip()]
        if args.sectors
        else list(SECTORS_WITH_FS)
    )

    # Validaciones
    unknown_w = [w for w in windows if w not in all_windows]
    if unknown_w:
        parser.error(f"Ventanas desconocidas: {unknown_w}. Disponibles: {all_windows}")
    unknown_h = [h for h in horizons if h not in ALL_HORIZONS]
    if unknown_h:
        parser.error(f"Horizontes desconocidos: {unknown_h}. Disponibles: {ALL_HORIZONS}")
    unknown_s = [s for s in sectors if s not in SECTORS_WITH_FS]
    if unknown_s:
        parser.error(f"Sectores desconocidos: {unknown_s}. Disponibles: {SECTORS_WITH_FS}")

    segmentations = _resolve_segmentations(args.segmentation, sectors)

    logger.info("=" * 70)
    logger.info(
        "Preparacion de datasets | segmentation=%s | windows=%s | horizons=%s | force=%s",
        args.segmentation, windows, horizons, args.force,
    )
    if args.segmentation in ("by_sector", "both"):
        logger.info("Sectores by_sector: %s", sectors)
    logger.info("=" * 70)

    # Contadores
    tally = {"done": 0, "skipped": 0, "error": 0}

    # PRIORIDAD: iterar por segmentacion primero garantiza que unified se
    # completa antes de empezar by_sector (segmentations ya viene ordenada).
    for segmentation in segmentations:
        logger.info("--- Segmentacion: %s ---", segmentation.label)
        for window in windows:
            for horizon in horizons:
                status = _prepare_one(
                    project_path, window, segmentation, horizon, args.force
                )
                tally[status] += 1

    logger.info("=" * 70)
    logger.info(
        "PREPARACION COMPLETADA | generados=%d, saltados=%d, errores=%d",
        tally["done"], tally["skipped"], tally["error"],
    )
    logger.info("=" * 70)

    if tally["error"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
