"""CLI maestro: prepara datasets y entrena con HPO en un solo comando (pr-run-all).

Encadena, con los MISMOS ejes (window / segmentation / horizon), las dos fases:

    1. Preparacion de datasets  (pr-prepare-datasets)
       -> features + ventanas temporales para cada combinacion.
    2. Entrenamiento con HPO     (pr-train-hpo)
       -> estudio Optuna (TPE + pruning) que minimiza el RMSE de validacion.

Ambas fases son REANUDABLES: la preparacion salta combinaciones ya generadas
(marker .prepared) y el HPO continua el estudio Optuna persistido. Cuando se
piden ambas segmentaciones, se prioriza completar 'unified' antes de 'by_sector'
en las dos fases.

Uso:
    # Todo de una: prepara y entrena (ambas segmentaciones, todas las ventanas/horizontes)
    uv run pr-run-all

    # Solo unificado, ventana w60, horizontes 2m y 3m
    uv run pr-run-all --segmentation unified --windows w60 --horizons 2m,3m

    # Solo la fase de preparacion (equivale a pr-prepare-datasets)
    uv run pr-run-all --only prepare

    # Solo la fase de HPO (asume datasets ya preparados)
    uv run pr-run-all --only train

    # Forzar regeneracion de datasets y reinicio de estudios HPO
    uv run pr-run-all --force
"""

from __future__ import annotations

import argparse
import logging
import sys

from portfolio_recsys.experiments import run_hpo_training, run_prepare_datasets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

try:  # UTF-8 en consola Windows
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:  # noqa: BLE001
    pass


def _forward_args(args: argparse.Namespace) -> list[str]:
    """Traduce los args comunes a la lista de argv para los sub-CLIs."""
    argv: list[str] = [
        "--segmentation", args.segmentation,
        "--horizons", args.horizons,
    ]
    if args.windows:
        argv += ["--windows", args.windows]
    if args.sectors:
        argv += ["--sectors", args.sectors]
    if args.force:
        argv += ["--force"]
    return argv


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "CLI maestro: prepara datasets y entrena con HPO en un solo comando, "
            "compartiendo los ejes window/segmentation/horizon."
        )
    )
    parser.add_argument(
        "--segmentation", choices=["unified", "by_sector", "both"], default="both",
        help="Segmentacion. 'both' hace unified primero (default).",
    )
    parser.add_argument(
        "--windows", type=str, default=None,
        help="Ventanas separadas por coma (ej: w60,w20). Por defecto, todas.",
    )
    parser.add_argument(
        "--horizons", type=str, default=",".join(run_hpo_training.TRAINING_HORIZONS),
        help=(
            "Horizontes separados por coma. Por defecto los de entrenamiento "
            f"(sin 1y): {','.join(run_hpo_training.TRAINING_HORIZONS)}."
        ),
    )
    parser.add_argument(
        "--sectors", type=str, default=None,
        help="Sectores (canonicos) para by_sector. Por defecto, todos.",
    )
    parser.add_argument(
        "--only", choices=["prepare", "train", "all"], default="all",
        help="Ejecutar solo una fase ('prepare' o 'train') o ambas (default 'all').",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Fuerza regeneracion de datasets y reinicio de estudios HPO.",
    )
    args = parser.parse_args(argv)

    forwarded = _forward_args(args)

    if args.only in ("prepare", "all"):
        logger.info("#" * 70)
        logger.info("# FASE 1/2: PREPARACION DE DATASETS")
        logger.info("#" * 70)
        try:
            run_prepare_datasets.main(forwarded)
        except SystemExit as exc:
            if exc.code not in (0, None):
                logger.error(
                    "La preparacion de datasets termino con errores (code=%s). "
                    "Se aborta antes del entrenamiento.", exc.code,
                )
                raise
        if args.only == "prepare":
            return

    if args.only in ("train", "all"):
        logger.info("#" * 70)
        logger.info("# FASE 2/2: ENTRENAMIENTO CON HPO (Optuna/TPE)")
        logger.info("#" * 70)
        run_hpo_training.main(forwarded)


if __name__ == "__main__":
    main()
