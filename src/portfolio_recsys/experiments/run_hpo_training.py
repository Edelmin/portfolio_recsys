"""Entrenamiento con optimizacion de hiperparametros (pr-train-hpo).

Para cada combinacion (window, segmentation, horizon) ejecuta DOS estudios de
Optuna independientes (TPE + pruning), uno por tipo de dataset (market y
enriched). Cada estudio minimiza el RMSE de VALIDACION buscando arquitectura e
hiperparametros de entrenamiento, con el MISMO presupuesto de n_trials, y
conserva su propio modelo ganador. El split de TEST no interviene en la busqueda.

Entradas (datasets preprocesados, generados por pr-prepare-datasets):
    market_prepared__{seg}__{window}__{horizon}, enriched_prepared__...,
    market_preprocessor__..., enriched_preprocessor__...

Salidas (NO catalogadas en Kedro; via portfolio_recsys.paths.models_dir):
    data/04_models/{window}/{segmentation}/{horizon}/hpo/
        market/
            optuna_study.db    -> estudio Optuna del dataset market (SQLite)
            best_params.json   -> mejor configuracion del dataset market
            best_model.pt      -> checkpoint del mejor trial (+ best_model.json)
        enriched/
            optuna_study.db    -> estudio Optuna del dataset enriched (SQLite)
            best_params.json   -> mejor configuracion del dataset enriched
            best_model.pt      -> checkpoint del mejor trial (+ best_model.json)
        best_params.json       -> resumen comparativo (winner_dataset_type)

REANUDABLE: cada estudio Optuna persiste en SQLite; relanzar continua hasta
alcanzar n_trials. --force borra los estudios y modelos y empieza de cero.

Uso:
    # Todo: ambas segmentaciones (unified primero), todas las ventanas/horizontes
    uv run pr-train-hpo
    # Solo unificado, w60, horizontes 2m y 3m
    uv run pr-train-hpo --segmentation unified --windows w60 --horizons 2m,3m
    # Por sector, subconjunto
    uv run pr-train-hpo --segmentation by_sector --sectors Energy,Financials
    # Forzar re-busqueda desde cero
    uv run pr-train-hpo --force
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

from portfolio_recsys.models.hpo import HpoObjective, best_config_summary, run_study
from portfolio_recsys.paths import Segmentation, dataset_name, models_dir
from portfolio_recsys.pipelines.sector_mapping import SECTORS_WITH_FS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

try:  # UTF-8 en consola Windows (evita UnicodeEncodeError en logs)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:  # noqa: BLE001
    pass

ALL_HORIZONS = ["1d", "1w", "2w", "1m", "2m", "3m", "6m", "1y"]
# Horizontes que se entrenan por defecto. Se excluye "1y": el efecto de borde del
# split de test (finales de 2021 a finales de 2024, ~3 anios) deja muy pocas
# ventanas de test utiles a un anio de holding, por lo que no se entrena.
TRAINING_HORIZONS = ["1d", "1w", "2w", "1m", "2m", "3m", "6m"]


# ─────────────────────────────────────────────────────────────────────────────
# Carga de datos y construccion de window stores
# ─────────────────────────────────────────────────────────────────────────────


def _load_windows_and_preprocessors(
    catalog: Any,
    window: str,
    segmentation: Segmentation,
    horizon: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Carga las VENTANAS ya materializadas (.npz) y los preprocesadores.

    Consume directamente el dataset de entrada DIRECTA al modelo persistido por
    pr-prepare-datasets (``{market,enriched}_window``), sin reconstruir las
    ventanas: el modelo entrena con exactamente los mismos tensores que se
    calcularon de antemano (incluida la cuarentena de ventanas anomalas ya
    aplicada en la materializacion).

    Returns:
        (window_arrays, preprocessors) con claves 'market' y 'enriched'.
    """
    window_arrays: dict[str, Any] = {}
    preprocessors: dict[str, Any] = {}

    for dataset_type in ("market", "enriched"):
        arrays = catalog.load(
            dataset_name(f"{dataset_type}_window", window, segmentation, horizon)
        )
        preprocessor = catalog.load(
            dataset_name(f"{dataset_type}_preprocessor", window, segmentation, horizon)
        )
        window_arrays[dataset_type] = arrays
        preprocessors[dataset_type] = preprocessor

    return window_arrays, preprocessors


# ─────────────────────────────────────────────────────────────────────────────
# HPO de una combinacion
# ─────────────────────────────────────────────────────────────────────────────


def _tune_one(
    catalog: Any,
    params: dict,
    window: str,
    segmentation: Segmentation,
    horizon: str,
    device: torch.device,
    force: bool,
) -> str:
    """Ejecuta el estudio HPO de una combinacion. Devuelve estado."""
    hpo_params = params["model_training"]["hpo"]
    n_trials = int(hpo_params["n_trials"])
    seed = int(hpo_params["random_seed"])
    base_training_config = params["model_training"]["training_config"]

    hpo_root = models_dir(window, segmentation, horizon) / "hpo"
    hpo_root.mkdir(parents=True, exist_ok=True)

    try:
        window_arrays, preprocessors = _load_windows_and_preprocessors(
            catalog, window, segmentation, horizon
        )
    except Exception:
        logger.exception(
            "  [FAIL] carga de ventanas %s | %s | %s (¿ejecutaste pr-prepare-datasets?)",
            window, segmentation.label, horizon,
        )
        return "error"

    # Un estudio independiente por tipo de dataset (market y enriched): cada uno
    # recibe el mismo presupuesto de n_trials y conserva su modelo ganador.
    results_per_dataset: dict[str, dict[str, Any]] = {}
    for dataset_type in ("market", "enriched"):
        ds_dir = hpo_root / dataset_type
        ds_dir.mkdir(parents=True, exist_ok=True)
        storage_path = ds_dir / "optuna_study.db"
        best_path = ds_dir / "best_params.json"
        best_model_path = ds_dir / "best_model.pt"

        if force:
            for p in (storage_path, best_model_path, best_model_path.with_suffix(".json")):
                if p.exists():
                    p.unlink()
            logger.info("  [FORCE] artefactos previos borrados: %s", ds_dir)

        study_name = f"{segmentation.dataset_token}__{window}__{horizon}__{dataset_type}"
        logger.info(
            "  [HPO ] %s | %s | %s | dataset=%s  (n_trials=%d, seed=%d)",
            window, segmentation.label, horizon, dataset_type, n_trials, seed,
        )

        objective = HpoObjective(
            dataset_type=dataset_type,
            window_arrays=window_arrays[dataset_type],
            preprocessor=preprocessors[dataset_type],
            base_training_config=base_training_config,
            device=device,
            checkpoint_dir=ds_dir / "trial_checkpoints",
            random_seed=seed,
            best_model_path=best_model_path,
        )

        try:
            study = run_study(
                objective=objective,
                study_name=study_name,
                storage_path=storage_path,
                n_trials=n_trials,
                random_seed=seed,
            )
        except Exception:
            logger.exception("  [FAIL] estudio HPO %s", study_name)
            return "error"

        summary = best_config_summary(study)
        summary.update({
            "window": window,
            "segmentation": segmentation.dataset_token,
            "horizon": horizon,
            "dataset_type": dataset_type,
            "study_name": study_name,
            "best_model_path": str(best_model_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        best_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        results_per_dataset[dataset_type] = summary
        logger.info(
            "  [DONE] %s | mejor RMSE_val=%.6f | modelo=%s",
            study_name, summary["best_value_rmse_val"], best_model_path,
        )

    # Resumen comparativo: cual de los dos datasets gano (menor RMSE de val).
    winner = min(
        results_per_dataset.values(), key=lambda s: s["best_value_rmse_val"]
    )
    comparison = {
        "window": window,
        "segmentation": segmentation.dataset_token,
        "horizon": horizon,
        "winner_dataset_type": winner["dataset_type"],
        "per_dataset": {
            dt: {
                "best_value_rmse_val": s["best_value_rmse_val"],
                "best_params": s["best_params"],
                "best_model_path": s["best_model_path"],
            }
            for dt, s in results_per_dataset.items()
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (hpo_root / "best_params.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info(
        "  [WIN ] %s|%s|%s -> dataset ganador=%s",
        window, segmentation.label, horizon, winner["dataset_type"],
    )
    return "done"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_segmentations(seg_arg: str, sectors: list[str]) -> list[Segmentation]:
    result: list[Segmentation] = []
    if seg_arg in ("unified", "both"):
        result.append(Segmentation.unified())
    if seg_arg in ("by_sector", "both"):
        result.extend(Segmentation.for_sector(s) for s in sectors)
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Entrenamiento con optimizacion de hiperparametros (Optuna/TPE) por "
            "(window, segmentation, horizon). Reanudable; --force reinicia."
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
        "--horizons", type=str, default=",".join(TRAINING_HORIZONS),
        help=(
            "Horizontes separados por coma. Por defecto los de entrenamiento "
            f"(sin 1y): {','.join(TRAINING_HORIZONS)}."
        ),
    )
    parser.add_argument(
        "--sectors", type=str, default=None,
        help="Sectores (canonicos) para by_sector. Por defecto, todos.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Reinicia el estudio HPO desde cero (borra el SQLite previo).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    project_path = Path.cwd()
    bootstrap_project(project_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Dispositivo: %s", device)

    with KedroSession.create(project_path=project_path) as session:
        context = session.load_context()
        catalog = context.catalog
        params = context.params

    window_experiments = params["compute_model_features"]["window_experiments"]
    all_windows = list(window_experiments.keys())
    windows = (
        [w.strip() for w in args.windows.split(",") if w.strip()]
        if args.windows else all_windows
    )
    horizons = [h.strip() for h in args.horizons.split(",") if h.strip()]
    sectors = (
        [s.strip() for s in args.sectors.split(",") if s.strip()]
        if args.sectors else list(SECTORS_WITH_FS)
    )

    unknown_w = [w for w in windows if w not in all_windows]
    if unknown_w:
        parser.error(f"Ventanas desconocidas: {unknown_w}. Disponibles: {all_windows}")
    unknown_h = [h for h in horizons if h not in ALL_HORIZONS]
    if unknown_h:
        parser.error(f"Horizontes desconocidos: {unknown_h}. Disponibles: {ALL_HORIZONS}")
    unknown_s = [s for s in sectors if s not in SECTORS_WITH_FS]
    if unknown_s:
        parser.error(f"Sectores desconocidos: {unknown_s}. Disponibles: {SECTORS_WITH_FS}")

    if not params["model_training"]["hpo"].get("enabled", True):
        logger.warning("HPO deshabilitado en parameters.yaml (model_training.hpo.enabled=false).")
        return

    segmentations = _resolve_segmentations(args.segmentation, sectors)

    logger.info("=" * 70)
    logger.info(
        "HPO | segmentation=%s | windows=%s | horizons=%s | force=%s",
        args.segmentation, windows, horizons, args.force,
    )
    logger.info("=" * 70)

    tally = {"done": 0, "error": 0}
    # unified primero (segmentations ya viene ordenada)
    for segmentation in segmentations:
        logger.info("--- Segmentacion: %s ---", segmentation.label)
        for window in windows:
            for horizon in horizons:
                status = _tune_one(
                    catalog, params, window, segmentation, horizon, device, args.force
                )
                tally[status] = tally.get(status, 0) + 1

    logger.info("=" * 70)
    logger.info("HPO COMPLETADO | estudios OK=%d, errores=%d", tally["done"], tally["error"])
    logger.info("=" * 70)

    if tally["error"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
