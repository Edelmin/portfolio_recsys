"""Genera la prediction_table sobre el split TEST reentrenando la config ganadora.

Layout multi-eje (window / segmentation / horizon).

Flujo por combinacion (window, segmentation, horizon):

    1. Lee la configuracion ganadora del HPO desde
       ``data/04_models/{window}/{seg}/{horizon}/hpo/best_params.json``
       (criterio: menor RMSE de VALIDACION; el TEST no intervino en la busqueda).
    2. Reconstruye la arquitectura + hiperparametros de entrenamiento y el tipo
       de dataset ganador (market / enriched).
    3. Carga el dataset preparado correspondiente
       (``{dataset_type}_prepared__{seg}__{window}__{horizon}``) y su preprocessor.
    4. Reentrena N semillas (train + validation) con esa configuracion y ejecuta
       inferencia sobre el split TEST (periodo posterior a validation_end_date,
       con el modelo congelado). Construye la tabla ensemble (media de semillas).
    5. Escribe ``prediction_table_test.parquet`` en la ruta de 3 ejes.

Motivacion: el backtest de carteras debe evaluarse sobre un periodo en el que el
modelo esta congelado y que NO intervino ni en el ajuste de pesos (train) ni en
la seleccion de arquitectura/hiperparametros (validation). Ese periodo es TEST.

Salida por combinacion:
    data/05_model_outputs/predictions/{window}/{seg}/{horizon}/prediction_table_test.parquet

Uso:
    # Horizontes de entrenamiento (sin 1y) de w20/unified con las 5 semillas de
    # robustez definidas en parameters.yaml (model_training.robustness_seeds)
    uv run pr-predict-test --window w20 --segmentation unified
    # Subconjunto de horizontes
    uv run pr-predict-test --window w20 --horizons 1d,3m
    # Por sector
    uv run pr-predict-test --segmentation by_sector --sectors Energy,Financials
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import polars as pl
import torch

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

from portfolio_recsys.models.architectures import RecurrentRegressor
from portfolio_recsys.models.config import RecurrentModelConfig, TrainingConfig
from portfolio_recsys.models.ensemble import build_seed_ensemble_table
from portfolio_recsys.models.prediction import build_prediction_table
from portfolio_recsys.models.reproducibility import set_random_seed
from portfolio_recsys.models.training import fit_recurrent_model
from portfolio_recsys.models.windows import create_fast_dataloaders_from_arrays
from portfolio_recsys.paths import Segmentation, dataset_name, models_dir, predictions_dir
from portfolio_recsys.pipelines.sector_mapping import SECTORS_WITH_FS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pr-predict-test")

try:  # UTF-8 en consola Windows
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:  # noqa: BLE001
    pass

ALL_HORIZONS = ["1d", "1w", "2w", "1m", "2m", "3m", "6m", "1y"]
# Horizontes que se entrenan/predicen por defecto. Se excluye "1y": el efecto de
# borde del split de test (finales de 2021 a finales de 2024, ~3 anios) deja muy
# pocas ventanas de test utiles a un anio de holding, por lo que no se entrena.
TRAINING_HORIZONS = ["1d", "1w", "2w", "1m", "2m", "3m", "6m"]
# Semillas de robustez multi-semilla (post-HPO). Se alinea con
# model_training.robustness_seeds de parameters.yaml (5 semillas). Si por algun
# motivo no estuviera disponible, se usa esta lista como respaldo.
DEFAULT_ROBUSTNESS_SEEDS = [11, 23, 42, 71, 101]


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruccion de la configuracion ganadora
# ─────────────────────────────────────────────────────────────────────────────


def _load_best_config(
    window: str, segmentation: Segmentation, horizon: str,
    base_training_config: dict[str, Any],
    dataset_type: str | None = None,
) -> tuple[RecurrentModelConfig, TrainingConfig, str] | None:
    """Lee best_params.json y reconstruye (model_config, training_config, dataset_type).

    Args:
        dataset_type: Si se indica ("market"/"enriched"), reconstruye la config
            ganadora de ESE dataset (hpo/{dataset_type}/best_params.json). Si es
            None, usa el dataset ganador (winner_dataset_type) del resumen.

    Devuelve None si no existe el best_params.json requerido (HPO no completado).
    """
    hpo_dir = models_dir(window, segmentation, horizon) / "hpo"

    if dataset_type is not None:
        # Config del dataset concreto solicitado.
        ds_path = hpo_dir / dataset_type / "best_params.json"
        if not ds_path.exists():
            logger.error(
                "[%s] no existe %s (¿termino el HPO de %s?). Se omite.",
                horizon, ds_path, dataset_type,
            )
            return None
        summary = json.loads(ds_path.read_text(encoding="utf-8"))
        best = summary["best_params"]
        dataset_type = summary.get("dataset_type", dataset_type)
    else:
        best_path = hpo_dir / "best_params.json"
        if not best_path.exists():
            logger.error(
                "[%s] no existe %s (¿termino el HPO?). Se omite.", horizon, best_path
            )
            return None

        summary = json.loads(best_path.read_text(encoding="utf-8"))

        # Layout nuevo (dos estudios por dataset): hpo/best_params.json es un resumen
        # comparativo con 'winner_dataset_type'; la config ganadora vive en
        # hpo/{dataset_type}/best_params.json. Layout antiguo: 'best_params' directo.
        if "winner_dataset_type" in summary:
            winner = summary["winner_dataset_type"]
            winner_path = hpo_dir / winner / "best_params.json"
            summary = json.loads(winner_path.read_text(encoding="utf-8"))
            best = summary["best_params"]
            dataset_type = summary.get("dataset_type", winner)
        else:
            best = summary["best_params"]
            dataset_type = best.get("dataset_type") or summary.get("dataset_type")

    model_config = RecurrentModelConfig(
        recurrent_type=best["recurrent_type"],
        hidden_size=best["hidden_size"],
        num_layers=best["num_layers"],
        bidirectional=best["bidirectional"],
        recurrent_dropout=best["recurrent_dropout"],
        head_dropout=best["head_dropout"],
        head_hidden_size=best["head_hidden_size"],
        head_activation=best["head_activation"],
    )

    # Hereda la config base y sobreescribe solo lo que el HPO busca (lr, wd).
    training_overrides = {
        **base_training_config,
        "learning_rate": best["learning_rate"],
        "weight_decay": best["weight_decay"],
    }
    training_config = TrainingConfig.from_dict(training_overrides)

    return model_config, training_config, dataset_type


# ─────────────────────────────────────────────────────────────────────────────
# Reentrenamiento multi-semilla + prediccion de TEST
# ─────────────────────────────────────────────────────────────────────────────


def predict_test_for_combo(
    catalog: Any,
    params: dict,
    window: str,
    segmentation: Segmentation,
    horizon: str,
    seeds: list[int],
    device: torch.device,
    dataset_type: str | None = None,
) -> pl.DataFrame | None:
    """Reentrena la config ganadora con las semillas dadas y arma la tabla ensemble de TEST.

    Args:
        seeds: Lista de semillas de robustez a promediar en el ensemble.
        dataset_type: "market"/"enriched" para reentrenar la config ganadora de
            ese dataset concreto; None para usar el dataset ganador global.
    """
    base_training_config = params["model_training"]["training_config"]
    best = _load_best_config(
        window, segmentation, horizon, base_training_config, dataset_type=dataset_type
    )
    if best is None:
        return None
    model_config, training_config, dataset_type = best

    logger.info(
        "[%s] ganador: %s h=%d L=%d bi=%s dataset=%s (lr=%.2e wd=%.2e)",
        horizon, model_config.recurrent_type, model_config.hidden_size,
        model_config.num_layers, model_config.bidirectional, dataset_type,
        training_config.learning_rate, training_config.weight_decay,
    )

    # --- Cargar VENTANAS persistidas (.npz) + indice de ventanas + preprocessor ---
    # No se reconstruyen las ventanas: se usan las ya materializadas de antemano
    # (dataset {dataset_type}_window), asegurando que el reentrenamiento ve
    # exactamente los mismos tensores que el HPO. El indice de ventanas persistido
    # ({dataset_type}_window_index) aporta la metadata (signal_date, fechas de
    # trade, split, target_log_return) para alinear las predicciones.
    try:
        window_arrays = catalog.load(
            dataset_name(f"{dataset_type}_window", window, segmentation, horizon)
        )
        window_index = catalog.load(
            dataset_name(f"{dataset_type}_window_index", window, segmentation, horizon)
        )
        preprocessor = catalog.load(
            dataset_name(f"{dataset_type}_preprocessor", window, segmentation, horizon)
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "[%s] fallo la carga de las ventanas %s (¿pr-prepare-datasets?)",
            horizon, dataset_type,
        )
        return None

    feature_columns = preprocessor["model_feature_columns"]

    n_test_windows = window_index.filter(pl.col("split") == "test").height
    if n_test_windows == 0:
        logger.warning("[%s] no hay ventanas de test; se omite.", horizon)
        return None
    logger.info("[%s] ventanas de test: %d", horizon, n_test_windows)

    batch_size = int(training_config.batch_size)
    store_on_gpu = bool(base_training_config.get("store_on_gpu", False))
    logger.info("[%s] semillas de robustez (%d): %s", horizon, len(seeds), seeds)

    out_dir = models_dir(window, segmentation, horizon) / "test_seeds"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_results: list[dict[str, Any]] = []
    seed_predictions: list[pl.DataFrame] = []

    for seed in seeds:
        logger.info("[%s] semilla %d: reentrenando config ganadora...", horizon, seed)
        set_random_seed(seed)

        train_loader, validation_loader, test_loader = create_fast_dataloaders_from_arrays(
            window_arrays=window_arrays,
            batch_size=batch_size,
            random_seed=seed,
            compute_device=device,
            store_on_gpu=store_on_gpu,
        )
        if test_loader is None:
            logger.warning("[%s] el artefacto de ventanas no tiene split test.", horizon)
            return None

        model = RecurrentRegressor(
            input_size=len(feature_columns),
            config=model_config,
        ).to(device)

        checkpoint_path = out_dir / f"{dataset_type}_seed{seed}_best.pt"

        try:
            model, history, _best_ckpt = fit_recurrent_model(
                model=model,
                train_loader=train_loader,
                validation_loader=validation_loader,
                model_config=model_config,
                training_config=training_config,
                feature_columns=feature_columns,
                preprocessor=preprocessor,
                device=device,
                checkpoint_path=checkpoint_path,
            )

            # Persistir el historial de entrenamiento por epoca (train/val loss,
            # early stopping) para poder graficar las curvas de aprendizaje y la
            # parada temprana de cada semilla a posteriori.
            history_path = out_dir / f"{dataset_type}_seed{seed}_history.parquet"
            try:
                history.write_parquet(history_path)
                logger.info(
                    "[%s] semilla %d: historial guardado (%d epocas) en %s",
                    horizon, seed, history.height, history_path.name,
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "[%s] semilla %d: no se pudo guardar el historial", horizon, seed
                )

            preds = build_prediction_table(
                model=model,
                dataloader=test_loader,
                window_index=window_index,
                preprocessor=preprocessor,
                device=device,
            )
        finally:
            del model, train_loader, validation_loader, test_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        preds_test = preds.filter(pl.col("split") == "test")
        seed_results.append({"seed": seed})
        seed_predictions.append(preds_test)
        logger.info("[%s] semilla %d: %d predicciones test", horizon, seed, preds_test.height)

    if not seed_predictions:
        logger.error("[%s] sin predicciones de test.", horizon)
        return None

    ensemble, _ = build_seed_ensemble_table(seed_results, seed_predictions)
    ensemble = ensemble.filter(pl.col("split") == "test")
    return ensemble


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
            "Genera prediction_table sobre TEST reentrenando la config ganadora "
            "del HPO con multiples semillas (layout window/segmentation/horizon)."
        )
    )
    parser.add_argument(
        "--window", type=str, default="w20",
        help="Ventana temporal (ej: w20, w60). Por defecto w20.",
    )
    parser.add_argument(
        "--segmentation", choices=["unified", "by_sector", "both"], default="unified",
        help="Segmentacion. Por defecto unified.",
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
        "--seeds", type=int, default=None,
        help=(
            "Numero de semillas para el ensemble. Por defecto usa "
            "model_training.robustness_seeds de parameters.yaml (5 semillas). Si "
            "se indica N, usa las primeras N de esa lista."
        ),
    )
    parser.add_argument(
        "--dataset-types", type=str, default="market,enriched",
        help=(
            "Tipos de dataset a predecir, separados por coma. Por defecto ambos "
            "(market,enriched): genera una tabla de test por cada uno."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    root = Path.cwd()
    bootstrap_project(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    horizons = [h.strip() for h in args.horizons.split(",") if h.strip()]
    unknown_h = [h for h in horizons if h not in ALL_HORIZONS]
    if unknown_h:
        parser.error(f"Horizontes desconocidos: {unknown_h}. Disponibles: {ALL_HORIZONS}")

    sectors = (
        [s.strip() for s in args.sectors.split(",") if s.strip()]
        if args.sectors else list(SECTORS_WITH_FS)
    )
    unknown_s = [s for s in sectors if s not in SECTORS_WITH_FS]
    if unknown_s:
        parser.error(f"Sectores desconocidos: {unknown_s}. Disponibles: {SECTORS_WITH_FS}")

    segmentations = _resolve_segmentations(args.segmentation, sectors)
    dataset_types = [
        d.strip() for d in args.dataset_types.split(",") if d.strip()
    ]
    unknown_d = [d for d in dataset_types if d not in ("market", "enriched")]
    if unknown_d:
        parser.error(
            f"Tipos de dataset desconocidos: {unknown_d}. Disponibles: market, enriched"
        )

    results: dict[str, bool] = {}
    with KedroSession.create(project_path=root) as session:
        context = session.load_context()
        catalog = context.catalog
        params = context.params

        # Semillas de robustez: fuente unica de verdad en parameters.yaml
        # (model_training.robustness_seeds). Con --seeds N se toman las primeras N.
        robustness_seeds = list(
            params["model_training"].get("robustness_seeds", DEFAULT_ROBUSTNESS_SEEDS)
        )
        if args.seeds is not None:
            if args.seeds < 1:
                parser.error("--seeds debe ser >= 1")
            seeds = robustness_seeds[: args.seeds]
        else:
            seeds = robustness_seeds
        logger.info("Dispositivo: %s | semillas de robustez (%d): %s", device, len(seeds), seeds)

        for segmentation in segmentations:
            for h in horizons:
                for dataset_type in dataset_types:
                    key = f"{segmentation.dataset_token}/{h}/{dataset_type}"
                    try:
                        ensemble = predict_test_for_combo(
                            catalog, params, args.window, segmentation, h,
                            seeds, device, dataset_type=dataset_type,
                        )
                    except Exception:  # noqa: BLE001
                        logger.exception("[%s] fallo la inferencia de test", key)
                        results[key] = False
                        continue
                    if ensemble is None:
                        results[key] = False
                        continue
                    out_dir = predictions_dir(args.window, segmentation, h)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    # Fichero por tipo de dataset (market / enriched).
                    out = out_dir / f"prediction_table_test_{dataset_type}.parquet"
                    ensemble.write_parquet(out)
                    logger.info(
                        "[%s] tabla de test guardada: %s (%d filas)",
                        key, out, ensemble.height,
                    )
                    results[key] = True

    ok = sum(1 for v in results.values() if v)
    total = len(results)
    logger.info("=" * 70)
    logger.info("Resumen: %d/%d combinaciones con prediccion de test", ok, total)
    for k, v in results.items():
        logger.info("  %-20s %s", k, "OK" if v else "FALLO/omitido")
    logger.info("=" * 70)
    return 0 if ok == total and total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
