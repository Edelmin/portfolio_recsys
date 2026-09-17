"""Optimizacion de hiperparametros con Optuna (TPE + pruning).

Este modulo define el espacio de busqueda y la funcion objetivo para optimizar
la arquitectura y los hiperparametros de entrenamiento de los modelos
recurrentes, minimizando el RMSE sobre el split de VALIDACION.

Principios:
    - Solo se usan train + validation. El split de TEST NUNCA interviene en el
      HPO (ni en el ajuste ni en la seleccion), para no contaminar la evaluacion
      final out-of-sample.
    - Metrica objetivo: RMSE de validacion (rmse_scaled), a minimizar.
    - 1 semilla por trial (la robustez multi-semilla se aplica DESPUES, sobre la
      configuracion ganadora).
    - Sampler TPE (bayesiano) + MedianPruner: los trials poco prometedores se
      abortan pronto (reporte por epoca via callback), ahorrando computo.
    - El tipo de dataset (market / enriched) forma parte del espacio de busqueda:
      un unico estudio compara ambos y devuelve la mejor combinacion conjunta.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import optuna
import torch

from portfolio_recsys.models.architectures import RecurrentRegressor
from portfolio_recsys.models.config import RecurrentModelConfig, TrainingConfig
from portfolio_recsys.models.reproducibility import set_random_seed
from portfolio_recsys.models.training import fit_recurrent_model
from portfolio_recsys.models.windows import create_fast_dataloaders_from_arrays

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Espacio de busqueda
# ─────────────────────────────────────────────────────────────────────────────


def suggest_configs(
    trial: optuna.Trial,
    base_training_config: dict[str, Any],
) -> tuple[RecurrentModelConfig, TrainingConfig]:
    """Muestrea una configuracion (arquitectura + entrenamiento).

    El tipo de dataset (market / enriched) NO forma parte del espacio de
    busqueda: se fija por estudio (un estudio independiente por dataset), de modo
    que ambos reciben el mismo presupuesto de trials y producen cada uno su
    propio modelo ganador comparable.

    Args:
        trial: Trial de Optuna.
        base_training_config: Config de entrenamiento base (dict de parameters.yaml).
            Los campos NO buscados (batch_size, epochs, scheduler, loader) se
            heredan de aqui; solo learning_rate y weight_decay se muestrean.

    Returns:
        (model_config, training_config).
    """
    # --- Arquitectura ---
    recurrent_type = trial.suggest_categorical("recurrent_type", ["rnn", "gru", "lstm"])
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 10)
    bidirectional = trial.suggest_categorical("bidirectional", [False, True])
    recurrent_dropout = trial.suggest_float("recurrent_dropout", 0.0, 0.4)
    head_dropout = trial.suggest_float("head_dropout", 0.0, 0.5)
    head_hidden_size = trial.suggest_categorical("head_hidden_size", [16, 32, 64])
    head_activation = trial.suggest_categorical(
        "head_activation", ["relu", "silu", "leaky_relu", "gelu", "linear"]
    )

    model_config = RecurrentModelConfig(
        recurrent_type=recurrent_type,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
        recurrent_dropout=recurrent_dropout,
        head_dropout=head_dropout,
        head_hidden_size=head_hidden_size,
        head_activation=head_activation,
    )

    # --- Entrenamiento ---
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    training_overrides = {
        **base_training_config,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
    }
    training_config = TrainingConfig.from_dict(training_overrides)

    return model_config, training_config


# ─────────────────────────────────────────────────────────────────────────────
# Funcion objetivo
# ─────────────────────────────────────────────────────────────────────────────


class HpoObjective:
    """Funcion objetivo para Optuna: entrena 1 semilla y devuelve RMSE de val.

    Cada instancia opera sobre UN unico tipo de dataset (market o enriched),
    fijado en el constructor: se lanza un estudio independiente por dataset. El
    mejor checkpoint del estudio se conserva en disco (best_model.pt) para que
    quede almacenado el modelo ganador de cada dataset.
    """

    def __init__(
        self,
        dataset_type: str,
        window_arrays: dict[str, Any],
        preprocessor: dict[str, Any],
        base_training_config: dict[str, Any],
        device: torch.device,
        checkpoint_dir: Path,
        random_seed: int = 42,
        best_model_path: Path | None = None,
    ) -> None:
        """
        Args:
            dataset_type: 'market' o 'enriched' (fijo para todo el estudio).
            window_arrays: Ventanas ya materializadas y persistidas (.npz) del
                dataset correspondiente, con claves ``{split}_X``, ``{split}_y``,
                ``{split}_sample_ids`` y metadata (feature_columns, etc.). El
                modelo entrena con estos tensores tal cual, sin reconstruirlos.
            preprocessor: Preprocesador del dataset correspondiente.
            base_training_config: Config base de entrenamiento (dict).
            device: Dispositivo de computo.
            checkpoint_dir: Carpeta para checkpoints temporales de trials.
            random_seed: Semilla fija para todos los trials (comparabilidad).
            best_model_path: Ruta donde conservar el mejor checkpoint del estudio.
                Si es None, no se conserva (comportamiento antiguo).
        """
        self.dataset_type = dataset_type
        self.window_arrays = window_arrays
        self.preprocessor = preprocessor
        self.base_training_config = base_training_config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.random_seed = random_seed
        self.best_model_path = Path(best_model_path) if best_model_path else None

    def __call__(self, trial: optuna.Trial) -> float:
        set_random_seed(self.random_seed)

        model_config, training_config = suggest_configs(
            trial, self.base_training_config
        )

        dataset_type = self.dataset_type
        preprocessor = self.preprocessor
        feature_columns = preprocessor["model_feature_columns"]

        # Loaders desde las ventanas persistidas (.npz): el modelo entrena con
        # exactamente los tensores calculados de antemano, sin reconstruirlos.
        train_loader, validation_loader, _ = create_fast_dataloaders_from_arrays(
            window_arrays=self.window_arrays,
            batch_size=training_config.batch_size,
            random_seed=self.random_seed,
            compute_device=self.device,
            store_on_gpu=training_config.store_on_gpu,
        )

        model = RecurrentRegressor(
            input_size=len(feature_columns),
            config=model_config,
        ).to(self.device)

        checkpoint_path = self.checkpoint_dir / f"trial_{trial.number}.pt"

        # Callback de pruning: reporta el RMSE (escalado) de validacion por epoca
        # y aborta el trial si Optuna lo considera poco prometedor.
        def _epoch_callback(epoch: int, validation_metrics: dict[str, float]) -> None:
            rmse = validation_metrics["rmse_scaled"]
            trial.report(rmse, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            _model, history, best_checkpoint = fit_recurrent_model(
                model=model,
                train_loader=train_loader,
                validation_loader=validation_loader,
                model_config=model_config,
                training_config=training_config,
                feature_columns=feature_columns,
                preprocessor=preprocessor,
                device=self.device,
                checkpoint_path=checkpoint_path,
                epoch_callback=_epoch_callback,
            )
        finally:
            # Limpieza de memoria GPU entre trials
            del model, train_loader, validation_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # RMSE de validacion del mejor checkpoint (escala del target escalado).
        # Recuperamos el RMSE de la epoca del mejor checkpoint desde el historial.
        best_epoch = int(best_checkpoint["epoch"])
        best_row = history.filter(history["epoch"] == best_epoch)
        rmse_val = float(best_row["validation_rmse_scaled"][0])

        # Guardar metadata util en el trial
        trial.set_user_attr("dataset_type", dataset_type)
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("epochs_executed", int(history.height))
        trial.set_user_attr("total_parameters", sum(p.numel() for p in _model.parameters()))

        # Conservar el mejor modelo del estudio: si este trial mejora el mejor
        # RMSE de validacion visto hasta ahora, promocionar su checkpoint a
        # best_model.pt (junto con la config que lo genero). Asi queda almacenado
        # el modelo ganador de este dataset.
        promoted = False
        if self.best_model_path is not None:
            prior_best = None
            for t in trial.study.trials:
                if t.number == trial.number:
                    continue
                if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None:
                    prior_best = t.value if prior_best is None else min(prior_best, t.value)
            if prior_best is None or rmse_val < prior_best:
                try:
                    self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(checkpoint_path, self.best_model_path)
                    meta = {
                        "dataset_type": dataset_type,
                        "trial_number": trial.number,
                        "rmse_val": rmse_val,
                        "best_epoch": best_epoch,
                        "model_config": model_config.to_dict(),
                        "training_config": training_config.to_dict(),
                        "feature_columns": feature_columns,
                    }
                    self.best_model_path.with_suffix(".json").write_text(
                        json.dumps(meta, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8",
                    )
                    promoted = True
                except OSError:
                    logger.warning("No se pudo promover el checkpoint a %s", self.best_model_path)

        # Limpiar checkpoint temporal del trial (el ganador ya se copio aparte).
        if not (promoted and self.best_model_path == checkpoint_path):
            try:
                checkpoint_path.unlink(missing_ok=True)
            except OSError:
                pass

        return rmse_val


# ─────────────────────────────────────────────────────────────────────────────
# Runner del estudio
# ─────────────────────────────────────────────────────────────────────────────


def run_study(
    objective: HpoObjective,
    study_name: str,
    storage_path: Path,
    n_trials: int,
    random_seed: int = 42,
) -> optuna.Study:
    """Crea (o reanuda) un estudio Optuna y ejecuta n_trials.

    Args:
        objective: Funcion objetivo (HpoObjective).
        study_name: Nombre del estudio (identifica la combinacion window/seg/horizon).
        storage_path: Ruta del fichero SQLite para persistencia/reanudabilidad.
        n_trials: Numero de trials a ejecutar en esta llamada.
        random_seed: Semilla del sampler TPE (reproducibilidad de la busqueda).

    Returns:
        El estudio Optuna tras ejecutar los trials.
    """
    storage_path = Path(storage_path)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_path.as_posix()}"

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,      # no podar hasta tener 5 trials completos
        n_warmup_steps=3,        # dar al menos 3 epocas antes de considerar podar
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    # Trials ya completados (reanudabilidad): solo ejecutar los que falten.
    completed = len([
        t for t in study.trials
        if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    ])
    remaining = max(0, n_trials - completed)
    if remaining == 0:
        logger.info(
            "[%s] ya tiene %d/%d trials; nada que ejecutar.",
            study_name, completed, n_trials,
        )
        return study

    logger.info(
        "[%s] ejecutando %d trials (ya completados: %d/%d)",
        study_name, remaining, completed, n_trials,
    )
    study.optimize(objective, n_trials=remaining)
    return study


def best_config_summary(study: optuna.Study) -> dict[str, Any]:
    """Extrae la mejor configuracion del estudio en formato serializable."""
    best = study.best_trial
    return {
        "best_value_rmse_val": best.value,
        "best_params": dict(best.params),
        "best_user_attrs": dict(best.user_attrs),
        "n_trials": len(study.trials),
        "n_complete": len([
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]),
        "n_pruned": len([
            t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
        ]),
    }
