"""Scripts de experimentacion para entrenamiento y evaluacion de modelos.

Estos scripts ejecutan los flujos de entrenamiento que no encajan en
pipelines Kedro convencionales (ciclo iterativo GPU, checkpoints, etc.).

Scripts disponibles:
    run_prepare_datasets.py — Prepara datasets (features + ventanas temporales)
                              por (window, segmentation, horizon).
    run_hpo_training.py     — Optimizacion de hiperparametros (Optuna/TPE) por
                              combinacion; minimiza el RMSE de validacion.
    run_all.py              — CLI maestro: encadena preparacion + HPO.
"""
