"""Genera artefactos de explicabilidad para todas las combinaciones (Opcion A).

Para cada combinacion (dataset x horizonte, excluyendo 1y) produce, en
``data/06_reporting/explainability/{window}/unified/{horizon}/{dataset}/``:

  - ``traceability.parquet``: cadena de decision (ranking, seleccion, peso) para
    TODAS las configuraciones del modelo (todos los K y estrategias) y todos los
    rebalanceos.
  - ``markowitz_decomposition.parquet``: descomposicion retorno/riesgo por activo
    para las mismas configuraciones y rebalanceos (con cache de covarianza).
  - ``ig_global_importance.csv``: importancia global de variables (|IG| medio).
  - ``ig_global_importance.png``: grafico de la importancia global (top-20).

El calculo se paraleliza por combinacion mediante ProcessPoolExecutor. La parte
mas costosa (Integrated Gradients) corre en CPU con hilos BLAS limitados por
worker para evitar sobre-suscripcion.

Uso:
    uv run --no-sync pr-explainability
    uv run --no-sync pr-explainability --horizons 1w,1m --datasets market,enriched
    uv run --no-sync pr-explainability --ig-samples 200 --ig-steps 32 --workers 4
"""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pr-explainability")

DEFAULT_HORIZONS = ["1d", "1w", "2w", "1m", "2m", "3m", "6m"]  # sin 1y
DEFAULT_DATASETS = ["market", "enriched"]
WINDOW = "w20"


def _process_combo(args: dict) -> dict:
    """Procesa una combinacion (dataset, horizonte). Ejecutado en un worker."""
    # Limitar hilos BLAS por worker (evita sobre-suscripcion con N procesos).
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "POLARS_MAX_THREADS"):
        os.environ.setdefault(var, "1")

    import numpy as np
    import polars as pl
    import torch

    from portfolio_recsys.explainability import attribution as attr
    from portfolio_recsys.explainability import markowitz_decomposition as md
    from portfolio_recsys.explainability import traceability as tr
    from portfolio_recsys.models.config import WindowConfig
    from portfolio_recsys.models.windows import build_temporal_window_store

    horizon = args["horizon"]
    dataset = args["dataset"]
    root = Path(args["root"])
    ig_samples = int(args["ig_samples"])
    ig_steps = int(args["ig_steps"])
    seq_len = int(args["seq_len"])
    ig_seed = int(args.get("ig_seed", 11))

    bt = root / f"data/06_reporting/portfolio/{WINDOW}/unified/{horizon}/backtest_results_{dataset}.json"
    prices_path = root / "data/02_intermediate/normalized_prices/stock_prices_eur.parquet"
    prepared = root / f"data/03_processed/model_features/{WINDOW}/unified/{horizon}/{dataset}_prepared.parquet"
    # Para el IG se prioriza el checkpoint del modelo reentrenado (test_seeds, la
    # semilla indicada), que refleja los datasets/feature actuales. Si no existe,
    # se recurre al mejor modelo del HPO como respaldo.
    models_base = root / f"data/04_models/{WINDOW}/unified/{horizon}"
    ckpt_seed = models_base / "test_seeds" / f"{dataset}_seed{ig_seed}_best.pt"
    ckpt_hpo = models_base / "hpo" / dataset / "best_model.pt"
    ckpt_path = ckpt_seed if ckpt_seed.exists() else ckpt_hpo

    out_dir = root / f"data/06_reporting/explainability/{WINDOW}/unified/{horizon}/{dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{dataset}/{horizon}"
    result = {"combo": tag, "status": "ok", "errors": []}

    try:
        results = tr.load_backtest_results(bt)

        # 1. Trazabilidad (todas las configs, todos los K, todos los rebalanceos)
        trace = tr.build_full_traceability(results)
        trace.write_parquet(out_dir / "traceability.parquet")

        # 2. Descomposicion de Markowitz (con cache de covarianza)
        prices_eur = pl.read_parquet(prices_path)
        decomp = md.build_full_decomposition(results, prices_eur)
        decomp.write_parquet(out_dir / "markowitz_decomposition.parquet")
        result["n_trace_rows"] = trace.height
        result["n_decomp_rows"] = decomp.height
    except Exception as exc:  # noqa: BLE001
        result["status"] = "partial"
        result["errors"].append(f"traza/descomp: {exc}")

    # 3. Integrated Gradients (importancia global)
    try:
        device = torch.device("cpu")
        model, ckpt = attr.load_model_from_checkpoint(ckpt_path, device)
        fcols = ckpt["feature_columns"]
        prep = pl.read_parquet(prepared)
        store = build_temporal_window_store(
            dataframe=prep,
            feature_columns=fcols,
            target_scaled_column=ckpt["preprocessor"]["target_scaled_column"],
            window_config=WindowConfig(sequence_length=seq_len, max_calendar_gap_days=10),
        )
        idx = store["window_index"].filter(pl.col("split") == "test")
        n = min(ig_samples, idx.height)
        rows = idx.sample(n=n, seed=42).to_dicts()

        def _win(r):
            tk = r["ticker"]; s = int(r["start_position"]); e = int(r["end_position"])
            return store["feature_arrays"][tk][s:e + 1]

        windows = np.stack([_win(r) for r in rows]).astype(np.float32)
        importance = attr.global_feature_importance(model, windows, fcols, device, n_steps=ig_steps)

        imp_df = pl.DataFrame(
            {"feature": [k for k, _ in importance],
             "importance_abs_mean": [v for _, v in importance]}
        )
        imp_df.write_csv(out_dir / "ig_global_importance.csv")
        _plot_importance(importance, dataset, horizon, out_dir / "ig_global_importance.png")
        result["ig_samples_used"] = n
        result["ig_checkpoint"] = "test_seeds" if ckpt_path == ckpt_seed else "hpo"
    except Exception as exc:  # noqa: BLE001
        result["status"] = "partial"
        result["errors"].append(f"ig: {exc}")

    return result


def _plot_importance(importance, dataset, horizon, out_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = importance[:20][::-1]
    names = [n for n, _ in top]
    vals = [v for _, v in top]
    plt.figure(figsize=(8, 7))
    plt.barh(names, vals, color="#0096CC")
    plt.xlabel("Importancia media |IG|")
    plt.title(f"Importancia global de variables — {dataset}, {horizon}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Genera artefactos de explicabilidad (todas las combinaciones).")
    parser.add_argument("--horizons", type=str, default=",".join(DEFAULT_HORIZONS))
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--ig-samples", type=int, default=200)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--ig-seed", type=int, default=11,
                        help="Semilla del checkpoint reentrenado (test_seeds) que "
                             "usa el IG. Si no existe, usa el mejor del HPO.")
    parser.add_argument("--workers", type=int, default=0,
                        help="Numero de procesos (0 = auto: nucleos-2).")
    args = parser.parse_args(argv)

    root = Path.cwd()
    horizons = [h.strip() for h in args.horizons.split(",") if h.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    tasks = [
        {"horizon": h, "dataset": d, "root": str(root),
         "ig_samples": args.ig_samples, "ig_steps": args.ig_steps,
         "seq_len": args.seq_len, "ig_seed": args.ig_seed}
        for h in horizons for d in datasets
    ]

    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 2)
    workers = min(workers, len(tasks))
    logger.info("Combinaciones a procesar: %d | procesos: %d", len(tasks), workers)

    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_combo, t): (t["dataset"], t["horizon"]) for t in tasks}
        for fut in as_completed(futures):
            ds, h = futures[fut]
            try:
                res = fut.result()
                msg = f"[{res['status'].upper()}] {res['combo']}"
                if res.get("n_trace_rows") is not None:
                    msg += f" | traza={res['n_trace_rows']} descomp={res.get('n_decomp_rows')}"
                if res.get("ig_samples_used") is not None:
                    msg += f" | ig_muestras={res['ig_samples_used']} (ckpt={res.get('ig_checkpoint','?')})"
                logger.info(msg)
                if res["errors"]:
                    for e in res["errors"]:
                        logger.warning("    %s: %s", res["combo"], e)
                    failures.append(res["combo"])
            except Exception:  # noqa: BLE001
                logger.exception("Fallo la combinacion %s/%s", ds, h)
                failures.append(f"{ds}/{h}")

    logger.info("=" * 60)
    if failures:
        logger.warning("Combinaciones con incidencias: %s", failures)
    else:
        logger.info("EXPLICABILIDAD COMPLETADA sin incidencias.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
