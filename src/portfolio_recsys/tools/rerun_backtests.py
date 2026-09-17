"""Relanza el backtest de carteras Markowitz por (window, segmentation, horizon).

Reejecuta el pipeline `report_portfolio__{seg}__{window}__{horizon}` de Kedro
para cada horizonte seleccionado, regenerando:

- data/06_reporting/portfolio/{window}/{segmentation}/{horizon}/backtest_results.json
- data/06_reporting/portfolio/{window}/{segmentation}/{horizon}/portfolio_report.md

NO reentrena modelos ni recalcula features: consume la prediction_table,
split_configuration, stock_prices_eur y benchmark_prices_eur ya persistidos.

Los ejes window y segmentation tienen valores por defecto (w60 / unified) para
compatibilidad con el flujo previo.

Uso:
    uv run pr-backtest-all                                   # w60/unified, todos los horizontes
    uv run pr-backtest-all --horizons 1d,3m,1y               # solo algunos horizontes
    uv run pr-backtest-all --window w20                      # otra ventana
    uv run pr-backtest-all --segmentation by_sector --sector Energy
    uv run pr-backtest-all --list                            # lista horizontes disponibles
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

from portfolio_recsys.paths import Segmentation


def project_root() -> Path:
    """Raiz del proyecto (directorio de trabajo, donde vive pyproject.toml)."""
    return Path.cwd()


def _segmentation_from_args(segmentation: str, sector: str | None) -> Segmentation:
    """Construye la Segmentation desde los args del CLI."""
    if segmentation == "unified":
        return Segmentation.unified()
    if sector is None:
        raise SystemExit("--segmentation by_sector requiere --sector.")
    return Segmentation.for_sector(sector)


def _pipeline_name(
    segmentation: Segmentation,
    window: str,
    horizon: str,
    dataset_type: str | None = None,
) -> str:
    """Nombre del pipeline registrado.

    Si ``dataset_type`` es ``market``/``enriched`` se usa la variante por tipo de
    dataset (``report_portfolio_{dataset_type}__...``), que consume
    ``prediction_table_test_{dataset_type}`` y produce
    ``portfolio_backtest_results_{dataset_type}`` (fichero
    ``backtest_results_{dataset_type}.json``). Si es None, el modo de un unico
    dataset combinado (``report_portfolio__...``).
    """
    prefix = (
        f"report_portfolio_{dataset_type}"
        if dataset_type in ("market", "enriched")
        else "report_portfolio"
    )
    return f"{prefix}__{segmentation.dataset_token}__{window}__{horizon}"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pr-backtest-all")


def _available_horizons() -> list[str]:
    """Lee las claves de horizonte desde los parametros del proyecto."""
    project_path = project_root()
    bootstrap_project(project_path)
    with KedroSession.create(project_path=project_path) as session:
        params = session.load_context().params
    return list(params["compute_model_features"]["horizons"].keys())


def _run_backtest(
    segmentation: Segmentation,
    window: str,
    horizon: str,
    backtest_workers_override: int | None = None,
    dataset_type: str | None = None,
) -> bool:
    """Ejecuta report_portfolio[_{dataset_type}]__{seg}__{window}__{horizon}.
    True si tiene exito.

    backtest_workers_override permite forzar el nº de procesos de la
    paralelizacion por configuracion (util al paralelizar horizontes, para
    repartir los nucleos y evitar sobre-suscripcion).
    """
    project_path = project_root()
    bootstrap_project(project_path)
    pipeline_name = _pipeline_name(segmentation, window, horizon, dataset_type)
    try:
        runtime_params = None
        if backtest_workers_override is not None:
            # OJO: runtime_params REEMPLAZA la clave de primer nivel completa; si
            # pasaramos {"markowitz_portfolio": {"backtest_num_workers": N}} se
            # perderian el resto de sub-claves (top_k_configs, strategies, ...).
            # Cargamos el dict base y solo sobrescribimos el nº de workers.
            with KedroSession.create(project_path=project_path) as probe:
                base_mp = dict(probe.load_context().params.get("markowitz_portfolio", {}))
            base_mp["backtest_num_workers"] = backtest_workers_override
            runtime_params = {"markowitz_portfolio": base_mp}

        kwargs = {"project_path": project_path}
        if runtime_params is not None:
            kwargs["runtime_params"] = runtime_params
        with KedroSession.create(**kwargs) as session:
            session.run(pipeline_name=pipeline_name)
        return True
    except Exception:  # noqa: BLE001
        logger.exception("Fallo el backtest %s", pipeline_name)
        return False


def _run_backtest_subprocess(
    segmentation: Segmentation, window: str, horizon: str, backtest_workers: int
) -> bool:
    """Lanza un horizonte en un subproceso independiente (para paralelizar
    horizontes). Cada subproceso usa `backtest_workers` procesos internos."""
    import subprocess
    import sys
    cmd = [sys.executable, "-m", "portfolio_recsys.tools.rerun_backtests",
           "--horizons", horizon, "--config-workers", str(backtest_workers),
           "--window", window, "--segmentation", segmentation.kind]
    if segmentation.kind == "by_sector":
        cmd += ["--sector", segmentation.sector]
    proc = subprocess.run(cmd, cwd=str(project_root()))
    return proc.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Relanza el backtest de carteras Markowitz por horizonte.",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default=None,
        help="Lista separada por comas (ej: 1d,3m,1y). Por defecto, todos.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Lista los horizontes disponibles y sale.",
    )
    parser.add_argument(
        "--parallel-horizons",
        action="store_true",
        help="Ejecuta los horizontes en paralelo (subprocesos). Reparte los "
             "nucleos entre horizontes para NO sobre-suscribir la CPU.",
    )
    parser.add_argument(
        "--config-workers",
        type=int,
        default=None,
        help="Nº de procesos para la paralelizacion por configuracion dentro de "
             "cada horizonte (override de markowitz_portfolio.backtest_num_workers).",
    )
    parser.add_argument(
        "--window",
        type=str,
        default="w60",
        help="Clave de ventana temporal (por defecto w60).",
    )
    parser.add_argument(
        "--segmentation",
        choices=["unified", "by_sector"],
        default="unified",
        help="Segmentacion de los datos (por defecto unified).",
    )
    parser.add_argument(
        "--sector",
        type=str,
        default=None,
        help="Sector canonico (requerido si --segmentation by_sector).",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["market", "enriched", "both"],
        default=None,
        help="Conjunto de datos. 'market'/'enriched' ejecuta la variante por "
             "tipo (produce backtest_results_{tipo}.json). 'both' ejecuta ambas. "
             "Si se omite, usa el layout clasico (backtest_results.json).",
    )
    args = parser.parse_args(argv)

    segmentation = _segmentation_from_args(args.segmentation, args.sector)
    all_horizons = _available_horizons()

    if args.dataset_type == "both":
        dataset_types: list[str | None] = ["market", "enriched"]
    elif args.dataset_type in ("market", "enriched"):
        dataset_types = [args.dataset_type]
    else:
        dataset_types = [None]

    if args.list:
        print("Horizontes disponibles:", ", ".join(all_horizons))
        return 0

    if args.horizons:
        selected = [h.strip() for h in args.horizons.split(",") if h.strip()]
        unknown = [h for h in selected if h not in all_horizons]
        if unknown:
            logger.error("Horizontes desconocidos: %s. Disponibles: %s",
                         unknown, all_horizons)
            return 2
    else:
        selected = all_horizons

    dtype_label = (
        ", ".join(dt for dt in dataset_types if dt) if any(dataset_types) else "clasico"
    )
    logger.info("Relanzando backtest | window=%s | %s | dataset=%s | %d horizontes: %s",
                args.window, segmentation.label, dtype_label,
                len(selected), ", ".join(selected))

    results: dict[str, bool] = {}

    if args.parallel_horizons and len(selected) > 1:
        # Paralelizar horizontes en subprocesos, repartiendo nucleos entre ellos.
        from concurrent.futures import ThreadPoolExecutor
        total = os.cpu_count() or 1
        reserved = 4
        usable = max(1, total - reserved)
        # nucleos por horizonte concurrente (al menos 1)
        n_parallel = min(len(selected), usable)
        workers_per_horizon = max(1, usable // n_parallel)
        logger.info("Paralelizando %d horizontes (%d concurrentes, %d procesos/horizonte)",
                    len(selected), n_parallel, workers_per_horizon)
        start_all = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_parallel) as ex:
            futs = {ex.submit(_run_backtest_subprocess, segmentation, args.window,
                              h, workers_per_horizon): h
                    for h in selected}
            for fut in futs:
                h = futs[fut]
                results[h] = fut.result()
        logger.info("Horizontes en paralelo completados en %.1fs",
                    time.perf_counter() - start_all)
    else:
        for dtype in dataset_types:
            for horizon in selected:
                key = f"{dtype}/{horizon}" if dtype else horizon
                logger.info("=" * 60)
                logger.info("Backtest %s", key)
                start = time.perf_counter()
                ok = _run_backtest(segmentation, args.window, horizon,
                                   backtest_workers_override=args.config_workers,
                                   dataset_type=dtype)
                elapsed = time.perf_counter() - start
                results[key] = ok
                logger.info("Backtest %s %s (%.1fs)",
                            key, "OK" if ok else "FALLO", elapsed)

    logger.info("=" * 60)
    total = len(results)
    n_ok = sum(results.values())
    logger.info("Resumen: %d/%d backtests OK", n_ok, total)
    for key, ok in results.items():
        logger.info("  %-14s %s", key, "OK" if ok else "FALLO")

    return 0 if n_ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
