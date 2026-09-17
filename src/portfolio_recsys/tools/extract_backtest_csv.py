"""Extrae las metricas de los backtests a un CSV plano (fuente para la memoria).

Lee, para cada horizonte, los ficheros
``data/06_reporting/portfolio/{window}/{segmentation}/{horizon}/backtest_results_{tipo}.json``
(market y enriched) y produce un CSV con una fila por (horizonte, dataset,
config) y columnas:

    horizon,dataset,n_rebalances,split,config,total_return,annualized_return,
    annualized_volatility,sharpe_ratio,sortino_ratio,max_drawdown

El CSV se escribe por defecto en
``data/06_reporting/portfolio/{window}/{segmentation}/backtest_metrics.csv``.

Uso::

    uv run pr-backtest-csv                                   # w20/unified, todos los horizontes
    uv run pr-backtest-csv --window w20 --segmentation unified
    uv run pr-backtest-csv --horizons "1d,1w,1m"
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from portfolio_recsys.paths import Segmentation, portfolio_report_dir

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pr-backtest-csv")

DEFAULT_HORIZONS = ["1d", "1w", "2w", "1m", "2m", "3m", "6m"]
DATASETS = ("market", "enriched")

COLS = [
    "horizon", "dataset", "n_rebalances", "split", "config",
    "total_return", "annualized_return", "annualized_volatility",
    "sharpe_ratio", "sortino_ratio", "max_drawdown",
]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extrae las metricas de los backtests a un CSV plano.")
    parser.add_argument("--window", type=str, default="w20",
                        help="Ventana temporal (por defecto w20).")
    parser.add_argument("--segmentation", choices=["unified", "by_sector"],
                        default="unified", help="Segmentacion (por defecto unified).")
    parser.add_argument("--sector", type=str, default=None,
                        help="Sector canonico (si --segmentation by_sector).")
    parser.add_argument("--horizons", type=str, default=",".join(DEFAULT_HORIZONS),
                        help="Horizontes separados por comas.")
    parser.add_argument("--output", type=str, default=None,
                        help="Ruta del CSV de salida. Por defecto, "
                             "backtest_metrics.csv en la carpeta de carteras.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    segmentation = (
        Segmentation.unified()
        if args.segmentation == "unified"
        else Segmentation.for_sector(args.sector)
    )
    horizons = [h.strip() for h in args.horizons.split(",") if h.strip()]

    # La carpeta base (sin horizonte) es el padre del directorio de un horizonte.
    base_dir = portfolio_report_dir(args.window, segmentation, horizons[0]).parent
    out_path = Path(args.output) if args.output else base_dir / "backtest_metrics.csv"

    rows: list[dict] = []
    for h in horizons:
        hdir = portfolio_report_dir(args.window, segmentation, h)
        for dataset in DATASETS:
            p = hdir / f"backtest_results_{dataset}.json"
            if not p.exists():
                logger.warning("No existe %s", p)
                continue
            data = json.loads(p.read_text(encoding="utf-8"))
            split = data.get("evaluation_split", "test")
            n_reb = data.get("n_rebalances", 0)
            for cfg_name, cfg in data.get("configs", {}).items():
                m = cfg.get("metrics", {})
                rows.append({
                    "horizon": h,
                    "dataset": dataset,
                    "n_rebalances": n_reb,
                    "split": split,
                    "config": cfg_name,
                    "total_return": m.get("total_return"),
                    "annualized_return": m.get("annualized_return"),
                    "annualized_volatility": m.get("annualized_volatility"),
                    "sharpe_ratio": m.get("sharpe_ratio"),
                    "sortino_ratio": m.get("sortino_ratio"),
                    "max_drawdown": m.get("max_drawdown"),
                })

    if not rows:
        logger.error("No se encontro ningun backtest. Revisa window/segmentation.")
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Escrito %s con %d filas.", out_path, len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
