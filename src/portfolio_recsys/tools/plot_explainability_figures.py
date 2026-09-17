"""Genera las figuras de explicabilidad de la memoria a partir de los artefactos.

Lee los artefactos que produce ``pr-explainability`` en
``data/06_reporting/explainability/{window}/unified/{horizon}/{dataset}/`` y
produce, en ``reports/Memoria/images/explainability/``, seis PDF:

  - ``traceability_{tipo}.pdf``: cadena de decision de un rebalanceo (retorno
    predicho que ordena la seleccion vs peso asignado por Markowitz).
  - ``markowitz_decomposition_{tipo}.pdf``: contribucion al retorno vs al riesgo
    por activo para el mismo rebalanceo.
  - ``ig_importance_{tipo}.pdf``: importancia global de variables (top-15) por IG.

Los casos representativos por defecto son los que usa la memoria: el conjunto
``market`` a horizonte semanal (mejor rendimiento del estudio) y el ``enriched``
a horizonte diario. Para trazabilidad y descomposicion se emplea la
configuracion ``top10_long_only`` y el primer rebalanceo con optimizacion
factible.

Uso::

    uv run pr-explainability-figures
    uv run pr-explainability-figures --config top10_long_only
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pr-explainability-figures")

WINDOW = "w20"
# Casos representativos (dataset, horizonte) que usa la memoria.
CASES = {"market": "1w", "enriched": "1d"}


def _artifact_dir(root: Path, dataset: str, horizon: str) -> Path:
    return root / f"data/06_reporting/explainability/{WINDOW}/unified/{horizon}/{dataset}"


def _pick_rebalance(trace: pl.DataFrame, config: str) -> str:
    """Elige el primer rebalanceo (con optimizacion factible si existe)."""
    sub = trace.filter(pl.col("config") == config)
    if sub.is_empty():
        raise ValueError(f"config {config} no encontrada en la trazabilidad")
    optimal = sub.filter(pl.col("optimization_status") == "optimal")
    pool = optimal if not optimal.is_empty() else sub
    return sorted(pool["rebalance_date"].unique().to_list())[0]


def plot_traceability(trace: pl.DataFrame, config: str, out_path: Path) -> str:
    reb = _pick_rebalance(trace, config)
    sub = (
        trace.filter((pl.col("config") == config) & (pl.col("rebalance_date") == reb))
        .sort("rank")
    )
    tickers = sub["ticker"].to_list()
    pred = sub["predicted_return"].to_numpy() * 100.0
    weight = sub["weight_pct"].to_numpy()
    order = list(range(len(tickers)))[::-1]  # rank 1 arriba

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].barh([tickers[i] for i in order], [pred[i] for i in order], color="#0096CC")
    axes[0].set_xlabel("Log-retorno predicho (×100)")
    axes[0].grid(True, axis="x", alpha=0.3)

    axes[1].barh([tickers[i] for i in order], [weight[i] for i in order], color="#F2A900")
    axes[1].set_xlabel("Peso asignado (%)")
    axes[1].grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura guardada: %s (rebalanceo %s)", out_path.name, reb[:10])
    return reb


def plot_markowitz_decomposition(
    decomp: pl.DataFrame, config: str, out_path: Path, reb: str | None = None,
) -> None:
    if reb is None:
        reb = sorted(
            decomp.filter(pl.col("config") == config)["rebalance_date"].unique().to_list()
        )[0]
    sub = (
        decomp.filter((pl.col("config") == config) & (pl.col("rebalance_date") == reb))
        .sort("weight", descending=True)
    )
    tickers = sub["ticker"].to_list()[::-1]
    ret_contrib = (sub["return_contribution"].to_numpy() * 100.0)[::-1]
    risk_contrib = sub["risk_contribution_pct"].to_numpy()[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].barh(tickers, ret_contrib, color="#2ca02c")
    axes[0].set_xlabel("Contribución al retorno esperado (%)")
    axes[0].grid(True, axis="x", alpha=0.3)

    axes[1].barh(tickers, risk_contrib, color="#d62728")
    axes[1].set_xlabel("Contribución a la varianza (%)")
    axes[1].grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura guardada: %s (rebalanceo %s)", out_path.name, reb[:10])


def plot_ig_importance(csv_path: Path, dataset: str, out_path: Path, top_n: int = 15) -> None:
    imp = pl.read_csv(csv_path).head(top_n)
    names = imp["feature"].to_list()[::-1]
    vals = imp["importance_abs_mean"].to_numpy()[::-1]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.barh(names, vals, color="#0096CC")
    ax.set_xlabel("Importancia media |IG|")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura guardada: %s", out_path.name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Genera las figuras de explicabilidad de la memoria.")
    parser.add_argument("--config", type=str, default="top10_long_only",
                        help="Configuracion de cartera para trazabilidad/descomposición.")
    parser.add_argument("--only-ig", action="store_true",
                        help="Genera únicamente las figuras de Integrated Gradients "
                             "(no toca trazabilidad ni descomposición).")
    parser.add_argument("--skip-ig", action="store_true",
                        help="No regenera las figuras de Integrated Gradients "
                             "(solo trazabilidad y descomposición).")
    args = parser.parse_args(argv)

    root = Path.cwd()
    images_dir = root / "reports" / "Memoria" / "images" / "explainability"

    for dataset, horizon in CASES.items():
        adir = _artifact_dir(root, dataset, horizon)

        # IG (para ambos casos, salvo que se pida omitirla)
        if not args.skip_ig:
            ig_csv = adir / "ig_global_importance.csv"
            if ig_csv.exists():
                plot_ig_importance(ig_csv, dataset, images_dir / f"ig_importance_{dataset}.pdf")
            else:
                logger.warning("No existe %s", ig_csv)

        if args.only_ig:
            continue

        # Trazabilidad y descomposición: solo para el caso market (el que ilustra
        # la memoria), pero se generan para el dataset del caso por consistencia.
        trace_p = adir / "traceability.parquet"
        decomp_p = adir / "markowitz_decomposition.parquet"
        chosen_reb: str | None = None
        if trace_p.exists():
            chosen_reb = plot_traceability(pl.read_parquet(trace_p), args.config,
                                           images_dir / f"traceability_{dataset}.pdf")
        if decomp_p.exists():
            plot_markowitz_decomposition(pl.read_parquet(decomp_p), args.config,
                                         images_dir / f"markowitz_decomposition_{dataset}.pdf",
                                         reb=chosen_reb)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
