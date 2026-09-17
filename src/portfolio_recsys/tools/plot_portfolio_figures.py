"""Genera las figuras del capitulo de Evaluacion a partir de los backtests.

Lee los ``backtest_results_{market,enriched}.json`` de cada horizonte y produce
tres figuras PDF para la memoria (con sufijo del conjunto de datos):

1. ``equity_curves_all_horizons_{tipo}.pdf``: rejilla de curvas de equity (una
   por horizonte) para la configuracion top-10, comparando las variantes del
   modelo, la equiponderada y el S&P 500 como referencia.
2. ``sharpe_comparison_all_horizons_{tipo}.pdf``: ratio de Sharpe por horizonte
   y variante (top-10), con el S&P 500 como referencia.
3. ``strategy_comparison_all_horizons_{tipo}.pdf``: ratio de Sharpe por
   estrategia de optimizacion y horizonte (top-10, long-only).

La prediccion perfecta (oracle) se excluye de todas las figuras. El horizonte 1y
no se incluye por defecto.

Uso:
    uv run pr-portfolio-figures --dataset-type market
    uv run pr-portfolio-figures --dataset-type enriched --window w20
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from portfolio_recsys.paths import Segmentation, portfolio_report_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("pr-portfolio-figures")

DEFAULT_HORIZONS = ["1d", "1w", "2w", "1m", "2m", "3m", "6m"]
HORIZON_LABELS = {
    "1d": "1 día", "1w": "1 semana", "2w": "2 semanas", "1m": "1 mes",
    "2m": "2 meses", "3m": "3 meses", "6m": "6 meses", "1y": "1 año",
}

# Estilo por serie (color y etiqueta legible)
SERIES_STYLE = {
    "long_only":   ("#1f77b4", "Long-only"),
    "long_short":  ("#ff7f0e", "Long-short"),
    "equal_weight": ("#2ca02c", "Equiponderada"),
    "oracle":      ("#d62728", "Predicción perfecta"),
}
BENCH_STYLE = ("#111111", "S&P 500 (referencia)")

# Estrategias de optimizacion (sufijo de clave -> color, etiqueta). Para la
# comparacion de estrategias a top-K y direccion fijos.
STRATEGY_STYLE = {
    "long_only":            ("#1f77b4", "Máx. retorno"),
    "minvar_long_only":     ("#2ca02c", "Mínima varianza"),
    "meanvar_l1_long_only": ("#ffbb78", "Media-var. λ=1"),
    "meanvar_l3_long_only": ("#ff7f0e", "Media-var. λ=3"),
    "meanvar_l10_long_only": ("#d62728", "Media-var. λ=10"),
    "target_long_only":     ("#9467bd", "Retorno objetivo"),
    "equal_weight":         ("#8c564b", "Equiponderada"),
}


def _project_root() -> Path:
    return Path.cwd()


def _load_backtest(
    segmentation: Segmentation, window: str, horizon: str,
    dataset_type: str | None = None,
) -> dict | None:
    """Carga el backtest de un horizonte.

    Si ``dataset_type`` es ``market`` o ``enriched``, lee el fichero separado
    ``backtest_results_{dataset_type}.json``. Si es None, usa el fichero
    ``backtest_results.json`` (modo de un unico dataset combinado).
    """
    base = portfolio_report_dir(window, segmentation, horizon)
    if dataset_type in ("market", "enriched"):
        p = base / f"backtest_results_{dataset_type}.json"
    else:
        p = base / "backtest_results.json"
    if not p.exists():
        logger.warning("No existe %s", p)
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _sharpe(cfg: dict) -> float | None:
    return cfg.get("metrics", {}).get("sharpe_ratio")


def plot_equity_curves(backtests: dict[str, dict], top_k: int, out_path: Path) -> None:
    """Rejilla de curvas de equity, una por horizonte.

    Se emplea una rejilla de doble resolucion horizontal (2*ncols columnas) para
    poder centrar el ultimo panel cuando el numero de horizontes es impar: cada
    panel ocupa dos columnas de la rejilla fina y, si sobra un hueco en la ultima
    fila, el panel restante se desplaza para quedar centrado.
    """
    horizons = list(backtests.keys())
    n = len(horizons)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(12, 3.2 * nrows))
    gs = fig.add_gridspec(nrows, 2 * ncols)

    def _panel_span(idx: int):
        """Devuelve (fila, col_ini, col_fin) en la rejilla fina para el panel idx."""
        row = idx // ncols
        col = idx % ncols
        is_last_row = row == nrows - 1
        items_last_row = n - (nrows - 1) * ncols
        if is_last_row and items_last_row < ncols:
            # centrar los paneles sueltos de la ultima fila
            offset = (2 * ncols - 2 * items_last_row) // 2
            start = offset + 2 * col
        else:
            start = 2 * col
        return row, start, start + 2

    for idx, horizon in enumerate(horizons):
        row, c0, c1 = _panel_span(idx)
        ax = fig.add_subplot(gs[row, c0:c1])
        bt = backtests[horizon]
        configs = bt.get("configs", {})

        # Se excluye el oracle de las curvas: a horizontes largos alcanza miles
        # de millones y aplasta la escala del resto de series.
        for variant, (color, label) in SERIES_STYLE.items():
            if variant == "oracle":
                continue
            key = f"top{top_k}_{variant}"
            if key not in configs:
                continue
            ec = configs[key].get("equity_curve", [])
            if not ec:
                continue
            ax.plot(range(len(ec)), ec, color=color, label=label, linewidth=1.4)

        # Benchmark S&P 500 (referencia principal, trazo mas grueso y oscuro)
        if "benchmark_sp500" in configs:
            ec = configs["benchmark_sp500"].get("equity_curve", [])
            if ec:
                ax.plot(range(len(ec)), ec, color=BENCH_STYLE[0], label=BENCH_STYLE[1],
                        linewidth=2.0, linestyle="--")

        ax.set_yscale("log")
        ax.set_title(f"{HORIZON_LABELS.get(horizon, horizon)}", fontsize=10)
        ax.set_xlabel("Periodo de rebalanceo", fontsize=8)
        ax.set_ylabel("Valor cartera (log)", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()

    # Leyenda unica
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.03, 1, 1.0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura escrita: %s", out_path)


def plot_sharpe_comparison(backtests: dict[str, dict], top_k: int, out_path: Path,
                           truncate: float = 6.0) -> None:
    """Barras de Sharpe por horizonte y variante del modelo + S&P 500.

    Se excluye la predicción perfecta (oracle) porque a horizontes largos su
    Sharpe alcanza valores de miles (escala incomparable) que harían ilegible la
    figura.
    """
    horizons = list(backtests.keys())
    series = ["long_only", "long_short", "equal_weight"]

    # Recolectar valores
    data = {s: [] for s in series}
    bench = []
    for horizon in horizons:
        configs = backtests[horizon].get("configs", {})
        for s in series:
            data[s].append(_sharpe(configs.get(f"top{top_k}_{s}", {})))
        bench.append(_sharpe(configs.get("benchmark_sp500", {})))

    x = np.arange(len(horizons))
    all_series = series + ["benchmark"]
    width = 0.8 / len(all_series)
    fig, ax = plt.subplots(figsize=(12, 5))

    def _plot_bars(offset_idx, values, color, label):
        pos = x + (offset_idx - (len(all_series) - 1) / 2) * width
        clipped = []
        for v in values:
            clipped.append(np.nan if v is None else min(v, truncate))
        bars = ax.bar(pos, clipped, width, color=color, label=label)
        # Anotar valores truncados
        for xi, v in zip(pos, values):
            if v is not None and v > truncate:
                ax.text(xi, truncate + 0.05, f"{v:.1f}", ha="center", va="bottom",
                        fontsize=7, rotation=90)
        return bars

    for i, s in enumerate(series):
        color, label = SERIES_STYLE[s]
        _plot_bars(i, data[s], color, label)
    _plot_bars(len(series), bench, BENCH_STYLE[0], BENCH_STYLE[1])

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([HORIZON_LABELS.get(h, h) for h in horizons])
    ax.set_ylabel("Ratio de Sharpe")
    ax.set_title(f"Ratio de Sharpe por horizonte y variante (top-{top_k}; "
                 f"valores > {truncate:.0f} anotados)")
    ax.set_ylim(top=truncate + 1.0)
    ax.legend(fontsize=9, ncol=5, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura escrita: %s", out_path)


def plot_strategy_comparison(backtests: dict[str, dict], top_k: int, out_path: Path,
                             truncate: float = 3.0) -> None:
    """Compara las estrategias de optimizacion (top-K, long-only) por horizonte.

    Barras de ratio de Sharpe por horizonte para cada estrategia: maxima
    rentabilidad, minima varianza, media-varianza (lambda=1,3,10), retorno
    objetivo y equiponderada, con el S&P 500 como referencia. El Sharpe es
    comparable entre estrategias (adimensional) aunque los retornos absolutos no
    lo sean a horizontes largos.
    """
    horizons = list(backtests.keys())
    series = list(STRATEGY_STYLE.keys())
    all_series = series + ["benchmark"]

    x = np.arange(len(horizons))
    width = 0.8 / len(all_series)
    fig, ax = plt.subplots(figsize=(13, 5.5))

    def _sharpe_for(configs, suffix):
        return _sharpe(configs.get(f"top{top_k}_{suffix}", {}))

    def _bars(offset_idx, values, color, label):
        pos = x + (offset_idx - (len(all_series) - 1) / 2) * width
        clipped = [np.nan if v is None else min(v, truncate) for v in values]
        ax.bar(pos, clipped, width, color=color, label=label)
        for xi, v in zip(pos, values):
            if v is not None and v > truncate:
                ax.text(xi, truncate + 0.02, f"{v:.1f}", ha="center", va="bottom",
                        fontsize=7, rotation=90)

    for i, suffix in enumerate(series):
        color, label = STRATEGY_STYLE[suffix]
        vals = [_sharpe_for(backtests[h]["configs"], suffix) for h in horizons]
        _bars(i, vals, color, label)
    # S&P 500 referencia
    bench = [_sharpe(backtests[h]["configs"].get("benchmark_sp500", {})) for h in horizons]
    _bars(len(series), bench, BENCH_STYLE[0], BENCH_STYLE[1])

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([HORIZON_LABELS.get(h, h) for h in horizons])
    ax.set_ylabel("Ratio de Sharpe")
    ax.set_title(f"Comparación de estrategias de optimización por horizonte "
                 f"(top-{top_k}, long-only; Sharpe truncado a {truncate:.0f})")
    ax.set_ylim(top=truncate + 0.6)
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura escrita: %s", out_path)


def plot_cumulative_excess(backtests: dict[str, dict], top_k: int, out_path: Path) -> None:
    """Exceso de retorno acumulado de la cartera del modelo sobre el S&P 500.

    Para cada horizonte, calcula la diferencia entre la curva de equity de la
    configuracion top-K (variante long-only) y la del S&P 500 en buy & hold,
    ambas normalizadas a 1 al inicio, y la representa a lo largo de los periodos
    de rebalanceo. Una curva creciente indica que el modelo bate al indice de
    forma sostenida; una plana o decreciente, que la ventaja no es persistente.

    Responde a la pregunta de CONSISTENCIA TEMPORAL: no si el modelo bate al
    indice en promedio, sino si lo hace de forma sostenida en el tiempo.
    """
    horizons = list(backtests.keys())
    n = len(horizons)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows), squeeze=False)

    for idx, horizon in enumerate(horizons):
        ax = axes[idx // ncols][idx % ncols]
        configs = backtests[horizon].get("configs", {})
        model = configs.get(f"top{top_k}_long_only", {}).get("equity_curve", [])
        bench = configs.get("benchmark_sp500", {}).get("equity_curve", [])
        if not model or not bench:
            ax.axis("off")
            continue
        m = min(len(model), len(bench))
        model_n = np.array(model[:m], dtype=float) / model[0]
        bench_n = np.array(bench[:m], dtype=float) / bench[0]
        excess = (model_n - bench_n) * 100.0  # puntos porcentuales sobre el indice

        ax.plot(range(m), excess, color="#1f77b4", linewidth=1.6)
        ax.axhline(0, color=BENCH_STYLE[0], linewidth=1.0, linestyle="--")
        ax.fill_between(range(m), excess, 0, where=(excess >= 0), color="#2ca02c", alpha=0.15)
        ax.fill_between(range(m), excess, 0, where=(excess < 0), color="#d62728", alpha=0.15)
        ax.set_title(f"{HORIZON_LABELS.get(horizon, horizon)}", fontsize=10)
        ax.set_xlabel("Periodo de rebalanceo", fontsize=8)
        ax.set_ylabel("Exceso acum. vs S&P 500 (p.p.)", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(f"Exceso de retorno acumulado sobre el S&P 500 por horizonte (top-{top_k})",
                 fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0.02, 1, 0.99])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura escrita: %s", out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Genera figuras de carteras para la memoria.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--horizons", type=str, default=",".join(DEFAULT_HORIZONS))
    parser.add_argument("--truncate-sharpe", type=float, default=6.0)
    parser.add_argument("--window", type=str, default="w20",
                        help="Clave de ventana temporal (por defecto w20).")
    parser.add_argument("--segmentation", choices=["unified", "by_sector"],
                        default="unified", help="Segmentacion (por defecto unified).")
    parser.add_argument("--sector", type=str, default=None,
                        help="Sector canonico (si --segmentation by_sector).")
    parser.add_argument("--dataset-type", choices=["market", "enriched"], default=None,
                        help="Conjunto de datos: market o enriched. Si se indica, "
                             "lee backtest_results_{tipo}.json y sufija las figuras.")
    args = parser.parse_args(argv)

    segmentation = (
        Segmentation.unified()
        if args.segmentation == "unified"
        else Segmentation.for_sector(args.sector)
    )

    root = _project_root()
    images_dir = root / "reports" / "Memoria" / "images"

    horizons = [h.strip() for h in args.horizons.split(",") if h.strip()]
    backtests: dict[str, dict] = {}
    for h in horizons:
        bt = _load_backtest(segmentation, args.window, h, args.dataset_type)
        if bt is not None and "configs" in bt:
            backtests[h] = bt
    if not backtests:
        logger.error("No se cargo ningun backtest. Revisa window/segmentation/dataset-type.")
        return 1

    logger.info("Horizontes cargados: %s", ", ".join(backtests.keys()))

    # Sufijo del conjunto de datos en el nombre de fichero (si aplica).
    suffix = f"_{args.dataset_type}" if args.dataset_type else ""

    plot_equity_curves(backtests, args.top_k,
                       images_dir / f"equity_curves_all_horizons{suffix}.pdf")
    plot_sharpe_comparison(backtests, args.top_k,
                           images_dir / f"sharpe_comparison_all_horizons{suffix}.pdf",
                           truncate=args.truncate_sharpe)
    plot_strategy_comparison(backtests, args.top_k,
                             images_dir / f"strategy_comparison_all_horizons{suffix}.pdf",
                             truncate=3.0)
    plot_cumulative_excess(backtests, args.top_k,
                           images_dir / f"cumulative_excess_all_horizons{suffix}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
