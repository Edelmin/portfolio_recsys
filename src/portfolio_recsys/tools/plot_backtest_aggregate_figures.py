"""Genera las figuras de ANALISIS AGREGADO de backtests para la memoria.

A diferencia de ``pr-portfolio-figures`` (que dibuja curvas de equity y Sharpe de
la configuracion top-10), este CLI produce las vistas *generalistas* que resumen
las ~70 configuraciones por horizonte de un vistazo, sin entrar caso por caso.
Es la version "materializada" del analisis interactivo del notebook
``notebooks/8.5 Analisis exhaustivo de backtests.ipynb``.

Fuente de datos: el CSV plano ``backtest_metrics.csv`` que produce
``pr-backtest-csv`` en::

    data/06_reporting/portfolio/{window}/{segmentation}/backtest_metrics.csv

Figuras generadas (PDF, para la memoria), por metrica (Sharpe y Sortino):

1. ``backtest_excess_heatmap_{metric}_{tipo}.pdf``: diferencial de la metrica
   respecto al S&P 500 por configuracion y horizonte (verde = bate al indice).
2. ``backtest_box_{metric}_{tipo}.pdf``: distribucion de la metrica por horizonte
   (robustez), con el S&P 500 como referencia.
3. ``backtest_marginal_{metric}.pdf``: valor medio de la metrica por nivel de cada
   eje de diseno (dataset, estrategia, top-K, direccion) -> "que decision pesa mas".
4. ``backtest_pct_beat_{metric}.pdf``: porcentaje de las 64 configuraciones del
   modelo que superan al S&P 500 por horizonte y conjunto.
5. ``backtest_risk_return_{tipo}.pdf``: dispersion volatilidad-retorno de todas
   las configuraciones, con el S&P 500 marcado (independiente de la metrica).

Las referencias (equiponderada, benchmarks, oracle) se excluyen del conjunto de
configuraciones "del modelo"; el S&P 500 se usa como referencia.

Uso::

    uv run pr-backtest-aggregate-figures
    uv run pr-backtest-aggregate-figures --window w20 --segmentation unified
    uv run pr-backtest-aggregate-figures --metrics sharpe
    uv run pr-backtest-aggregate-figures --dataset-type market
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from portfolio_recsys.paths import (
    PORTFOLIO_REPORT_ROOT,
    Segmentation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pr-backtest-aggregate-figures")

# ─────────────────────────────────────────────────────────────────────────────
# Convenciones de estilo (coherentes con las otras figuras de la memoria)
# ─────────────────────────────────────────────────────────────────────────────
HORIZON_ORDER = ["1d", "1w", "2w", "1m", "2m", "3m", "6m"]
HORIZON_LABELS = {
    "1d": "1 día", "1w": "1 semana", "2w": "2 semanas", "1m": "1 mes",
    "2m": "2 meses", "3m": "3 meses", "6m": "6 meses", "1y": "1 año",
}
HORIZON_SHORT = {
    "1d": "1 día", "1w": "1 sem", "2w": "2 sem", "1m": "1 mes",
    "2m": "2 mes", "3m": "3 mes", "6m": "6 mes", "1y": "1 año",
}
METRIC_COL = {"sharpe": "sharpe_ratio", "sortino": "sortino_ratio"}
METRIC_LABEL = {"sharpe": "Sharpe", "sortino": "Sortino"}

# Tope del eje Y de la version recortada del boxplot de Sortino, por conjunto.
# enriched tiene outliers muy altos (hasta ~38) y necesita mas rango que market,
# cuyos valores son mas bajos y quedarian aplastados con un tope alto.
SORTINO_YLIM_BY_DATASET = {"market": 4.0, "enriched": 10.0}

# Paleta por dataset (consistente con el resto de la memoria)
DATASET_COLOR = {"market": "#1f77b4", "enriched": "#ff7f0e"}
# Colormaps secuenciales por dataset para distinguir horizontes en la dispersion,
# coherentes con el color de identidad de cada conjunto (azules para market,
# naranjas para enriched).
SCATTER_CMAP_BY_DATASET = {"market": "Blues", "enriched": "Oranges"}
BENCH_COLOR = "#111111"
POSITIVE_BAR = "#2ca02c"

STRATEGY_ORDER_LABEL = {
    "max_return": "Máx. rent.",
    "min_variance": "Mín. var.",
    "mean_variance": "Media-var.",
    "target_return": "Ret. obj.",
}


def _project_root() -> Path:
    return Path.cwd()


def _csv_path(window: str, segmentation: Segmentation) -> Path:
    """Ruta del CSV plano de metricas (una carpeta por encima de los horizontes)."""
    return (
        PORTFOLIO_REPORT_ROOT
        / window
        / segmentation.path_fragment
        / "backtest_metrics.csv"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Carga y parseo (misma logica que el notebook 8.5)
# ─────────────────────────────────────────────────────────────────────────────
def _parse_config(name: str) -> dict:
    """Descompone un nombre de configuracion en sus dimensiones."""
    out = {
        "config": name, "top_k": None, "strategy": None,
        "lambda": None, "direction": None, "is_reference": False,
    }
    if name.startswith("benchmark"):
        out["strategy"] = "benchmark"
        out["is_reference"] = True
        return out
    m = re.match(r"top(\d+)_(.+)", name)
    if not m:
        return out
    out["top_k"] = int(m.group(1))
    rest = m.group(2)
    if rest == "equal_weight":
        out.update(strategy="equal_weight", direction="long_only", is_reference=True)
        return out
    if rest == "oracle":
        out.update(strategy="oracle", is_reference=True)
        return out
    if rest.endswith("long_only"):
        out["direction"] = "long_only"
        rest = rest[: -len("_long_only")] if rest != "long_only" else ""
    elif rest.endswith("long_short"):
        out["direction"] = "long_short"
        rest = rest[: -len("_long_short")] if rest != "long_short" else ""
    if rest == "":
        out["strategy"] = "max_return"
    elif rest == "minvar":
        out["strategy"] = "min_variance"
    elif rest == "target":
        out["strategy"] = "target_return"
    elif rest.startswith("meanvar_l"):
        out["strategy"] = "mean_variance"
        out["lambda"] = int(rest.replace("meanvar_l", ""))
    return out


def _load(window: str, segmentation: Segmentation) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Devuelve (df completo parseado, solo-modelo, S&P 500 por horizonte/dataset)."""
    csv_path = _csv_path(window, segmentation)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No existe {csv_path}. Genera antes el CSV con: pr-backtest-csv "
            f"--window {window} --segmentation {segmentation.dataset_token}"
        )
    raw = pl.read_csv(csv_path)
    dims = pl.DataFrame([_parse_config(c) for c in raw["config"].to_list()]).unique(
        subset=["config"]
    )
    df = raw.join(dims, on="config", how="left")
    modelo = df.filter(~pl.col("is_reference") & (pl.col("strategy") != "oracle"))
    return df, modelo, df


def _bench_metric(df: pl.DataFrame, metric_col: str) -> pl.DataFrame:
    return (
        df.filter(pl.col("config") == "benchmark_sp500")
        .select(["horizon", "dataset", metric_col])
        .rename({metric_col: "bench"})
    )


def _present_horizons(df: pl.DataFrame) -> list[str]:
    have = set(df["horizon"].unique().to_list())
    return [h for h in HORIZON_ORDER if h in have]


# ─────────────────────────────────────────────────────────────────────────────
# Figuras
# ─────────────────────────────────────────────────────────────────────────────
def plot_excess_heatmap(
    modelo: pl.DataFrame, df: pl.DataFrame, dataset: str, metric: str,
    out_path: Path, top_n: int | None = 25,
) -> bool:
    metric_col = METRIC_COL[metric]
    mlab = METRIC_LABEL[metric]
    bench = _bench_metric(df, metric_col).filter(pl.col("dataset") == dataset)
    sub = modelo.filter(pl.col("dataset") == dataset).join(
        bench.select(["horizon", "bench"]), on="horizon", how="left"
    )
    sub = sub.with_columns((pl.col(metric_col) - pl.col("bench")).alias("excess"))
    piv = sub.pivot(values="excess", index="config", on="horizon", aggregate_function="first")
    cols = [h for h in HORIZON_ORDER if h in piv.columns]
    if not cols:
        logger.warning("Sin columnas de horizonte para %s/%s", dataset, metric)
        return False
    arr = piv.select(cols).to_numpy().astype(float)
    order = np.argsort(-np.nanmean(arr, axis=1))
    labels = [piv["config"][int(i)] for i in order]
    arr = arr[order]
    if top_n:
        labels, arr = labels[:top_n], arr[:top_n]
    vmax = float(np.nanmax(np.abs(arr))) or 1.0

    fig, ax = plt.subplots(figsize=(9, max(4.0, len(labels) * 0.28)))
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(cols)), [HORIZON_SHORT[c] for c in cols])
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    fig.colorbar(im, ax=ax, label=f"{mlab} $-$ {mlab}(S&P 500)", shrink=0.6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura escrita: %s", out_path.name)
    return True


def plot_box(
    modelo: pl.DataFrame, df: pl.DataFrame, dataset: str, metric: str, out_path: Path,
    y_max: float | None = None,
) -> bool:
    metric_col = METRIC_COL[metric]
    mlab = METRIC_LABEL[metric]
    horizons = _present_horizons(modelo.filter(pl.col("dataset") == dataset))
    if not horizons:
        return False
    sub = modelo.filter(pl.col("dataset") == dataset)
    data = [sub.filter(pl.col("horizon") == h)[metric_col].drop_nulls().to_list() for h in horizons]

    def _ref_series(config_name: str) -> list[float]:
        ref = (
            df.filter((pl.col("config") == config_name) & (pl.col("dataset") == dataset))
            .select(["horizon", metric_col])
        )
        vals = []
        for h in horizons:
            r = ref.filter(pl.col("horizon") == h)[metric_col]
            v = r[0] if len(r) else None
            vals.append(float(v) if v is not None else np.nan)
        return vals

    sp_vals = _ref_series("benchmark_sp500")
    mktas_vals = _ref_series("benchmark_sp500_riskfree")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bp = ax.boxplot(data, tick_labels=[HORIZON_SHORT[h] for h in horizons], showmeans=True)
    sp_line, = ax.plot(range(1, len(horizons) + 1), sp_vals, color=BENCH_COLOR, linestyle="--",
                       marker="o", markersize=4, linewidth=1.6, label="S&P 500 (buy & hold)")
    mktas_line, = ax.plot(range(1, len(horizons) + 1), mktas_vals, color="#7f7f7f", linestyle=":",
                          marker="s", markersize=4, linewidth=1.6, label="S&P 500 + activo seguro")
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.set_ylabel(mlab, fontsize=9)
    ax.set_xlabel("Horizonte", fontsize=9)
    if y_max is not None:
        ax.set_ylim(top=y_max)
    ax.grid(True, alpha=0.3)

    # Entradas de leyenda para los marcadores del propio diagrama de caja: la
    # media (triangulo, showmeans) y la mediana (linea central), que matplotlib
    # no incluye en la leyenda por defecto.
    from matplotlib.lines import Line2D
    mean_marker = bp["means"][0] if bp.get("means") else None
    median_line = bp["medians"][0] if bp.get("medians") else None
    handles = [sp_line, mktas_line]
    if mean_marker is not None:
        handles.append(Line2D([], [], linestyle="none",
                              marker=mean_marker.get_marker(),
                              markerfacecolor=mean_marker.get_markerfacecolor(),
                              markeredgecolor=mean_marker.get_markeredgecolor(),
                              markersize=6, label="Media"))
    if median_line is not None:
        handles.append(Line2D([], [], color=median_line.get_color(),
                              linewidth=median_line.get_linewidth(), label="Mediana"))
    ax.legend(handles=handles, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura escrita: %s", out_path.name)
    return True


def _marginal(modelo: pl.DataFrame, eje: str, metric_col: str) -> pl.DataFrame:
    return (
        modelo.group_by(eje)
        .agg(pl.col(metric_col).mean().alias("valor_medio"), pl.len().alias("n"))
        .sort("valor_medio", descending=True)
    )


def plot_marginal(modelo: pl.DataFrame, metric: str, out_path: Path) -> bool:
    metric_col = METRIC_COL[metric]
    mlab = METRIC_LABEL[metric]
    ejes = ["dataset", "strategy", "top_k", "direction"]
    titulos = {"dataset": "Conjunto", "strategy": "Estrategia",
               "top_k": "Top-K", "direction": "Dirección"}
    fig, axes = plt.subplots(1, len(ejes), figsize=(14, 3.8))
    for ax, eje in zip(axes, ejes):
        m = _marginal(modelo, eje, metric_col).drop_nulls("valor_medio")
        raw_labels = m[eje].to_list()
        if eje == "strategy":
            labels = [STRATEGY_ORDER_LABEL.get(x, str(x)) for x in raw_labels]
        elif eje == "top_k":
            labels = [f"top{int(x)}" for x in raw_labels]
        else:
            labels = [str(x) for x in raw_labels]
        ax.bar(labels, m["valor_medio"].to_list(), color=POSITIVE_BAR, alpha=0.8)
        ax.set_title(titulos[eje], fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel(f"{mlab} medio", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura escrita: %s", out_path.name)
    return True


def plot_pct_beat(modelo: pl.DataFrame, df: pl.DataFrame, metric: str, out_path: Path) -> bool:
    metric_col = METRIC_COL[metric]
    mlab = METRIC_LABEL[metric]
    bench = _bench_metric(df, metric_col)
    tab = (
        modelo.join(bench, on=["horizon", "dataset"], how="left")
        .with_columns((pl.col(metric_col) > pl.col("bench")).alias("bate"))
        .group_by(["horizon", "dataset"])
        .agg((pl.col("bate").mean() * 100).alias("pct"))
    )
    horizons = _present_horizons(modelo)
    x = np.arange(len(horizons))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4))
    for k, ds in enumerate(["market", "enriched"]):
        vals = []
        for h in horizons:
            r = tab.filter((pl.col("horizon") == h) & (pl.col("dataset") == ds))["pct"]
            vals.append(float(r[0]) if len(r) else 0.0)
        ax.bar(x + (k - 0.5) * width, vals, width, color=DATASET_COLOR[ds], label=ds)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x, [HORIZON_SHORT[h] for h in horizons])
    ax.set_ylabel(f"% de configuraciones que baten\nal S&P 500 en {mlab}", fontsize=9)
    ax.set_xlabel("Horizonte", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura escrita: %s", out_path.name)
    return True


def plot_risk_return(modelo: pl.DataFrame, df: pl.DataFrame, dataset: str, out_path: Path) -> bool:
    """Dispersion retorno-riesgo con isolineas de Sharpe y color por horizonte.

    Cada punto es una configuracion, coloreada segun su horizonte con una escala
    secuencial coherente con el color de identidad del conjunto. Se anaden como
    referencia las isolineas de Sharpe (rectas y = s*x que pasan por el origen,
    pues Sharpe = retorno/volatilidad con tasa libre de riesgo nula) y, resaltada,
    la recta correspondiente al Sharpe del S&P 500: los puntos por encima de esa
    recta baten al indice en rendimiento ajustado al riesgo.
    """
    sub = modelo.filter(pl.col("dataset") == dataset)
    if sub.height == 0:
        return False

    from matplotlib.lines import Line2D

    horizons = _present_horizons(sub)
    cmap = plt.get_cmap(SCATTER_CMAP_BY_DATASET.get(dataset, "Blues"))
    # Recortar el rango del colormap para evitar tonos casi blancos (poco visibles).
    h_colors = {h: cmap(0.35 + 0.6 * i / max(1, len(horizons) - 1))
                for i, h in enumerate(horizons)}

    fig, ax = plt.subplots(figsize=(8, 6))

    # --- Isolineas de Sharpe (rectas por el origen de pendiente s) ---
    vmax = float(sub["annualized_volatility"].max() or 0.0) * 1.05
    xs = np.array([0.0, vmax if vmax > 0 else 1.0])
    for s in (0.5, 1.0, 1.5, 2.0):
        ax.plot(xs, s * xs, color="gray", linewidth=0.8, linestyle=":", alpha=0.6, zorder=1)
        y_lab = s * xs[1]
        ax.annotate(f"Sharpe {s:g}", xy=(xs[1], y_lab), fontsize=7, color="gray",
                    ha="right", va="bottom", alpha=0.8)

    # --- Recta del Sharpe del S&P 500 (frontera de batir al indice) ---
    spd = df.filter((pl.col("config") == "benchmark_sp500") & (pl.col("dataset") == dataset))
    sp_sharpe = None
    if spd.height:
        sp_sharpe = float(spd["sharpe_ratio"].drop_nulls().mean())
        ax.plot(xs, sp_sharpe * xs, color=BENCH_COLOR, linewidth=1.8, linestyle="--",
                zorder=2, label=f"Sharpe S&P 500 (med. {sp_sharpe:.2f})")

    # --- Puntos coloreados por horizonte ---
    for h in horizons:
        d = sub.filter(pl.col("horizon") == h)
        ax.scatter(d["annualized_volatility"].to_list(), d["annualized_return"].to_list(),
                   s=22, alpha=0.75, color=h_colors[h], edgecolors="none", zorder=3)

    # --- S&P 500 como estrella (media de sus puntos) ---
    if spd.height:
        ax.scatter([float(spd["annualized_volatility"].mean())],
                   [float(spd["annualized_return"].mean())],
                   s=130, c=BENCH_COLOR, marker="*", zorder=5)

    ax.axhline(0, color="gray", linewidth=0.6)
    ax.set_xlim(left=0)
    ax.set_xlabel("Volatilidad anualizada", fontsize=9)
    ax.set_ylabel("Retorno anualizado", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Leyenda: horizontes (por color) + referencia del indice.
    handles = [Line2D([], [], marker="o", linestyle="none", markersize=6,
                      markerfacecolor=h_colors[h], markeredgecolor="none",
                      label=HORIZON_LABELS.get(h, h)) for h in horizons]
    handles.append(Line2D([], [], color=BENCH_COLOR, marker="*", linestyle="none",
                          markersize=11, label="S&P 500"))
    if sp_sharpe is not None:
        handles.append(Line2D([], [], color=BENCH_COLOR, linestyle="--", linewidth=1.8,
                              label=f"Sharpe S&P 500 ({sp_sharpe:.2f})"))
    ax.legend(handles=handles, fontsize=8, title="Horizonte", title_fontsize=8, loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura escrita: %s", out_path.name)
    return True


def plot_return_drawdown(modelo: pl.DataFrame, df: pl.DataFrame, dataset: str, out_path: Path) -> bool:
    """Dispersion retorno vs maximo drawdown, con color por horizonte.

    Analoga a la de retorno-volatilidad, pero usando el maximo drawdown como
    medida de riesgo (mas tangible para el inversor: la mayor caida desde
    maximos). Se marcan con lineas de referencia el retorno y el drawdown del
    S&P 500, de modo que el cuadrante superior-izquierdo (mas retorno con menor
    caida que el indice) es el favorable.
    """
    sub = modelo.filter(pl.col("dataset") == dataset)
    if sub.height == 0:
        return False

    from matplotlib.lines import Line2D

    horizons = _present_horizons(sub)
    cmap = plt.get_cmap(SCATTER_CMAP_BY_DATASET.get(dataset, "Blues"))
    h_colors = {h: cmap(0.35 + 0.6 * i / max(1, len(horizons) - 1))
                for i, h in enumerate(horizons)}

    fig, ax = plt.subplots(figsize=(8, 6))

    # Referencias del S&P 500 (media sobre horizontes): retorno y drawdown.
    spd = df.filter((pl.col("config") == "benchmark_sp500") & (pl.col("dataset") == dataset))
    sp_ret = sp_dd = None
    if spd.height:
        sp_ret = float(spd["annualized_return"].drop_nulls().mean())
        sp_dd = float(spd["max_drawdown"].drop_nulls().mean())
        ax.axhline(sp_ret, color=BENCH_COLOR, linestyle="--", linewidth=1.4, alpha=0.8, zorder=2)
        ax.axvline(sp_dd, color=BENCH_COLOR, linestyle=":", linewidth=1.4, alpha=0.8, zorder=2)

    # Puntos coloreados por horizonte (drawdown en X, retorno en Y).
    for h in horizons:
        d = sub.filter(pl.col("horizon") == h)
        ax.scatter(d["max_drawdown"].to_list(), d["annualized_return"].to_list(),
                   s=22, alpha=0.75, color=h_colors[h], edgecolors="none", zorder=3)

    # S&P 500 como estrella.
    if spd.height:
        ax.scatter([sp_dd], [sp_ret], s=130, c=BENCH_COLOR, marker="*", zorder=5)

    ax.axhline(0, color="gray", linewidth=0.6)
    ax.set_xlim(left=0)
    ax.set_xlabel("Máximo drawdown", fontsize=9)
    ax.set_ylabel("Retorno anualizado", fontsize=9)
    ax.grid(True, alpha=0.3)

    handles = [Line2D([], [], marker="o", linestyle="none", markersize=6,
                      markerfacecolor=h_colors[h], markeredgecolor="none",
                      label=HORIZON_LABELS.get(h, h)) for h in horizons]
    handles.append(Line2D([], [], color=BENCH_COLOR, marker="*", linestyle="none",
                          markersize=11, label="S&P 500"))
    if sp_ret is not None:
        handles.append(Line2D([], [], color=BENCH_COLOR, linestyle="--", linewidth=1.4,
                              label=f"Retorno S&P 500 (med. {sp_ret:.2f})"))
        handles.append(Line2D([], [], color=BENCH_COLOR, linestyle=":", linewidth=1.4,
                              label=f"Drawdown S&P 500 (med. {sp_dd:.2f})"))
    ax.legend(handles=handles, fontsize=8, title="Horizonte", title_fontsize=8, loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura escrita: %s", out_path.name)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Genera las figuras de analisis agregado de backtests para la memoria "
            "(heatmaps de diferencial, boxplots, marginal por eje, % que baten y dispersion "
            "retorno-riesgo), en Sharpe y Sortino."
        ),
    )
    parser.add_argument("--window", type=str, default="w20",
                        help="Ventana temporal (por defecto w20).")
    parser.add_argument("--segmentation", choices=["unified", "by_sector"],
                        default="unified", help="Segmentacion (por defecto unified).")
    parser.add_argument("--sector", type=str, default=None,
                        help="Sector canonico (si --segmentation by_sector).")
    parser.add_argument("--dataset-type", choices=["market", "enriched"], default=None,
                        help="Conjunto de datos. Si se omite, genera market y enriched.")
    parser.add_argument("--metrics", type=str, default="sharpe,sortino",
                        help="Metricas separadas por comas: sharpe,sortino.")
    parser.add_argument("--top-n", type=int, default=25,
                        help="Nº de configuraciones a mostrar en los heatmaps (por defecto 25).")
    parser.add_argument("--sortino-ylim", type=float, default=10.0,
                        help="Tope del eje Y para la version recortada del boxplot de Sortino "
                             "(por defecto 10). Genera un fichero adicional _clip.pdf.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    segmentation = (
        Segmentation.unified()
        if args.segmentation == "unified"
        else Segmentation.for_sector(args.sector)
    )

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    for m in metrics:
        if m not in METRIC_COL:
            parser.error(f"Metrica desconocida: {m!r} (usa: sharpe, sortino)")

    dataset_types = [args.dataset_type] if args.dataset_type else ["market", "enriched"]

    try:
        df, modelo, _ = _load(args.window, segmentation)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    images_dir = _project_root() / "reports" / "Memoria" / "images"
    ok_any = False

    # Figuras por metrica y dataset (heatmap de exceso, boxplot)
    for metric in metrics:
        for dtype in dataset_types:
            ok_any |= plot_excess_heatmap(
                modelo, df, dtype, metric,
                images_dir / f"backtest_excess_heatmap_{metric}_{dtype}.pdf",
                top_n=args.top_n,
            )
            ok_any |= plot_box(
                modelo, df, dtype, metric,
                images_dir / f"backtest_box_{metric}_{dtype}.pdf",
            )
            # Version del Sortino con eje recortado (los outliers extremos
            # impiden leer los boxplots a escala completa). El tope se ajusta por
            # conjunto: enriched necesita mas rango (outliers hasta ~38) que
            # market, cuyos valores son mas bajos. Ver SORTINO_YLIM_BY_DATASET.
            if metric == "sortino":
                y_cap = SORTINO_YLIM_BY_DATASET.get(dtype, args.sortino_ylim)
                if y_cap is not None:
                    ok_any |= plot_box(
                        modelo, df, dtype, metric,
                        images_dir / f"backtest_box_{metric}_{dtype}_clip.pdf",
                        y_max=y_cap,
                    )
        # Figuras por metrica (agregan ambos datasets): marginal y % que baten
        ok_any |= plot_marginal(
            modelo, metric, images_dir / f"backtest_marginal_{metric}.pdf",
        )
        ok_any |= plot_pct_beat(
            modelo, df, metric, images_dir / f"backtest_pct_beat_{metric}.pdf",
        )

    # Dispersiones (independientes de la metrica), por dataset:
    # retorno-riesgo (volatilidad) y retorno-drawdown.
    for dtype in dataset_types:
        ok_any |= plot_risk_return(
            modelo, df, dtype, images_dir / f"backtest_risk_return_{dtype}.pdf",
        )
        ok_any |= plot_return_drawdown(
            modelo, df, dtype, images_dir / f"backtest_return_drawdown_{dtype}.pdf",
        )

    return 0 if ok_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
