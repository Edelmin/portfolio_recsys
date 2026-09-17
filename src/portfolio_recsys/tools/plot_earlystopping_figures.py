"""Genera las figuras de early stopping (curvas de aprendizaje) para la memoria.

Lee los historiales de entrenamiento por semilla persistidos por
``predict_test_split`` en::

    data/04_models/{window}/{segmentation}/{horizon}/test_seeds/
        {dataset_type}_seed{seed}_history.parquet

y produce, por conjunto de datos (market/enriched) y horizonte, una figura
independiente con DOS paneles apilados: arriba la curva de perdida de VALIDACION
por epoca (una por semilla, con la epoca del mejor checkpoint, la parada
temprana, marcada) y abajo la de ENTRENAMIENTO. Ambas curvas se separan en
paneles distintos porque miden el error sobre conjuntos diferentes y en escalas
que no son directamente comparables.

Salida (PDF, para la memoria)::

    reports/Memoria/images/early_stopping_{tipo}_{horizonte}.pdf

Uso::

    uv run pr-earlystopping-figures --dataset-type market
    uv run pr-earlystopping-figures --dataset-type enriched --window w20
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from portfolio_recsys.paths import Segmentation, models_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pr-earlystopping-figures")

DEFAULT_HORIZONS = ["1d", "1w", "2w", "1m", "2m", "3m", "6m"]
HORIZON_LABELS = {
    "1d": "1 día", "1w": "1 semana", "2w": "2 semanas", "1m": "1 mes",
    "2m": "2 meses", "3m": "3 meses", "6m": "6 meses", "1y": "1 año",
}
DEFAULT_SEEDS = [11, 23, 42, 71, 101]

# Paleta discreta por semilla (consistente entre celdas).
SEED_COLORS = {
    11: "#1f77b4",
    23: "#ff7f0e",
    42: "#2ca02c",
    71: "#d62728",
    101: "#9467bd",
}


def _project_root() -> Path:
    return Path.cwd()


def _history_path(
    window: str, segmentation: Segmentation, horizon: str,
    dataset_type: str, seed: int,
) -> Path:
    base = models_dir(window, segmentation, horizon) / "test_seeds"
    return base / f"{dataset_type}_seed{seed}_history.parquet"


def _load_histories(
    window: str, segmentation: Segmentation, horizon: str,
    dataset_type: str, seeds: list[int],
) -> dict[int, pl.DataFrame]:
    """Carga los historiales disponibles para un horizonte y conjunto."""
    out: dict[int, pl.DataFrame] = {}
    for seed in seeds:
        path = _history_path(window, segmentation, horizon, dataset_type, seed)
        if not path.exists():
            logger.warning("No existe historial %s", path)
            continue
        try:
            out[seed] = pl.read_parquet(path)
        except Exception:  # noqa: BLE001
            logger.exception("No se pudo leer %s", path)
    return out


def _best_epoch(history: pl.DataFrame) -> int:
    """Devuelve la epoca del mejor checkpoint (minimo de validation_loss)."""
    idx = int(history["validation_loss"].arg_min())
    return int(history["epoch"][idx])


def _plot_split_panel(
    ax, histories: dict[int, pl.DataFrame], loss_col: str, title: str,
    mark_best: bool,
) -> None:
    """Dibuja, por semilla, la curva de perdida de un split (entrenamiento o
    validacion) en un eje. Si ``mark_best`` es True, marca la epoca del mejor
    checkpoint (minimo de validation_loss) sobre la curva."""
    for seed, hist in sorted(histories.items()):
        color = SEED_COLORS.get(seed, None)
        epochs = hist["epoch"].to_numpy()
        loss = hist[loss_col].to_numpy()
        ax.plot(epochs, loss, color=color, linewidth=1.6, label=f"Semilla {seed}")

        if mark_best:
            best_ep = _best_epoch(hist)
            best_row = hist.filter(pl.col("epoch") == best_ep)
            best_val = float(best_row[loss_col][0])
            ax.scatter([best_ep], [best_val], color=color, s=36, zorder=5,
                       edgecolors="black", linewidths=0.6)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida (Smooth L1)")
    ax.grid(True, alpha=0.3)


def plot_early_stopping_horizon(
    window: str, segmentation: Segmentation, dataset_type: str,
    horizon: str, seeds: list[int], out_path: Path,
) -> bool:
    """Figura de un horizonte: dos paneles apilados (entrenamiento arriba,
    validacion abajo), cada uno con una curva por semilla."""
    histories = _load_histories(window, segmentation, horizon, dataset_type, seeds)
    if not histories:
        logger.warning("Sin historiales para %s/%s (%s).", dataset_type, horizon, window)
        return False

    label = HORIZON_LABELS.get(horizon, horizon)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True)

    _plot_split_panel(
        axes[0], histories, "validation_loss",
        f"Validación — horizonte {label}", mark_best=True,
    )
    _plot_split_panel(
        axes[1], histories, "train_loss",
        f"Entrenamiento — horizonte {label}", mark_best=False,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, bbox_to_anchor=(0.5, -0.03))

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura guardada: %s", out_path.name)
    return True


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera figuras de early stopping (curvas de aprendizaje) para la memoria.",
    )
    parser.add_argument("--window", type=str, default="w20",
                        help="Ventana temporal (por defecto w20).")
    parser.add_argument("--segmentation", choices=["unified", "by_sector"],
                        default="unified", help="Segmentacion (por defecto unified).")
    parser.add_argument("--sector", type=str, default=None,
                        help="Sector canonico (si --segmentation by_sector).")
    parser.add_argument("--dataset-type", choices=["market", "enriched"], default=None,
                        help="Conjunto de datos. Si se omite, genera market y enriched.")
    parser.add_argument("--horizons", type=str, default=",".join(DEFAULT_HORIZONS),
                        help="Horizontes separados por comas.")
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS),
                        help="Semillas separadas por comas.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    segmentation = (
        Segmentation.unified()
        if args.segmentation == "unified"
        else Segmentation.for_sector(args.sector)
    )

    root = _project_root()
    images_dir = root / "reports" / "Memoria" / "images"

    horizons = [h.strip() for h in args.horizons.split(",") if h.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    dataset_types = (
        [args.dataset_type] if args.dataset_type else ["market", "enriched"]
    )

    ok_any = False
    for dtype in dataset_types:
        for horizon in horizons:
            out_path = images_dir / f"early_stopping_{dtype}_{horizon}.pdf"
            ok = plot_early_stopping_horizon(
                args.window, segmentation, dtype, horizon, seeds, out_path,
            )
            ok_any = ok_any or ok

    return 0 if ok_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
