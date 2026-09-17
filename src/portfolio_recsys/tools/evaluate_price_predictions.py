"""Reconstruye precios en EUR predichos vs reales y calcula metricas por horizonte.

Para cada horizonte, cruza la ``prediction_table`` (que contiene log-retornos
predichos y reales sobre el periodo de holding) con ``stock_prices_eur`` para
reconstruir, por operacion (ticker + fecha de rebalanceo):

- ``entry_close_eur``       : precio en EUR en la fecha de entrada al trade.
- ``predicted_exit_close_eur``: precio en EUR predicho al final del horizonte
                               (= entry * exp(predicted_log_return)).
- ``actual_exit_close_eur`` : precio en EUR real al final del horizonte.
- retornos simples predicho y real, y errores en precio.

Guarda una tabla enriquecida por horizonte en
``05_model_outputs/predictions/{window}/{segmentation}/{horizon}/prediction_prices_eur.parquet``
y muestra un resumen de metricas (MAE/RMSE en log-retorno, en retorno simple y
en precio EUR).

Uso:
    uv run pr-price-eval
    uv run pr-price-eval --horizons 1d,3m,1y --split validation
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import polars as pl

from portfolio_recsys.paths import Segmentation, predictions_dir as predictions_dir_for

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("pr-price-eval")

DEFAULT_HORIZONS = ["1d", "1w", "2w", "1m", "2m", "3m", "6m", "1y"]


def _project_root() -> Path:
    return Path.cwd()


def enrich_horizon(
    segmentation: Segmentation,
    window: str,
    prices_eur: pl.DataFrame,
    horizon: str,
    split: str,
) -> pl.DataFrame | None:
    """Cruza la prediction_table de un horizonte con precios EUR de entrada/salida."""
    p = predictions_dir_for(window, segmentation, horizon) / "prediction_table.parquet"
    if not p.exists():
        logger.warning("No existe %s", p)
        return None

    df = pl.read_parquet(p)
    if split != "all":
        df = df.filter(pl.col("split") == split)
    if df.is_empty():
        logger.warning("[%s] sin filas para split=%s", horizon, split)
        return None

    entry_prices = prices_eur.select(
        pl.col("ticker"),
        pl.col("date").alias("trade_entry_date"),
        pl.col("close_eur").alias("entry_close_eur"),
    )
    exit_prices = prices_eur.select(
        pl.col("ticker"),
        pl.col("date").alias("trade_exit_date"),
        pl.col("close_eur").alias("actual_exit_close_eur"),
    )

    df = (
        df.join(entry_prices, on=["ticker", "trade_entry_date"], how="left")
        .join(exit_prices, on=["ticker", "trade_exit_date"], how="left")
    )

    # Reconstruir precio predicho y retornos simples.
    df = df.with_columns(
        (pl.col("entry_close_eur") * pl.col("predicted_log_return").exp())
        .alias("predicted_exit_close_eur"),
        (pl.col("predicted_log_return").exp() - 1.0).alias("predicted_simple_return"),
        (pl.col("target_log_return").exp() - 1.0).alias("actual_simple_return"),
    )
    df = df.with_columns(
        (pl.col("predicted_exit_close_eur") - pl.col("actual_exit_close_eur"))
        .alias("price_error_eur"),
        (
            (pl.col("predicted_exit_close_eur") - pl.col("actual_exit_close_eur"))
            / pl.col("actual_exit_close_eur")
        ).alias("price_error_pct"),
    )
    return df


def compute_metrics(df: pl.DataFrame, horizon: str) -> dict:
    """Metricas de error en log-retorno, retorno simple y precio EUR."""
    # Solo filas con precios disponibles en ambos extremos
    valid = df.filter(
        pl.col("entry_close_eur").is_not_null()
        & pl.col("actual_exit_close_eur").is_not_null()
        & pl.col("actual_exit_close_eur").is_finite()
    )
    n = valid.height
    if n == 0:
        return {"horizon": horizon, "n": 0}

    log_err = (valid.get_column("predicted_log_return")
               - valid.get_column("target_log_return")).to_numpy()
    simple_err = (valid.get_column("predicted_simple_return")
                  - valid.get_column("actual_simple_return")).to_numpy()
    price_err = valid.get_column("price_error_eur").to_numpy()
    price_pct = valid.get_column("price_error_pct").to_numpy()
    entry = valid.get_column("entry_close_eur").to_numpy()

    def _mae(a):
        return float(np.mean(np.abs(a)))

    def _rmse(a):
        return float(np.sqrt(np.mean(a ** 2)))

    return {
        "horizon": horizon,
        "n": n,
        "mae_log": _mae(log_err),
        "rmse_log": _rmse(log_err),
        "mae_simple": _mae(simple_err),
        "rmse_simple": _rmse(simple_err),
        "mae_price_eur": _mae(price_err),
        "rmse_price_eur": _rmse(price_err),
        "mae_price_pct": _mae(price_pct),
        "mean_entry_price_eur": float(np.mean(entry)),
        # Error de precio relativo al precio medio de entrada (adimensional,
        # comparable entre horizontes).
        "rmse_price_over_entry": _rmse(price_err) / float(np.mean(entry)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruye precios EUR predichos vs reales y calcula metricas.")
    parser.add_argument("--horizons", type=str, default=",".join(DEFAULT_HORIZONS))
    parser.add_argument("--split", type=str, default="validation",
                        choices=["validation", "test", "train", "all"])
    parser.add_argument("--window", type=str, default="w60",
                        help="Clave de ventana temporal (por defecto w60).")
    parser.add_argument("--segmentation", choices=["unified", "by_sector"],
                        default="unified", help="Segmentacion (por defecto unified).")
    parser.add_argument("--sector", type=str, default=None,
                        help="Sector canonico (si --segmentation by_sector).")
    args = parser.parse_args(argv)

    segmentation = (
        Segmentation.unified()
        if args.segmentation == "unified"
        else Segmentation.for_sector(args.sector)
    )

    root = _project_root()
    prices_path = (root / "data" / "02_intermediate" / "normalized_prices"
                   / "stock_prices_eur.parquet")
    if not prices_path.exists():
        logger.error("No existe %s", prices_path)
        return 1
    prices_eur = pl.read_parquet(prices_path).select(["ticker", "date", "close_eur"])

    horizons = [h.strip() for h in args.horizons.split(",") if h.strip()]
    all_metrics = []
    for h in horizons:
        enriched = enrich_horizon(segmentation, args.window, prices_eur, h, args.split)
        if enriched is None:
            continue
        out = (predictions_dir_for(args.window, segmentation, h)
               / "prediction_prices_eur.parquet")
        out.parent.mkdir(parents=True, exist_ok=True)
        enriched.write_parquet(out)
        logger.info("[%s] tabla enriquecida guardada: %s (%d filas)", h, out, enriched.height)
        all_metrics.append(compute_metrics(enriched, h))

    # Resumen
    print("\n" + "=" * 118)
    print(f"CAPACIDAD PREDICTIVA EN PRECIO — split={args.split}")
    print("=" * 118)
    print(f"{'h':<4}{'n':>8}{'MAE_log':>9}{'RMSE_log':>10}{'MAE_simp':>10}"
          f"{'RMSE_simp':>10}{'MAE_€':>10}{'RMSE_€':>10}{'MAE_%pre':>10}"
          f"{'RMSE€/ent':>11}{'€ent_med':>10}")
    for m in all_metrics:
        if m.get("n", 0) == 0:
            print(f"{m['horizon']:<4}  (sin datos)")
            continue
        print(f"{m['horizon']:<4}{m['n']:>8}{m['mae_log']:>9.4f}{m['rmse_log']:>10.4f}"
              f"{m['mae_simple']:>10.4f}{m['rmse_simple']:>10.4f}"
              f"{m['mae_price_eur']:>10.2f}{m['rmse_price_eur']:>10.2f}"
              f"{m['mae_price_pct']*100:>9.1f}%{m['rmse_price_over_entry']*100:>10.1f}%"
              f"{m['mean_entry_price_eur']:>10.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
