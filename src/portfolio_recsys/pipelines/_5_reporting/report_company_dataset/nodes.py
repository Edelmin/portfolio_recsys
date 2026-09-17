"""Nodos del pipeline report_company_dataset.

Aplica comprobaciones de calidad sobre los datasets consolidados COMPANY_*
y genera un informe JSON con métricas por sector y por ticker.

Comprobaciones realizadas:
- Columnas obligatorias presentes
- Duplicados en clave (ticker, date)
- Precios close_eur inválidos (null, no finito, ≤ 0)
- Data leakage: fiscal_date posterior a date
- Completitud: ratio de nulls por columna
- Resumen por ticker: rango temporal, filas, nulls en campos clave
"""

import json
import logging
from datetime import datetime

import polars as pl

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"date", "ticker", "close_eur", "fiscal_date", "sector"}


def _inspect_sector(sector: str, df: pl.DataFrame) -> dict:
    """Ejecuta todas las comprobaciones sobre un dataset de sector.

    Returns:
        Dict con métricas y resultados de las validaciones.
    """
    n_rows = df.shape[0]
    n_cols = df.shape[1]
    n_tickers = df["ticker"].n_unique()

    # 1. Columnas obligatorias
    missing_cols = sorted(REQUIRED_COLUMNS - set(df.columns))

    # 2. Duplicados (ticker, date)
    duplicates = (
        df
        .group_by(["ticker", "date"])
        .len()
        .filter(pl.col("len") > 1)
    )
    n_duplicate_keys = duplicates.height

    # 3. Precios close_eur inválidos
    invalid_prices = df.filter(
        pl.col("close_eur").is_null()
        | ~pl.col("close_eur").is_finite()
        | (pl.col("close_eur") <= 0)
    )
    n_invalid_prices = invalid_prices.height

    # 4. Data leakage: fiscal_date > date
    future_fiscal = df.filter(
        pl.col("fiscal_date").is_not_null()
        & (pl.col("fiscal_date") > pl.col("date"))
    )
    n_future_fiscal = future_fiscal.height

    # 5. Completitud por columna (% de nulls)
    null_ratios = {}
    for col in df.columns:
        null_count = df[col].null_count()
        null_ratios[col] = round(null_count / n_rows * 100, 2) if n_rows > 0 else 0.0

    # 6. Resumen por ticker
    ticker_summary = (
        df
        .group_by("ticker")
        .agg(
            pl.len().alias("n_rows"),
            pl.col("date").min().alias("first_date"),
            pl.col("date").max().alias("last_date"),
            pl.col("close_eur").null_count().alias("null_close_eur"),
            pl.col("fiscal_date").null_count().alias("null_fiscal_date"),
        )
        .sort("n_rows", descending=True)
    )

    tickers_detail = []
    for row in ticker_summary.iter_rows(named=True):
        tickers_detail.append({
            "ticker": row["ticker"],
            "n_rows": row["n_rows"],
            "first_date": str(row["first_date"].date()) if row["first_date"] else None,
            "last_date": str(row["last_date"].date()) if row["last_date"] else None,
            "null_close_eur": row["null_close_eur"],
            "null_fiscal_date": row["null_fiscal_date"],
        })

    # 7. Rango temporal global
    date_min = df["date"].min()
    date_max = df["date"].max()

    # 8. Estadísticas de log-returns (si existen)
    log_return_cols = [c for c in df.columns if c.startswith("log_return_")]
    log_return_stats = {}
    for col in log_return_cols:
        series = df[col].drop_nulls()
        if series.len() > 0:
            log_return_stats[col] = {
                "count": series.len(),
                "null_pct": round(df[col].null_count() / n_rows * 100, 2),
                "mean": round(float(series.mean()), 6),
                "std": round(float(series.std()), 6),
                "min": round(float(series.min()), 6),
                "max": round(float(series.max()), 6),
            }

    # 9. Estadísticas de ratios derivados
    ratio_cols = [
        "gross_profit_ratio", "ebitda_ratio", "operating_income_ratio",
        "net_income_ratio", "per",
    ]
    ratio_stats = {}
    for col in ratio_cols:
        if col in df.columns:
            series = df[col].drop_nulls()
            if series.len() > 0:
                # Filtrar infinitos para estadísticas
                finite_series = series.filter(series.is_finite())
                ratio_stats[col] = {
                    "count": series.len(),
                    "null_pct": round(df[col].null_count() / n_rows * 100, 2),
                    "inf_count": int(series.len() - finite_series.len()),
                    "mean": round(float(finite_series.mean()), 6) if finite_series.len() > 0 else None,
                    "std": round(float(finite_series.std()), 6) if finite_series.len() > 0 else None,
                    "median": round(float(finite_series.median()), 6) if finite_series.len() > 0 else None,
                }

    return {
        "sector": sector,
        "shape": {"rows": n_rows, "columns": n_cols},
        "n_tickers": n_tickers,
        "date_range": {
            "min": str(date_min.date()) if date_min else None,
            "max": str(date_max.date()) if date_max else None,
        },
        "validations": {
            "missing_required_columns": missing_cols,
            "duplicate_ticker_date": n_duplicate_keys,
            "invalid_close_eur": n_invalid_prices,
            "future_fiscal_date": n_future_fiscal,
        },
        "null_ratios_pct": null_ratios,
        "log_return_stats": log_return_stats,
        "ratio_stats": ratio_stats,
        "tickers": tickers_detail,
    }


def generate_company_dataset_report(
    **company_datasets: pl.DataFrame,
) -> str:
    """Genera un informe JSON de calidad para todos los datasets COMPANY_*.

    Args:
        **company_datasets: DataFrames COMPANY_{sector} inyectados por Kedro.

    Returns:
        JSON string con el informe completo.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    sector_reports = []
    total_rows = 0
    total_tickers = 0
    total_issues = 0

    for key, df in sorted(company_datasets.items()):
        # Extraer nombre de sector del key (formato: "{cap}_{sector}" o "{sector}")
        sector = key.split("_", 1)[1] if "_" in key else key

        if df.is_empty():
            sector_reports.append({
                "sector": sector,
                "status": "EMPTY",
            })
            continue

        report = _inspect_sector(sector, df)
        sector_reports.append(report)

        total_rows += report["shape"]["rows"]
        total_tickers += report["n_tickers"]
        total_issues += (
            report["validations"]["duplicate_ticker_date"]
            + report["validations"]["invalid_close_eur"]
            + report["validations"]["future_fiscal_date"]
            + len(report["validations"]["missing_required_columns"])
        )

    # Resumen global
    full_report = {
        "generated_at": timestamp,
        "summary": {
            "sectors_analyzed": len(sector_reports),
            "total_rows": total_rows,
            "total_tickers": total_tickers,
            "total_issues": total_issues,
            "all_clean": total_issues == 0,
        },
        "by_sector": sector_reports,
    }

    logger.info(
        "Report company_dataset: %d sectores, %d filas, %d issues",
        len(sector_reports), total_rows, total_issues,
    )

    return json.dumps(full_report, indent=2, ensure_ascii=False)
