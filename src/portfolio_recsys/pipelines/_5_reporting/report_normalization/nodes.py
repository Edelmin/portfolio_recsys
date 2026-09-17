"""Nodos del pipeline report_normalization.

Genera un informe JSON con metricas de la conversion de precios a EUR,
incluyendo analisis de cobertura temporal (empresas con 15 años completos).
"""

import json
from datetime import datetime

import polars as pl


def generate_normalization_report(
    conversion_status: pl.DataFrame,
    quarantine: pl.DataFrame,
    prices_eur: pl.DataFrame,
) -> str:
    """Genera informe de la conversion de monedas a EUR.

    Args:
        conversion_status: Estado por ticker (CONVERTED/QUARANTINE).
        quarantine: Detalle de tickers en cuarentena con razon.
        prices_eur: Precios convertidos a EUR (para analisis de cobertura).

    Returns:
        JSON con metricas de conversion y cobertura temporal.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    if conversion_status.is_empty():
        return json.dumps({"generated_at": timestamp, "error": "Sin datos"})

    total = conversion_status.shape[0]
    converted = conversion_status.filter(pl.col("status") == "CONVERTED")
    quarantined = conversion_status.filter(pl.col("status") == "QUARANTINE")

    # Por moneda
    by_currency_converted = (
        converted.group_by("currency_reported").len().sort("len", descending=True)
    )
    by_currency_quarantine = (
        quarantined.group_by("currency_reported").len().sort("len", descending=True)
    )

    # Detalle cuarentena
    quarantine_detail = []
    if not quarantine.is_empty():
        for row in quarantine.iter_rows(named=True):
            quarantine_detail.append({
                "ticker": row.get("ticker", ""),
                "currency": row.get("currency_reported", ""),
                "reason": row.get("reason", ""),
            })

    # --- Analisis de cobertura temporal (15 años) ---
    min_years = 15
    coverage_analysis = {"min_years_required": min_years}
    full_coverage_tickers = []
    partial_coverage_tickers = []

    if not prices_eur.is_empty():
        ticker_coverage = (
            prices_eur
            .group_by("ticker")
            .agg([
                pl.col("date").min().alias("date_min"),
                pl.col("date").max().alias("date_max"),
                pl.len().alias("records"),
            ])
            .with_columns(
                ((pl.col("date_max") - pl.col("date_min")).dt.total_days() / 365.25)
                .alias("years_covered")
            )
        )

        full_coverage = ticker_coverage.filter(pl.col("years_covered") >= min_years)
        partial_coverage = ticker_coverage.filter(pl.col("years_covered") < min_years)

        full_coverage_tickers = (
            full_coverage
            .sort("ticker")
            .select(["ticker", "years_covered", "records", "date_min", "date_max"])
            .with_columns([
                pl.col("years_covered").round(1),
                pl.col("date_min").cast(pl.Utf8).str.slice(0, 10),
                pl.col("date_max").cast(pl.Utf8).str.slice(0, 10),
            ])
            .to_dicts()
        )

        partial_coverage_tickers = (
            partial_coverage
            .sort("years_covered", descending=True)
            .select(["ticker", "years_covered", "records", "date_min", "date_max"])
            .with_columns([
                pl.col("years_covered").round(1),
                pl.col("date_min").cast(pl.Utf8).str.slice(0, 10),
                pl.col("date_max").cast(pl.Utf8).str.slice(0, 10),
            ])
            .to_dicts()
        )

        coverage_analysis["full_coverage_count"] = len(full_coverage_tickers)
        coverage_analysis["partial_coverage_count"] = len(partial_coverage_tickers)
        coverage_analysis["full_coverage_tickers"] = full_coverage_tickers
        coverage_analysis["partial_coverage_tickers"] = partial_coverage_tickers[:50]
    else:
        coverage_analysis["full_coverage_count"] = 0
        coverage_analysis["partial_coverage_count"] = 0
        coverage_analysis["full_coverage_tickers"] = []
        coverage_analysis["partial_coverage_tickers"] = []

    report = {
        "generated_at": timestamp,
        "summary": {
            "total_tickers": total,
            "converted": converted.shape[0],
            "quarantined": quarantined.shape[0],
            "conversion_rate": round(converted.shape[0] / total * 100, 1) if total > 0 else 0,
            "total_records_converted": int(converted["records_converted"].sum()) if not converted.is_empty() else 0,
            "with_15y_coverage": coverage_analysis.get("full_coverage_count", 0),
        },
        "converted_by_currency": dict(zip(
            by_currency_converted["currency_reported"].to_list(),
            by_currency_converted["len"].to_list(),
        )) if not by_currency_converted.is_empty() else {},
        "quarantined_by_currency": dict(zip(
            by_currency_quarantine["currency_reported"].to_list(),
            by_currency_quarantine["len"].to_list(),
        )) if not by_currency_quarantine.is_empty() else {},
        "quarantine_detail": quarantine_detail[:100],
        "coverage_analysis": coverage_analysis,
    }

    return json.dumps(report, indent=2, ensure_ascii=False)
