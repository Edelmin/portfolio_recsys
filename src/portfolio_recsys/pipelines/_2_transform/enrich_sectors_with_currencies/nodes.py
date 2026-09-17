"""Nodos del pipeline enrich_sectors_with_currencies.

Hace el left join entre el dataset de empresas (SECTOR_*) y las divisas
extraidas (currencies_*).

El informe de discrepancias del matching se genera en el pipeline de reporting
`report_matching`.
"""

import polars as pl


def enrich_sector_with_currency(
    sector_df: pl.DataFrame, currencies_df: pl.DataFrame
) -> pl.DataFrame:
    """Enriquece el dataset de empresas con la informacion de divisa.

    Hace un left join usando Simbolo (sector_df) = ticker (currencies_df).

    Args:
        sector_df: DataFrame del sector con columna 'Simbolo'.
        currencies_df: DataFrame de divisas con columnas: ticker, currency,
                       period_start, period_end.

    Returns:
        DataFrame enriquecido con columnas adicionales: currency, period_start, period_end.
    """
    if currencies_df.is_empty():
        return sector_df.with_columns([
            pl.lit(None).cast(pl.Utf8).alias("currency"),
            pl.lit(None).cast(pl.Datetime("us", "UTC")).alias("period_start"),
            pl.lit(None).cast(pl.Datetime("us", "UTC")).alias("period_end"),
        ])

    enriched = sector_df.join(
        currencies_df.select(["ticker", "currency", "period_start", "period_end"]),
        left_on="Simbolo",
        right_on="ticker",
        how="left",
    )

    return enriched
