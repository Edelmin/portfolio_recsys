"""Nodos del pipeline impute_missing_values.

Rellena valores faltantes en los datasets consolidados por sector:
- Campos financieros: interpolacion lineal entre valores conocidos.
  Si los huecos estan al final (sin referencia posterior), forward-fill.
- Precios (close_eur): forward-fill (precio se mantiene hasta siguiente sesion).

Cada sector se procesa independientemente (un nodo por sector).
"""

import logging

import polars as pl

logger = logging.getLogger(__name__)

# Columnas base que NO son campos financieros
BASE_COLS = {
    "date", "ticker", "close", "close_eur", "currency_original",
    "rate_to_eur", "sector", "fiscal_date",
}

# Columnas de precios que usan forward-fill
PRICE_COLS = {"close", "close_eur", "rate_to_eur"}


def impute_sector(sector_df: pl.DataFrame) -> pl.DataFrame:
    """Rellena valores faltantes en un dataset de sector.

    Estrategia por tipo de columna:
    - close, close_eur, rate_to_eur: forward-fill por ticker.
    - Campos financieros (todo lo demas): interpolacion lineal por ticker,
      con forward-fill para valores al final sin referencia posterior.

    Args:
        sector_df: DataFrame de un sector (output de build_company_dataset).

    Returns:
        DataFrame con los mismos campos pero con nulls imputados.
    """
    if sector_df.is_empty():
        return sector_df

    # Identificar columnas financieras (excluir log-returns: sus nulls son estructurales)
    financial_cols = [
        c for c in sector_df.columns
        if c not in BASE_COLS and not c.startswith("log_return_")
    ]

    # Ordenar por ticker + fecha para que ffill/interpolate funcionen correctamente
    df = sector_df.sort(["ticker", "date"])

    # --- Forward-fill para columnas de precio ---
    price_fill_cols = [c for c in PRICE_COLS if c in df.columns]
    if price_fill_cols:
        df = df.with_columns([
            pl.col(c).forward_fill().over("ticker").alias(c)
            for c in price_fill_cols
        ])

    # --- Interpolacion lineal + forward-fill para campos financieros ---
    if financial_cols:
        df = df.with_columns([
            pl.col(c)
            .interpolate()
            .over("ticker")
            .alias(c)
            for c in financial_cols
        ])
        # Forward-fill para los valores al final que no tienen referencia posterior
        df = df.with_columns([
            pl.col(c)
            .forward_fill()
            .over("ticker")
            .alias(c)
            for c in financial_cols
        ])

    # Estadisticas
    total_cells = df.shape[0] * df.shape[1]
    remaining_nulls = sum(df[c].null_count() for c in df.columns)
    logger.info(
        "Sector imputado: %d registros, %d nulls restantes (%.2f%%)",
        df.shape[0],
        remaining_nulls,
        remaining_nulls / total_cells * 100 if total_cells > 0 else 0,
    )

    return df
