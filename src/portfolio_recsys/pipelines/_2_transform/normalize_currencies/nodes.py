"""Nodos del pipeline normalize_currencies.

Convierte los precios de cierre historicos a EUR usando los tipos de cambio
descargados. Los tickers cuya moneda no tiene par de conversion disponible
se ponen en cuarentena.
"""

import logging

import polars as pl

logger = logging.getLogger(__name__)


def normalize_prices_to_eur(
    stock_prices: pl.DataFrame,
    exchange_rates: pl.DataFrame,
    price_quality: dict | None = None,
) -> dict[str, pl.DataFrame]:
    """Convierte precios de cierre a EUR y separa los que no se pueden convertir.

    Logica:
    - Si currency_reported == "EUR" → close_eur = close (sin conversion).
    - Si no → join asof por (currency, date) con exchange_rates para obtener
      el rate_to_eur mas cercano anterior (forward-fill implícito via join_asof).
    - close_eur = close * rate_to_eur.
    - Tickers sin rate_to_eur disponible → cuarentena.
    - Tickers con cotizaciones invalidas o cercanas a cero (precio <= min_price_eur
      o mediana < min_median_price_eur) → cuarentena por calidad de datos.

    Args:
        stock_prices: DataFrame con columnas: date, ticker, open, high, low,
            close, volume, currency_reported.
        exchange_rates: DataFrame con columnas: date, currency, rate_to_eur.
        price_quality: Dict de configuracion de calidad de precios. Claves:
            drop_nonpositive_prices (bool), min_price_eur (float),
            min_median_price_eur (float). Si es None, no se aplica el descarte.

    Returns:
        Diccionario con tres claves:
        - "prices_eur": Precios convertidos a EUR (date, ticker, close, close_eur,
          currency_original, rate_to_eur).
        - "quarantine": Tickers que no se pudieron convertir (ticker,
          currency_reported, reason).
        - "conversion_status": Resumen por ticker (ticker, currency_reported,
          status, records_converted, records_missing_rate).
    """
    if stock_prices.is_empty():
        empty_prices = pl.DataFrame(schema={
            "date": pl.Datetime("us", "UTC"), "ticker": pl.Utf8,
            "close": pl.Float64, "close_eur": pl.Float64,
            "currency_original": pl.Utf8, "rate_to_eur": pl.Float64,
        })
        empty_quarantine = pl.DataFrame(schema={
            "ticker": pl.Utf8, "currency_reported": pl.Utf8, "reason": pl.Utf8,
        })
        empty_status = pl.DataFrame(schema={
            "ticker": pl.Utf8, "currency_reported": pl.Utf8, "status": pl.Utf8,
            "records_converted": pl.UInt32, "records_missing_rate": pl.UInt32,
        })
        return {
            "prices_eur": empty_prices,
            "quarantine": empty_quarantine,
            "conversion_status": empty_status,
        }

    # Separar EUR y non-EUR
    eur_prices = stock_prices.filter(pl.col("currency_reported") == "EUR")
    non_eur_prices = stock_prices.filter(pl.col("currency_reported") != "EUR")

    # Compatibilidad: si el Parquet todavía tiene "rate" (nombre antiguo), renombrar
    if "rate" in exchange_rates.columns and "rate_to_eur" not in exchange_rates.columns:
        exchange_rates = exchange_rates.rename({"rate": "rate_to_eur"})

    # Divisas disponibles en exchange_rates
    available_currencies = set(
        exchange_rates["currency"].unique().to_list()
    ) if not exchange_rates.is_empty() else set()

    # Identificar tickers sin par de conversion
    non_eur_currencies = non_eur_prices["currency_reported"].unique().to_list()
    missing_currencies = [c for c in non_eur_currencies if c not in available_currencies]

    # Separar tickers convertibles vs en cuarentena
    quarantine_mask = non_eur_prices["currency_reported"].is_in(missing_currencies)
    quarantine_prices = non_eur_prices.filter(quarantine_mask)
    convertible_prices = non_eur_prices.filter(~quarantine_mask)

    # --- Convertir EUR directamente ---
    eur_converted = eur_prices.select([
        pl.col("date"),
        pl.col("ticker"),
        pl.col("close"),
        pl.col("close").alias("close_eur"),
        pl.col("currency_reported").alias("currency_original"),
        pl.lit(1.0).alias("rate_to_eur"),
    ])

    # --- Convertir non-EUR via join asof ---
    converted_parts = [eur_converted]

    if not convertible_prices.is_empty() and not exchange_rates.is_empty():
        # Preparar exchange_rates para join: ordenar por date
        rates_sorted = exchange_rates.sort("date")

        # Join asof: para cada (date, currency) en prices, encontrar el rate_to_eur
        # mas cercano anterior en exchange_rates
        prices_sorted = convertible_prices.sort("date")

        joined = prices_sorted.join_asof(
            rates_sorted.rename({"currency": "rate_currency"}),
            left_on="date",
            right_on="date",
            by_left="currency_reported",
            by_right="rate_currency",
            strategy="backward",  # Usar el rate mas reciente <= date
        )

        # Calcular close_eur
        converted = joined.select([
            pl.col("date"),
            pl.col("ticker"),
            pl.col("close"),
            (pl.col("close") * pl.col("rate_to_eur")).alias("close_eur"),
            pl.col("currency_reported").alias("currency_original"),
            pl.col("rate_to_eur"),
        ])

        converted_parts.append(converted)

    # Consolidar precios convertidos
    if converted_parts:
        prices_eur = pl.concat(converted_parts)
    else:
        prices_eur = eur_converted

    # --- Cuarentena ---
    quarantine_tickers = (
        quarantine_prices
        .select(["ticker", "currency_reported"])
        .unique()
        .with_columns(pl.lit("Sin par de conversion disponible").alias("reason"))
    )

    # Tambien poner en cuarentena tickers con rate_to_eur = null (gap sin fill)
    if "rate_to_eur" in prices_eur.columns:
        null_rate_tickers = (
            prices_eur
            .filter(pl.col("rate_to_eur").is_null())
            .select("ticker")
            .unique()
        )
        if not null_rate_tickers.is_empty():
            # Mover estos tickers completos a cuarentena
            null_tickers_list = null_rate_tickers["ticker"].to_list()
            extra_quarantine = (
                stock_prices
                .filter(pl.col("ticker").is_in(null_tickers_list))
                .select(["ticker", "currency_reported"])
                .unique()
                .with_columns(pl.lit("Rate no disponible para algunas fechas").alias("reason"))
            )
            quarantine_tickers = pl.concat([quarantine_tickers, extra_quarantine]).unique()

            # Quitar de prices_eur
            prices_eur = prices_eur.filter(~pl.col("ticker").is_in(null_tickers_list))

    # --- Cuarentena por calidad de precios (no positivos / penny stocks) ---
    if price_quality and price_quality.get("drop_nonpositive_prices", False) \
            and not prices_eur.is_empty():
        min_price = float(price_quality.get("min_price_eur", 0.0))
        min_median = float(price_quality.get("min_median_price_eur", 0.0))

        # Estadisticas de precio por ticker sobre close_eur
        ticker_stats = prices_eur.group_by("ticker").agg(
            pl.col("close_eur").min().alias("min_close_eur"),
            pl.col("close_eur").median().alias("median_close_eur"),
        )

        bad_quality = ticker_stats.filter(
            (pl.col("min_close_eur") <= min_price)
            | (pl.col("median_close_eur") < min_median)
        )

        if not bad_quality.is_empty():
            bad_tickers = bad_quality.get_column("ticker").to_list()

            quality_quarantine = (
                stock_prices
                .filter(pl.col("ticker").is_in(bad_tickers))
                .select(["ticker", "currency_reported"])
                .unique()
                .with_columns(
                    pl.lit(
                        f"Precios invalidos o penny stock "
                        f"(min <= {min_price} EUR o mediana < {min_median} EUR)"
                    ).alias("reason")
                )
            )
            quarantine_tickers = pl.concat(
                [quarantine_tickers, quality_quarantine]
            ).unique()

            prices_eur = prices_eur.filter(~pl.col("ticker").is_in(bad_tickers))
            logger.info(
                "Calidad de precios: %d tickers descartados por precios invalidos/penny: %s",
                len(bad_tickers),
                ", ".join(sorted(bad_tickers)[:20]) + ("..." if len(bad_tickers) > 20 else ""),
            )

    # --- Status por ticker ---
    all_tickers = stock_prices.select(["ticker", "currency_reported"]).unique()

    converted_tickers = set(prices_eur["ticker"].unique().to_list()) if not prices_eur.is_empty() else set()
    quarantine_set = set(quarantine_tickers["ticker"].to_list()) if not quarantine_tickers.is_empty() else set()

    status_records = []
    for row in all_tickers.iter_rows(named=True):
        ticker = row["ticker"]
        curr = row["currency_reported"]
        if ticker in converted_tickers:
            n_converted = prices_eur.filter(pl.col("ticker") == ticker).shape[0]
            status_records.append({
                "ticker": ticker,
                "currency_reported": curr,
                "status": "CONVERTED",
                "records_converted": n_converted,
                "records_missing_rate": 0,
            })
        elif ticker in quarantine_set:
            n_original = stock_prices.filter(pl.col("ticker") == ticker).shape[0]
            status_records.append({
                "ticker": ticker,
                "currency_reported": curr,
                "status": "QUARANTINE",
                "records_converted": 0,
                "records_missing_rate": n_original,
            })
        else:
            status_records.append({
                "ticker": ticker,
                "currency_reported": curr,
                "status": "UNKNOWN",
                "records_converted": 0,
                "records_missing_rate": 0,
            })

    conversion_status = pl.DataFrame(status_records)

    # Log resumen
    n_converted = conversion_status.filter(pl.col("status") == "CONVERTED").shape[0]
    n_quarantine = conversion_status.filter(pl.col("status") == "QUARANTINE").shape[0]
    logger.info(
        "Normalizacion completada: %d tickers convertidos, %d en cuarentena, %d registros EUR",
        n_converted, n_quarantine, prices_eur.shape[0],
    )

    return {
        "prices_eur": prices_eur,
        "quarantine": quarantine_tickers,
        "conversion_status": conversion_status,
    }
