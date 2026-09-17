"""Nodos del pipeline fetch_stock_prices.

Descarga precios historicos OHLCV de acciones desde Yahoo Finance para
todos los tickers del universo enriquecido (ENRICHED_*).

Parametros compartidos (conf/base/parameters.yaml):
  historical_period:
    start_date: "2009-01-01"
    end_date: "2026-01-01"
"""

import logging
import time

import polars as pl
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_stock_prices(
    start_date: str,
    end_date: str,
    **enriched_sectors: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Descarga precios historicos OHLCV de los tickers del universo.

    Para cada ticker con divisa conocida en los ENRICHED_*, descarga
    precios diarios via yfinance y registra la moneda reportada.

    Args:
        start_date: Fecha de inicio ISO 8601.
        end_date: Fecha de fin ISO 8601.
        **enriched_sectors: DataFrames ENRICHED_{cap}_{canonical}.

    Returns:
        Diccionario con dos claves:
          - "prices": DataFrame consolidado con columnas:
            date, ticker, open, high, low, close, volume, currency_reported.
          - "prices_fetch_status": DataFrame con el estado por ticker:
            ticker, sector, cap, status, records, date_min, date_max,
            currency_expected, currency_reported, error.
    """
    # Recopilar todos los tickers unicos con su divisa esperada
    tickers_info: dict[str, dict] = {}  # ticker -> {currency, sector, cap}

    for key, df in enriched_sectors.items():
        if df.is_empty():
            continue
        # Extraer cap y sector del key (formato: ENRICHED_{cap}_{canonical} -> key sin prefijo)
        parts = key.split("_", 1) if "_" in key else [key, ""]
        cap = parts[0] if len(parts) > 1 else "Unknown"
        sector = parts[1] if len(parts) > 1 else key

        for row in df.filter(pl.col("currency").is_not_null()).iter_rows(named=True):
            ticker = row.get("Simbolo", "")
            if ticker and ticker not in tickers_info:
                tickers_info[ticker] = {
                    "currency": row.get("currency", ""),
                    "sector": sector,
                    "cap": cap,
                }

    logger.info("Tickers a descargar: %d", len(tickers_info))

    prices_list: list[pl.DataFrame] = []
    status_list: list[dict] = []

    for ticker, info in sorted(tickers_info.items()):
        try:
            stock = yf.Ticker(ticker)
            # auto_adjust=True: ajusta Close por splits y dividendos (evita saltos
            #   espurios como splits inversos no aplicados).
            # repair=True: repara errores conocidos de Yahoo (precios en unidades
            #   equivocadas, ceros aislados, splits mal aplicados).
            hist = stock.history(
                start=start_date,
                end=end_date,
                auto_adjust=True,
                repair=True,
            )

            if hist.empty:
                status_list.append({
                    "ticker": ticker,
                    "sector": info["sector"],
                    "cap": info["cap"],
                    "status": "NO_DATA",
                    "records": 0,
                    "date_min": "",
                    "date_max": "",
                    "currency_expected": info["currency"],
                    "currency_reported": "",
                    "error": "yfinance devolvio dataset vacio",
                })
                logger.warning("Ticker %s: sin datos", ticker)
                continue

            # Obtener moneda reportada
            try:
                currency_reported = stock.info.get("currency", "UNKNOWN")
            except Exception:
                currency_reported = "UNKNOWN"

            df = (
                pl.from_pandas(hist.reset_index())
                .select([
                    pl.col("Date").alias("date"),
                    pl.lit(ticker).alias("ticker"),
                    pl.col("Open").alias("open"),
                    pl.col("High").alias("high"),
                    pl.col("Low").alias("low"),
                    pl.col("Close").alias("close"),
                    pl.col("Volume").alias("volume"),
                    pl.lit(currency_reported).alias("currency_reported"),
                ])
            )

            # Normalizar date a UTC
            if df["date"].dtype != pl.Datetime("us", "UTC"):
                df = df.with_columns(pl.col("date").cast(pl.Datetime("us", "UTC")))

            date_min = str(df["date"].min())[:10]
            date_max = str(df["date"].max())[:10]

            prices_list.append(df)
            status_list.append({
                "ticker": ticker,
                "sector": info["sector"],
                "cap": info["cap"],
                "status": "OK",
                "records": df.shape[0],
                "date_min": date_min,
                "date_max": date_max,
                "currency_expected": info["currency"],
                "currency_reported": currency_reported,
                "error": "",
            })

            logger.info(
                "Ticker %s: %d registros (%s → %s), moneda: %s",
                ticker, df.shape[0], date_min, date_max, currency_reported,
            )

        except Exception as e:
            status_list.append({
                "ticker": ticker,
                "sector": info["sector"],
                "cap": info["cap"],
                "status": "ERROR",
                "records": 0,
                "date_min": "",
                "date_max": "",
                "currency_expected": info["currency"],
                "currency_reported": "",
                "error": str(e),
            })
            logger.error("Ticker %s: error - %s", ticker, e)

        time.sleep(0.3)  # Rate limiting

    # Consolidar
    if prices_list:
        prices_df = pl.concat(prices_list)
    else:
        prices_df = pl.DataFrame(
            schema={
                "date": pl.Datetime("us", "UTC"),
                "ticker": pl.Utf8,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
                "currency_reported": pl.Utf8,
            }
        )

    fetch_status_df = pl.DataFrame(status_list)

    ok_count = fetch_status_df.filter(pl.col("status") == "OK").shape[0]
    logger.info(
        "Descarga completada: %d/%d tickers exitosos, %d registros totales",
        ok_count, len(tickers_info), prices_df.shape[0],
    )

    return {"prices": prices_df, "prices_fetch_status": fetch_status_df}
