"""Nodos del pipeline fetch_benchmark_prices.

Descarga el indice de mercado (S&P 500) desde Yahoo Finance y genera la serie
sintetica del activo seguro (crecimiento a una inflacion fija anual).

Estos activos NO pasan por el modelo recurrente. Se materializan como
`benchmark_prices` (moneda nativa) para que despues se conviertan a EUR junto
al resto de precios y el backtest los pueda usar como comparadores.

Parametros compartidos (conf/base/parameters.yaml):
  historical_period:
    start_date: "2009-01-01"
    end_date: "2026-01-01"
  markowitz_portfolio.benchmark:
    market_ticker: "^GSPC"
    risk_free_annual_inflation: 0.0329
"""

import logging

import polars as pl
import yfinance as yf

logger = logging.getLogger(__name__)

# Ticker sintetico del activo seguro. ASCII, sin colision con tickers reales.
RISK_FREE_TICKER = "RISK_FREE"

# Sesiones bursatiles por anho, usadas para repartir la inflacion anual en
# crecimiento diario compuesto del activo seguro.
TRADING_SESSIONS_PER_YEAR = 252.0


def fetch_benchmark_prices(
    start_date: str,
    end_date: str,
    benchmark_params: dict,
) -> dict[str, pl.DataFrame]:
    """Descarga el indice de mercado y construye la serie del activo seguro.

    El indice (ej: ^GSPC) se descarga en su moneda nativa (USD). El activo
    seguro se genera en la misma malla de fechas de sesion del indice, con
    crecimiento compuesto diario equivalente a la inflacion anual configurada.
    Se etiqueta con moneda EUR para que la normalizacion posterior lo deje
    intacto (no requiere conversion).

    Args:
        start_date: Fecha de inicio ISO 8601.
        end_date: Fecha de fin ISO 8601 (exclusiva en yfinance).
        benchmark_params: Sub-dict `markowitz_portfolio.benchmark` con
            `market_ticker` y `risk_free_annual_inflation`.

    Returns:
        Diccionario con dos claves:
          - "prices": DataFrame OHLCV consolidado (indice + activo seguro) con
            columnas: date, ticker, open, high, low, close, volume,
            currency_reported. Compatible con el esquema de `stock_prices`.
          - "fetch_status": DataFrame con el estado por serie:
            ticker, kind, status, records, date_min, date_max,
            currency_reported, error.
    """
    market_ticker = benchmark_params["market_ticker"]
    annual_inflation = float(benchmark_params["risk_free_annual_inflation"])

    prices_list: list[pl.DataFrame] = []
    status_list: list[dict] = []

    # --- 1. Descargar el indice de mercado (S&P 500) ---
    index_df: pl.DataFrame | None = None
    try:
        stock = yf.Ticker(market_ticker)
        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            status_list.append({
                "ticker": market_ticker,
                "kind": "market_index",
                "status": "NO_DATA",
                "records": 0,
                "date_min": "",
                "date_max": "",
                "currency_reported": "",
                "error": "yfinance devolvio dataset vacio",
            })
            logger.warning("Benchmark %s: sin datos", market_ticker)
        else:
            try:
                currency_reported = stock.info.get("currency", "USD")
            except Exception:
                currency_reported = "USD"

            index_df = (
                pl.from_pandas(hist.reset_index())
                .select([
                    pl.col("Date").alias("date"),
                    pl.lit(market_ticker).alias("ticker"),
                    pl.col("Open").alias("open"),
                    pl.col("High").alias("high"),
                    pl.col("Low").alias("low"),
                    pl.col("Close").alias("close"),
                    pl.col("Volume").alias("volume"),
                    pl.lit(currency_reported).alias("currency_reported"),
                ])
            )

            if index_df["date"].dtype != pl.Datetime("us", "UTC"):
                index_df = index_df.with_columns(
                    pl.col("date").cast(pl.Datetime("us", "UTC"))
                )

            date_min = str(index_df["date"].min())[:10]
            date_max = str(index_df["date"].max())[:10]

            prices_list.append(index_df)
            status_list.append({
                "ticker": market_ticker,
                "kind": "market_index",
                "status": "OK",
                "records": index_df.shape[0],
                "date_min": date_min,
                "date_max": date_max,
                "currency_reported": currency_reported,
                "error": "",
            })
            logger.info(
                "Benchmark %s: %d registros (%s -> %s), moneda: %s",
                market_ticker, index_df.shape[0], date_min, date_max,
                currency_reported,
            )
    except Exception as e:  # noqa: BLE001 - registrar cualquier fallo de red
        status_list.append({
            "ticker": market_ticker,
            "kind": "market_index",
            "status": "ERROR",
            "records": 0,
            "date_min": "",
            "date_max": "",
            "currency_reported": "",
            "error": str(e),
        })
        logger.error("Benchmark %s: error %s", market_ticker, e)

    # --- 2. Generar la serie sintetica del activo seguro ---
    # Usa la malla de fechas del indice si esta disponible; si no, un rango
    # diario de negocio entre start_date y end_date.
    if index_df is not None and index_df.height > 0:
        session_dates = index_df.get_column("date")
    else:
        session_dates = (
            pl.datetime_range(
                start=pl.lit(start_date).str.to_datetime(),
                end=pl.lit(end_date).str.to_datetime(),
                interval="1d",
                eager=True,
                time_zone="UTC",
            )
            .to_frame("date")
            .filter(pl.col("date").dt.weekday() <= 5)  # 1=Lun ... 5=Vie
            .get_column("date")
        )

    n = session_dates.len()
    if n > 0:
        daily_growth = (1.0 + annual_inflation) ** (1.0 / TRADING_SESSIONS_PER_YEAR)
        # close_t = base * daily_growth ^ t, con base = 100.0
        base = 100.0
        exponents = pl.int_range(0, n, eager=True).cast(pl.Float64)
        close_series = (base * (daily_growth ** exponents)).alias("close")

        risk_free_df = pl.DataFrame({"date": session_dates}).with_columns([
            pl.lit(RISK_FREE_TICKER).alias("ticker"),
            close_series,
        ]).with_columns([
            pl.col("close").alias("open"),
            pl.col("close").alias("high"),
            pl.col("close").alias("low"),
            pl.lit(0.0).alias("volume"),
            # Etiquetado como EUR: la normalizacion posterior lo deja intacto.
            pl.lit("EUR").alias("currency_reported"),
        ]).select([
            "date", "ticker", "open", "high", "low", "close", "volume",
            "currency_reported",
        ])

        prices_list.append(risk_free_df)
        status_list.append({
            "ticker": RISK_FREE_TICKER,
            "kind": "risk_free",
            "status": "OK",
            "records": risk_free_df.shape[0],
            "date_min": str(risk_free_df["date"].min())[:10],
            "date_max": str(risk_free_df["date"].max())[:10],
            "currency_reported": "EUR",
            "error": "",
        })
        logger.info(
            "Activo seguro %s: %d registros, inflacion anual %.4f (diaria %.8f)",
            RISK_FREE_TICKER, risk_free_df.shape[0], annual_inflation,
            daily_growth - 1.0,
        )
    else:
        status_list.append({
            "ticker": RISK_FREE_TICKER,
            "kind": "risk_free",
            "status": "NO_DATA",
            "records": 0,
            "date_min": "",
            "date_max": "",
            "currency_reported": "EUR",
            "error": "sin fechas de sesion para generar la serie",
        })
        logger.warning("Activo seguro: no se pudo generar (sin fechas)")

    # --- 3. Consolidar ---
    if prices_list:
        prices = pl.concat(prices_list, how="vertical_relaxed").sort(["ticker", "date"])
    else:
        prices = pl.DataFrame(schema={
            "date": pl.Datetime("us", "UTC"), "ticker": pl.Utf8,
            "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
            "close": pl.Float64, "volume": pl.Float64, "currency_reported": pl.Utf8,
        })

    fetch_status = pl.DataFrame(status_list)

    return {"prices": prices, "fetch_status": fetch_status}
