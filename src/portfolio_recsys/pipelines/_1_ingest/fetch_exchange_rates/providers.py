"""Proveedores de tipos de cambio (patrón Strategy).

Cada provider implementa el mismo contrato: recibe una moneda y un rango de fechas,
y devuelve un DataFrame polars con columnas (date, currency, rate_to_eur) o None si falla.

Convención de símbolos:
  - yfinance:    {currency}EUR=X   (ej: USDEUR=X)
  - Twelve Data: {currency}/EUR    (ej: USD/EUR)

Las fechas de salida se normalizan a Datetime[us, UTC] (ISO 8601).
"""

import logging
import time
from typing import Protocol

import polars as pl

logger = logging.getLogger(__name__)


class ExchangeRateProvider(Protocol):
    """Protocolo para proveedores de tipos de cambio."""

    @property
    def name(self) -> str:
        """Nombre identificador del provider."""
        ...

    def fetch(self, currency: str, start_date: str, end_date: str) -> pl.DataFrame | None:
        """Descarga el par {currency}/EUR para el rango dado.

        Returns:
            DataFrame con columnas (date: Datetime UTC, currency: Utf8, rate_to_eur: Float64)
            o None si no hay datos o falla.
        """
        ...


class YFinanceProvider:
    """Proveedor de tipos de cambio via Yahoo Finance."""

    @property
    def name(self) -> str:
        return "yfinance"

    def fetch(self, currency: str, start_date: str, end_date: str) -> pl.DataFrame | None:
        import yfinance as yf

        pair = f"{currency}EUR=X"
        try:
            ticker = yf.Ticker(pair)
            hist = ticker.history(start=start_date, end=end_date)

            if hist.empty:
                logger.warning("[yfinance] %s: sin datos", pair)
                return None

            df = (
                pl.from_pandas(hist.reset_index()[["Date", "Close"]])
                .rename({"Date": "date", "Close": "rate_to_eur"})
                .with_columns(pl.lit(currency).alias("currency"))
            )

            # Normalizar date a Datetime UTC
            if df["date"].dtype != pl.Datetime("us", "UTC"):
                df = df.with_columns(
                    pl.col("date").cast(pl.Datetime("us", "UTC"))
                )

            logger.info("[yfinance] %s: %d registros", pair, df.shape[0])
            return df

        except Exception as e:
            logger.error("[yfinance] %s: error - %s", pair, e)
            return None


class TwelveDataProvider:
    """Proveedor de tipos de cambio via Twelve Data API.

    Requiere API key en variable de entorno TWELVE_DATA_API_KEY.
    Plan gratuito: 8 requests/min, 800/día.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def name(self) -> str:
        return "twelve_data"

    def fetch(self, currency: str, start_date: str, end_date: str) -> pl.DataFrame | None:
        import requests

        if not self.api_key:
            logger.warning("[twelve_data] API key no configurada, saltando")
            return None

        symbol = f"{currency}/EUR"
        all_records: list[dict] = []
        current_start = start_date

        while True:
            params = {
                "symbol": symbol,
                "interval": "1day",
                "start_date": current_start,
                "end_date": end_date,
                "outputsize": 5000,
                "apikey": self.api_key,
            }

            try:
                response = requests.get(
                    "https://api.twelvedata.com/time_series",
                    params=params,
                    timeout=30,
                )
                data = response.json()
            except Exception as e:
                logger.error("[twelve_data] %s: request error - %s", symbol, e)
                break

            if "values" not in data or not data["values"]:
                if not all_records:
                    logger.warning("[twelve_data] %s: sin datos", symbol)
                break

            all_records.extend(data["values"])

            # Si devolvió menos de 5000, ya tenemos todo
            if len(data["values"]) < 5000:
                break

            # Paginar: avanzar start_date
            # Los valores vienen en orden descendente (más reciente primero)
            oldest_in_batch = data["values"][-1]["datetime"]
            current_start = oldest_in_batch
            time.sleep(8)  # Rate limiting

        if not all_records:
            return None

        records = [
            {"date": v["datetime"], "rate_to_eur": float(v["close"]), "currency": currency}
            for v in all_records
            if v.get("close") is not None
        ]

        if not records:
            return None

        df = (
            pl.DataFrame(records)
            .with_columns(
                pl.col("date")
                .str.to_datetime("%Y-%m-%d", time_zone="UTC", strict=False)
            )
            .filter(pl.col("date").is_not_null())
            .unique(subset=["date"])
            .sort("date")
        )

        logger.info("[twelve_data] %s: %d registros", symbol, df.shape[0])
        return df


def fetch_with_fallback(
    currency: str,
    start_date: str,
    end_date: str,
    providers: list[ExchangeRateProvider],
) -> tuple[pl.DataFrame | None, str]:
    """Intenta descargar un par usando múltiples providers en orden.

    Devuelve el primer resultado exitoso y el nombre del provider usado.

    Args:
        currency: Código ISO 4217 de la moneda (ej: "USD").
        start_date: Fecha inicio ISO 8601.
        end_date: Fecha fin ISO 8601.
        providers: Lista ordenada de providers (el primero tiene prioridad).

    Returns:
        Tupla (DataFrame o None, nombre del provider que tuvo éxito o "none").
    """
    for provider in providers:
        result = provider.fetch(currency, start_date, end_date)
        if result is not None and not result.is_empty():
            return result, provider.name
    return None, "none"
