"""Nodos del pipeline fetch_exchange_rates.

Descarga series historicas de pares de divisa desde múltiples fuentes
(Yahoo Finance + Twelve Data como fallback) usando el patrón Strategy.

Parametros (conf/base/parameters.yaml):
  fetch_exchange_rates:
    start_date: "2009-01-01"
    end_date: "2026-01-01"
"""

import logging
import os
import time

import polars as pl

from .providers import (
    TwelveDataProvider,
    YFinanceProvider,
    fetch_with_fallback,
)

logger = logging.getLogger(__name__)


def fetch_exchange_rates(
    start_date: str,
    end_date: str,
    **sector_currencies: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Descarga pares de divisa historicos con fallback entre proveedores.

    Para cada moneda != EUR, intenta primero con Yahoo Finance y si falla
    usa Twelve Data como fuente alternativa.

    Args:
        start_date: Fecha de inicio ISO 8601. Ej: "2009-01-01".
        end_date: Fecha de fin ISO 8601. Ej: "2026-01-01".
        **sector_currencies: DataFrames de currencies por sector (currencies_*).

    Returns:
        Diccionario con dos claves:
          - "rates": DataFrame consolidado (date, currency, rate_to_eur) o vacío.
          - "fetch_status": DataFrame con el estado de cada par descargado
            (pair, currency, status, records, date_min, date_max, provider, error).
    """
    # Consolidar todas las divisas
    dfs = [df for df in sector_currencies.values() if not df.is_empty()]
    if not dfs:
        logger.warning("No hay datos de divisas. Nada que descargar.")
        return {
            "rates": pl.DataFrame(
                schema={"date": pl.Datetime("us", "UTC"), "currency": pl.Utf8, "rate_to_eur": pl.Float64}
            ),
            "fetch_status": pl.DataFrame(
                schema={
                    "pair": pl.Utf8, "currency": pl.Utf8, "status": pl.Utf8,
                    "records": pl.Int64, "date_min": pl.Utf8, "date_max": pl.Utf8,
                    "provider": pl.Utf8, "error": pl.Utf8,
                }
            ),
        }

    currencies_df = pl.concat(dfs)
    all_currencies = currencies_df["currency"].unique().sort().to_list()
    currencies_to_fetch = [c for c in all_currencies if c != "EUR"]

    logger.info(
        "Divisas a descargar: %d pares (de %d divisas unicas)",
        len(currencies_to_fetch),
        len(all_currencies),
    )

    # Configurar providers
    twelve_data_key = os.getenv("TWELVE_DATA_API_KEY", "")
    providers = [
        YFinanceProvider(),
        TwelveDataProvider(api_key=twelve_data_key),
    ]

    rates_list: list[pl.DataFrame] = []
    status_list: list[dict] = []

    for currency in currencies_to_fetch:
        pair = f"{currency}/EUR"

        result, provider_name = fetch_with_fallback(
            currency, start_date, end_date, providers
        )

        if result is not None and not result.is_empty():
            date_min = str(result["date"].min())[:10]
            date_max = str(result["date"].max())[:10]

            status_list.append({
                "pair": pair,
                "currency": currency,
                "status": "OK",
                "records": result.shape[0],
                "date_min": date_min,
                "date_max": date_max,
                "provider": provider_name,
                "error": None,
            })

            rates_list.append(result)
            logger.info(
                "Par %s: %d registros via %s (%s → %s)",
                pair, result.shape[0], provider_name, date_min, date_max,
            )
        else:
            status_list.append({
                "pair": pair,
                "currency": currency,
                "status": "FAILED",
                "records": 0,
                "date_min": None,
                "date_max": None,
                "provider": "none",
                "error": "Todos los providers fallaron",
            })
            logger.error("Par %s: TODOS los providers fallaron", pair)

        time.sleep(0.5)  # Rate limiting entre pares

    # Consolidar
    if rates_list:
        rates_df = pl.concat(rates_list)
    else:
        rates_df = pl.DataFrame(
            schema={"date": pl.Datetime("us", "UTC"), "currency": pl.Utf8, "rate_to_eur": pl.Float64}
        )

    fetch_status_df = pl.DataFrame(status_list)

    ok_count = fetch_status_df.filter(pl.col("status") == "OK").shape[0]
    logger.info(
        "Descarga completada: %d/%d pares exitosos, %d registros totales",
        ok_count,
        len(currencies_to_fetch),
        rates_df.shape[0],
    )

    return {"rates": rates_df, "fetch_status": fetch_status_df}
