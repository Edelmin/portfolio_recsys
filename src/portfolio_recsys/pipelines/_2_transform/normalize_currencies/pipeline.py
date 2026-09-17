"""Pipeline normalize_currencies.

Convierte los precios de cierre a EUR usando tipos de cambio historicos.
Los tickers sin conversion disponible se ponen en cuarentena.

Datasets:
  - Input:  stock_prices, exchange_rates
  - Output: stock_prices_eur (precios en EUR)
            stock_prices_quarantine (tickers sin conversion)
            currency_conversion_status (resumen por ticker)
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import normalize_prices_to_eur


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=normalize_prices_to_eur,
                inputs={
                    "stock_prices": "stock_prices",
                    "exchange_rates": "exchange_rates",
                    "price_quality": "params:price_quality",
                },
                outputs={
                    "prices_eur": "stock_prices_eur",
                    "quarantine": "stock_prices_quarantine",
                    "conversion_status": "currency_conversion_status",
                },
                name="normalize_prices_to_eur",
            ),
            # Benchmark (S&P 500 + activo seguro): misma logica de conversion a
            # EUR, pero en un dataset separado para no contaminar el universo de
            # modelado (build_company_dataset consume solo stock_prices_eur).
            node(
                func=normalize_prices_to_eur,
                inputs={
                    "stock_prices": "benchmark_prices",
                    "exchange_rates": "exchange_rates",
                },
                outputs={
                    "prices_eur": "benchmark_prices_eur",
                    "quarantine": "benchmark_prices_quarantine",
                    "conversion_status": "benchmark_conversion_status",
                },
                name="normalize_benchmark_prices_to_eur",
            ),
        ],
        tags=["normalize_currencies"],
    )
