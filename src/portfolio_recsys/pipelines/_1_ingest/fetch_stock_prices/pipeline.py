"""Pipeline fetch_stock_prices.

Descarga precios historicos OHLCV de acciones desde Yahoo Finance para
los tickers del universo enriquecido, usando el periodo historico compartido.

Datasets:
  - Input:  ENRICHED_{cap}_{canonical} (22 datasets)
  - Output: stock_prices (DataFrame OHLCV consolidado)
            stock_prices_fetch_status (estado por ticker)
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import fetch_stock_prices
from portfolio_recsys.pipelines.sector_mapping import SECTORS_WITH_FS, CAPS


def create_pipeline(**kwargs) -> Pipeline:
    # Construir inputs: todos los ENRICHED_* + parametros de periodo
    inputs = {}
    for cap in CAPS:
        for canonical in SECTORS_WITH_FS:
            key = f"{cap}_{canonical}"
            inputs[key] = f"ENRICHED_{cap}_{canonical}"

    inputs["start_date"] = "params:historical_period.start_date"
    inputs["end_date"] = "params:historical_period.end_date"

    return pipeline(
        [
            node(
                func=fetch_stock_prices,
                inputs=inputs,
                outputs={
                    "prices": "stock_prices",
                    "prices_fetch_status": "stock_prices_fetch_status",
                },
                name="fetch_stock_prices",
            ),
        ],
        tags=["fetch_stock_prices"],
    )
