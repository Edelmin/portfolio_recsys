"""Pipeline report_stock_prices.

Genera metricas de la descarga de precios: tasa de exito, discrepancias
de moneda, cobertura temporal y tickers fallidos.

Datasets:
  - Input:  stock_prices_fetch_status
  - Output: stock_prices_report (JSON en data/06_reporting/stock_prices/)
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import generate_stock_prices_report


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=generate_stock_prices_report,
                inputs={
                    "fetch_status": "stock_prices_fetch_status",
                    "start_date": "params:historical_period.start_date",
                    "end_date": "params:historical_period.end_date",
                },
                outputs="stock_prices_report",
                name="generate_stock_prices_report",
            ),
        ],
        tags=["report_stock_prices"],
    )
