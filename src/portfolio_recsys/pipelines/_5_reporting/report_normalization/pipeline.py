"""Pipeline report_normalization.

Genera metricas sobre la conversion de precios a EUR.

Datasets:
  - Input:  currency_conversion_status, stock_prices_quarantine
  - Output: normalization_report (JSON en data/06_reporting/normalization/)
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import generate_normalization_report


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=generate_normalization_report,
                inputs={
                    "conversion_status": "currency_conversion_status",
                    "quarantine": "stock_prices_quarantine",
                    "prices_eur": "stock_prices_eur",
                },
                outputs="normalization_report",
                name="generate_normalization_report",
            ),
        ],
        tags=["report_normalization"],
    )
