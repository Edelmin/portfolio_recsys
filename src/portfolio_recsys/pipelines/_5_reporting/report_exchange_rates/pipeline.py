"""Pipeline report_exchange_rates.

Valida la completitud de los pares de divisa descargados y genera un informe
con pares fallidos, cobertura temporal y gaps en las series.

Datasets:
  - Input:  exchange_rates_fetch_status, exchange_rates
  - Output: exchange_rates_report (JSON en data/06_reporting/exchange_rates/)
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import generate_exchange_rates_report


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=generate_exchange_rates_report,
                inputs={
                    "fetch_status": "exchange_rates_fetch_status",
                    "rates": "exchange_rates",
                    "start_date": "params:historical_period.start_date",
                    "end_date": "params:historical_period.end_date",
                },
                outputs="exchange_rates_report",
                name="generate_exchange_rates_report",
            ),
        ],
        tags=["report_exchange_rates"],
    )
