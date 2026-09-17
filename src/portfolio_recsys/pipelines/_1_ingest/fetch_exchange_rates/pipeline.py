"""Pipeline fetch_exchange_rates.

Descarga series historicas de pares de divisa desde Yahoo Finance
para el periodo configurado en parameters.yaml.

Datasets:
  - Input:  currencies_{canonical} (11 sectores)
  - Output: exchange_rates (DataFrame con date, currency, rate)
            exchange_rates_fetch_status (estado de cada par descargado)
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import fetch_exchange_rates
from portfolio_recsys.pipelines.sector_mapping import SECTORS_WITH_FS


def create_pipeline(**kwargs) -> Pipeline:
    # Construir inputs: todos los currencies_* + parametros
    inputs = {
        f"currencies_{canonical}": f"currencies_{canonical}"
        for canonical in SECTORS_WITH_FS
    }
    inputs["start_date"] = "params:historical_period.start_date"
    inputs["end_date"] = "params:historical_period.end_date"

    return pipeline(
        [
            node(
                func=fetch_exchange_rates,
                inputs=inputs,
                outputs={
                    "rates": "exchange_rates",
                    "fetch_status": "exchange_rates_fetch_status",
                },
                name="fetch_exchange_rates",
            ),
        ],
        tags=["fetch_exchange_rates"],
    )
