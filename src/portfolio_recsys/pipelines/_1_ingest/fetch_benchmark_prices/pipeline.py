"""Pipeline fetch_benchmark_prices.

Descarga el indice de mercado (S&P 500) y genera la serie del activo seguro,
usando el mismo periodo historico compartido con el resto de precios.

Datasets:
  Inputs:  params:historical_period.*, params:markowitz_portfolio.benchmark
  Outputs: benchmark_prices, benchmark_prices_fetch_status
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import fetch_benchmark_prices


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=fetch_benchmark_prices,
                inputs={
                    "start_date": "params:historical_period.start_date",
                    "end_date": "params:historical_period.end_date",
                    "benchmark_params": "params:markowitz_portfolio.benchmark",
                },
                outputs={
                    "prices": "benchmark_prices",
                    "fetch_status": "benchmark_prices_fetch_status",
                },
                name="fetch_benchmark_prices",
            ),
        ],
        tags=["fetch_benchmark_prices"],
    )
