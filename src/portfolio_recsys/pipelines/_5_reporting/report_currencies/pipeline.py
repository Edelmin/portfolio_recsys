"""Pipeline report_currencies.

Genera un JSON con la distribución de empresas por divisa.

Datasets:
  - Input:  currencies_{canonical} (11 sectores)
  - Output: currency_report (JSON en data/06_reporting/currencies/)
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import generate_currency_report
from portfolio_recsys.pipelines.sector_mapping import SECTORS_WITH_FS


def create_pipeline(**kwargs) -> Pipeline:
    # Mapear cada sector como input
    inputs = {
        canonical: f"currencies_{canonical}"
        for canonical in SECTORS_WITH_FS
    }

    return pipeline(
        [
            node(
                func=generate_currency_report,
                inputs=inputs,
                outputs="currency_report",
                name="generate_currency_report",
            ),
        ],
        tags=["report_currencies"],
    )
