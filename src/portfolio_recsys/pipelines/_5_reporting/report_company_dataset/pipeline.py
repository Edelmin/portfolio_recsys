"""Pipeline report_company_dataset.

Genera un informe de calidad y métricas sobre los datasets consolidados
(COMPANY_{sector}) para todos los sectores disponibles.

Datasets:
  - Input:  COMPANY_{sector} (uno por sector)
  - Output: company_dataset_report (JSON en data/06_reporting/company_dataset/)
"""

from kedro.pipeline import Pipeline, node, pipeline

from portfolio_recsys.pipelines.sector_mapping import SECTORS_WITH_FS

from .nodes import generate_company_dataset_report


def create_pipeline(**kwargs) -> Pipeline:
    # Construir inputs: todos los COMPANY_* disponibles
    inputs = {}
    for sector in SECTORS_WITH_FS:
        inputs[sector] = f"COMPANY_{sector}"

    return pipeline(
        [
            node(
                func=generate_company_dataset_report,
                inputs=inputs,
                outputs="company_dataset_report",
                name="generate_company_dataset_report",
            ),
        ],
        tags=["report_company_dataset"],
    )
