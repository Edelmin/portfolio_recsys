"""Pipeline report_data_quality.

Analiza la cobertura y completitud de los estados financieros.

Datasets:
  - Input:  Parametros (paths, cap, periodo)
  - Output: data_quality_report (JSON en data/06_reporting/data_quality/)
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import generate_data_quality_report


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=generate_data_quality_report,
                inputs={
                    "fs_base_path": "params:parse_financial_statements.output_base_path",
                    "cap": "params:parse_financial_statements.cap",
                    "start_date": "params:historical_period.start_date",
                    "end_date": "params:historical_period.end_date",
                },
                outputs="data_quality_report",
                name="generate_data_quality_report",
            ),
        ],
        tags=["report_data_quality"],
    )
