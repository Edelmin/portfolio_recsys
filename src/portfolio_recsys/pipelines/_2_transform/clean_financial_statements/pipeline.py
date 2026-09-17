"""Pipeline clean_financial_statements.

Transforma los Parquets crudos de estados financieros a formato limpio
transpuesto, filtrando campos seleccionados. Produce un Parquet por empresa.

Datasets:
  - Input:  Parquets en 02_3_0. RawFinancialStatements_Parquet/{cap}/{sector}/
  - Output: Parquets en 02_3_1. CleanFinancialStatements/{cap}/{sector}/{ticker}.parquet
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import clean_financial_statements


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_financial_statements,
                inputs={
                    "fs_base_path": "params:parse_financial_statements.output_base_path",
                    "output_base_path": "params:clean_financial_statements.output_base_path",
                    "cap": "params:parse_financial_statements.cap",
                    "selected_fields": "params:build_company_dataset.selected_fields",
                },
                outputs="clean_fs_stats",
                name="clean_financial_statements",
            ),
        ],
        tags=["clean_financial_statements"],
    )
