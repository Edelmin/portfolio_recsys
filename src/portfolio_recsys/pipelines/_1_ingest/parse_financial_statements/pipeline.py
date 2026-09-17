"""Pipeline parse_financial_statements.

Convierte los ficheros Excel de estados financieros historicos (.xlsx)
a formato Parquet sin modificar los datos. Cada hoja de cada Excel
se guarda como un fichero Parquet individual.

Input:  data/01_raw/01.2 - FinancialStatementHistorical/{sector}/*.xlsx
Output: data/02_intermediate/02_3_0. RawFinancialStatements_Parquet/{cap}/{SectorCanonical}/*.parquet
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import convert_sector_excel_to_parquet


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=convert_sector_excel_to_parquet,
                inputs=[
                    "params:parse_financial_statements.raw_base_path",
                    "params:parse_financial_statements.output_base_path",
                    "params:parse_financial_statements.cap",
                ],
                outputs="parse_fs_stats",
                name="convert_excel_to_parquet",
                tags=["parse_financial_statements"],
            ),
        ],
        tags=["parse_financial_statements"],
    )
