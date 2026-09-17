"""Pipeline validate_price_currencies.

Convierte los valores monetarios de los estados financieros limpios a EUR
usando join_asof con los tipos de cambio historicos.

Datasets:
  - Input:  Parquets en 02_3_1. CleanFinancialStatements/{cap}/{sector}/{ticker}.parquet
            exchange_rates (date, currency, rate_to_eur)
  - Output: Parquets en 02_3_2. FinancialStatementsEUR/{cap}/{sector}/{ticker}.parquet
            fs_conversion_stats (MemoryDataset con estadisticas)
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import convert_financial_statements_to_eur


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=convert_financial_statements_to_eur,
                inputs={
                    "clean_fs_base_path": "params:clean_financial_statements.output_base_path",
                    "output_base_path": "params:validate_price_currencies.output_base_path",
                    "cap": "params:parse_financial_statements.cap",
                    "exchange_rates": "exchange_rates",
                },
                outputs="fs_conversion_stats",
                name="convert_financial_statements_to_eur",
            ),
        ],
        tags=["validate_price_currencies"],
    )
