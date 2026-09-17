"""Pipeline impute_missing_values.

Rellena huecos en los datasets consolidados por sector (COMPANY_*).
Produce datasets listos para modelado (COMPANY_IMPUTED_{sector}).

Datasets:
  - Input:  COMPANY_{sector}
  - Output: COMPANY_IMPUTED_{sector}
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import impute_sector
from portfolio_recsys.pipelines.sector_mapping import SECTORS_WITH_FS


def create_pipeline(**kwargs) -> Pipeline:
    nodes = []

    for sector in SECTORS_WITH_FS:
        nodes.append(
            node(
                func=impute_sector,
                inputs=f"COMPANY_{sector}",
                outputs=f"COMPANY_IMPUTED_{sector}",
                name=f"impute_{sector.lower()}",
            )
        )

    return pipeline(nodes, tags=["impute_missing_values"])
