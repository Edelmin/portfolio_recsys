"""Pipeline enrich_sectors_with_currencies.

Enriquece los datasets de empresas (SECTOR_*) con la divisa extraida
por extract_currencies_from_filenames.

El informe de discrepancias del matching se genera en el pipeline de reporting
`report_matching`, no aqui.

Datasets:
  - Input:  SECTOR_{cap}_{canonical} + currencies_{canonical}
  - Output: ENRICHED_{cap}_{canonical}
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import enrich_sector_with_currency
from portfolio_recsys.pipelines.sector_mapping import (
    SECTORS_WITH_FS,
    CAPS,
)


def create_pipeline(**kwargs) -> Pipeline:
    nodes = []

    for cap in CAPS:
        for canonical in SECTORS_WITH_FS:
            enriched_name = f"ENRICHED_{cap}_{canonical}"

            nodes.append(
                node(
                    func=enrich_sector_with_currency,
                    inputs={
                        "sector_df": f"SECTOR_{cap}_{canonical}",
                        "currencies_df": f"currencies_{canonical}",
                    },
                    outputs=enriched_name,
                    name=f"enrich_{cap.lower()}_{canonical.lower()}",
                )
            )

    return pipeline(nodes, tags=["enrich_sectors"])
