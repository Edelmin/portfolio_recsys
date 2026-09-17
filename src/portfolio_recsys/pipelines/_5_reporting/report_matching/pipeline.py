"""Pipeline report_matching.

Dominio de reporting `matching`: responde "¿se emparejaron bien sectores y
divisas? ¿que se perdio?". Genera un informe Markdown de discrepancias a partir
de los datasets enriquecidos ya persistidos.

Datasets:
  - Input:  ENRICHED_{cap}_{canonical}
  - Output: enrichment_matching_report  (Markdown en data/06_reporting/enrich_sectors/)
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import generate_matching_report
from portfolio_recsys.pipelines.sector_mapping import SECTORS_WITH_FS, CAPS


def create_pipeline(**kwargs) -> Pipeline:
    # Un input por cada dataset enriquecido (cap x sector). La key con prefijo
    # "enriched_" es la que espera generate_matching_report.
    report_inputs = {
        f"enriched_{cap}_{canonical}": f"ENRICHED_{cap}_{canonical}"
        for cap in CAPS
        for canonical in SECTORS_WITH_FS
    }

    return pipeline(
        [
            node(
                func=generate_matching_report,
                inputs=report_inputs,
                outputs="enrichment_matching_report",
                name="generate_matching_report",
            ),
        ],
        tags=["report_matching"],
    )
