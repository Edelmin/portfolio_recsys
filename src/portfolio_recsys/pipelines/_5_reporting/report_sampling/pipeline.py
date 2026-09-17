"""Pipeline report_sampling.

Dominio de reporting `sampling`: responde "¿que empresas descargar? ¿es
representativa la muestra?". Genera un informe Markdown (con checkboxes) a
partir de las muestras estratificadas ya persistidas.

Datasets:
  - Input:  SAMPLE_{cap}_{canonical}   (ej: SAMPLE_LargeCaps_Financials)
  - Output: sample_report              (Markdown en data/06_reporting/stratified_sampling/)

La conversion a PDF es una accion manual via CLI (`pr-report-pdf`), fuera del DAG.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import generate_sample_report
from portfolio_recsys.pipelines.sector_mapping import SECTORS, CAPS


def create_pipeline(**kwargs) -> Pipeline:
    # Un input por cada muestra (cap x sector). La key "{cap}_{canonical}" es la
    # misma que espera generate_sample_report para titular cada bloque.
    inputs = {
        f"{cap}_{canonical}": f"SAMPLE_{cap}_{canonical}"
        for cap in CAPS
        for canonical in SECTORS
    }

    return pipeline(
        [
            node(
                func=generate_sample_report,
                inputs=inputs,
                outputs="sample_report",
                name="generate_sample_report",
            ),
        ],
        tags=["report_sampling"],
    )
