"""Pipeline stratified_random_sampling.

Selecciona una muestra estratificada de empresas por sector y capitalizacion,
garantizando representatividad estadistica (confianza 95%, error 5%).

Datasets:
  - Input:  SECTOR_{cap}_{canonical}   (ej: SECTOR_LargeCaps_Financials)
  - Output: SAMPLE_{cap}_{canonical}   (ej: SAMPLE_LargeCaps_Financials)

El informe Markdown de la muestra (para descarga manual) se genera en el
pipeline de reporting `report_sampling`, no aqui.

Parametros (conf/base/parameters.yaml):
  stratified_random_sampling:
    extra_samples_override:
      LargeCaps_Energy: 15       # Ampliar muestra de este sector en 15
      SmallCaps_Healthcare: 10
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import get_sample
from portfolio_recsys.pipelines.sector_mapping import SECTORS, CAPS


def create_pipeline(**kwargs) -> Pipeline:
    nodes = []

    for cap in CAPS:
        for canonical in SECTORS:
            dataset_name = f"SAMPLE_{cap}_{canonical}"
            key = f"{cap}_{canonical}"
            nodes.append(
                node(
                    func=get_sample,
                    inputs=[
                        f"SECTOR_{cap}_{canonical}",
                        f"params:stratified_random_sampling.extra_samples_override.{key}",
                    ],
                    outputs=dataset_name,
                    name=f"sample_{cap.lower()}_{canonical.lower()}",
                )
            )

    return pipeline(nodes, tags=["stratified_sample"])
