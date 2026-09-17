"""Pipeline parse_financials_from_html.

Parsea ficheros HTML descargados manualmente que contienen tablas con
informacion de empresas cotizadas (simbolo, nombre, industria, valor de mercado,
sector, ubicacion), organizadas por sector GICS y segmento de capitalizacion
(large-cap / small-cap).

Datasets:
  - Input:  HTML_{cap}_{folder_html}       (ej: HTML_LargeCaps_BienesRaices)
  - Output: SECTOR_{cap}_{canonical}       (ej: SECTOR_LargeCaps_RealEstate)
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import extract_data_from_html
from portfolio_recsys.pipelines.sector_mapping import SECTORS, CAPS


def create_pipeline(**kwargs) -> Pipeline:
    nodes = []
    for cap in CAPS:
        for canonical, info in SECTORS.items():
            nodes.append(
                node(
                    func=extract_data_from_html,
                    inputs=f"HTML_{cap}_{info.folder_html}",
                    outputs=f"SECTOR_{cap}_{canonical}",
                    name=f"parse_{cap.lower()}_{canonical.lower()}",
                )
            )

    return pipeline(nodes, tags=["parse_financials"])
