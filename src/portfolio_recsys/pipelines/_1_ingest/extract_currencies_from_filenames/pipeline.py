"""Pipeline extract_currencies_from_filenames.

Extrae la moneda, ticker y periodo de cobertura de cada empresa a partir
de los nombres de fichero de los estados financieros historicos.

Las fechas se almacenan en ISO 8601 UTC (Datetime[us, UTC]).

Datasets:
  - Input:  FS_{folder_html}           (ej: FS_Financiero)
  - Output: currencies_{canonical}     (ej: currencies_Financials)
"""

from functools import partial, update_wrapper

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import extract_currencies
from portfolio_recsys.pipelines.sector_mapping import (
    SECTORS,
    SECTORS_WITH_FS,
)


def create_pipeline(**kwargs) -> Pipeline:
    nodes = []

    for canonical in SECTORS_WITH_FS:
        info = SECTORS[canonical]
        fs_key = info.folder_html

        func = partial(extract_currencies, sector=canonical)
        update_wrapper(func, extract_currencies)
        nodes.append(
            node(
                func=func,
                inputs=f"FS_{fs_key}",
                outputs=f"currencies_{canonical}",
                name=f"extract_currencies_{canonical.lower()}",
            )
        )

    return pipeline(nodes, tags=["extract_currencies"])
