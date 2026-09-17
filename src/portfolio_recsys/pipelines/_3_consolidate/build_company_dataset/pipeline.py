"""Pipeline build_company_dataset.

Consolida precios EUR + estados financieros (ya en EUR) en un dataset por sector.

Datasets:
  - Input:  stock_prices_eur, ENRICHED_{cap}_{canonical},
            FS_EUR_{cap}_{canonical} (PartitionedDataset, un Parquet por empresa)
  - Output: COMPANY_{canonical} (un Parquet por sector en data/03_processed/)

El pipeline procesa un unico segmento de capitalizacion por ejecucion, definido
por ``params:parse_financial_statements.cap``. Los estados financieros en EUR se
consumen desde el catalogo (factory ``FS_EUR_{cap}_{canonical}``), no por lectura
directa del filesystem.
"""

from pathlib import Path

import yaml
from kedro.pipeline import Pipeline, node, pipeline

from portfolio_recsys.pipelines.sector_mapping import CAPS, SECTORS_WITH_FS

from .nodes import build_company_dataset


def _resolve_active_cap() -> str:
    """Lee el cap activo desde la configuracion del proyecto.

    Devuelve ``parse_financial_statements.cap`` (por defecto "LargeCaps"),
    resuelto en tiempo de construccion del pipeline para inyectar unicamente los
    datasets FS_EUR del cap que se va a procesar (evita exigir en el DAG los FS de
    un cap que aun no existe en disco).

    IMPORTANTE: se lee el YAML de parametros directamente (sin instanciar un
    OmegaConfigLoader), porque crear un segundo config loader aqui re-registraria
    el resolver global ``${globals:...}`` de OmegaConf apuntando a una instancia
    con un entorno incompleto, rompiendo la resolucion de globals en todo el
    proyecto. El parametro ``cap`` es un escalar simple sin interpolaciones, asi
    que un yaml.safe_load es suficiente y seguro.
    """
    project_path = Path(__file__).resolve().parents[4]
    params_path = project_path / "conf" / "base" / "parameters.yaml"
    cap = "LargeCaps"
    try:
        with params_path.open(encoding="utf-8") as fh:
            parameters = yaml.safe_load(fh) or {}
        cap = (
            parameters.get("parse_financial_statements", {}).get("cap", "LargeCaps")
        )
    except Exception:  # noqa: BLE001 - si la config no es accesible, usar el defecto
        cap = "LargeCaps"

    if cap not in CAPS:
        raise ValueError(
            f"cap='{cap}' no es valido. Valores permitidos: {sorted(CAPS)}"
        )
    return cap


def create_pipeline(**kwargs) -> Pipeline:
    active_cap = _resolve_active_cap()

    # Inputs fijos + parametro cap (como literal, para el nodo).
    inputs = {
        "stock_prices_eur": "stock_prices_eur",
        "ticker_mapping": "ticker_mapping",
        "cap": "params:parse_financial_statements.cap",
    }

    # ENRICHED_* de ambos caps (para el mapeo ticker -> sector; existen en disco).
    for cap in CAPS:
        for canonical in SECTORS_WITH_FS:
            inputs[f"{cap}_{canonical}"] = f"ENRICHED_{cap}_{canonical}"

    # FS en EUR del cap activo, desde el catalogo (PartitionedDataset por sector).
    # Prefijo fs_eur__ para distinguir estos kwargs en el nodo.
    for canonical in SECTORS_WITH_FS:
        inputs[f"fs_eur__{canonical}"] = f"FS_EUR_{active_cap}_{canonical}"

    # Construir outputs: un dataset por sector
    outputs = {
        sector: f"COMPANY_{sector}" for sector in SECTORS_WITH_FS
    }

    return pipeline(
        [
            node(
                func=build_company_dataset,
                inputs=inputs,
                outputs=outputs,
                name="build_company_dataset",
            ),
        ],
        tags=["build_company_dataset"],
    )
