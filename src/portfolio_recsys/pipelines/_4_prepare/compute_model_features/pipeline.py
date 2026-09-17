"""Pipeline compute_model_features.

Genera features de mercado y fundamentales, aplica split temporal y
escalado robusto para producir datasets listos para ventanas temporales.

Datasets (identificadores con 3 ejes: window / segmentation / horizon):
  - Input:  COMPANY_{sector} (11 sectores)
  - Output: market_prepared__{seg}__{window}__{horizon}, enriched_prepared__...
            market_preprocessor__..., enriched_preprocessor__...
            feature_configuration__..., split_configuration__...

Se genera una instancia por combinacion (window, segmentation, horizon).

Segmentaciones:
  - unified   : concatena todos los COMPANY_{sector}; opcionalmente one-hot sector.
  - by_sector : un unico COMPANY_{sector}; escalado con estadisticas del sector.
"""

from kedro.pipeline import Pipeline, node, pipeline

from portfolio_recsys.paths import Segmentation, dataset_name
from portfolio_recsys.pipelines.sector_mapping import SECTORS_WITH_FS

from .nodes import compute_model_features

# Horizontes soportados (deben coincidir con parameters.yaml)
HORIZONS = ["1d", "1w", "2w", "1m", "2m", "3m", "6m", "1y"]


def create_pipeline(
    window: str = "w60",
    segmentation: str = "unified",
    sector: str | None = None,
    horizon: str | None = None,
    **kwargs,
) -> Pipeline:
    """Crea el pipeline de feature engineering para una combinacion de ejes.

    Args:
        window: Clave de ventana (ej: "w60", "w20").
        segmentation: "unified" o "by_sector".
        sector: Nombre canonico del sector (requerido si segmentation="by_sector").
        horizon: Horizonte de prediccion (ej: "3m"). Requerido.
    """
    if horizon is None:
        raise ValueError("compute_model_features requiere 'horizon'.")

    seg = (
        Segmentation.unified()
        if segmentation == "unified"
        else Segmentation.for_sector(sector)  # valida el sector
    )

    # Inputs de empresa:
    #   - unified   -> todos los COMPANY_{sector}
    #   - by_sector -> solo el COMPANY_{sector} correspondiente
    if seg.kind == "unified":
        company_inputs = {
            f"company_{s}": f"COMPANY_{s}" for s in SECTORS_WITH_FS
        }
    else:
        company_inputs = {f"company_{seg.sector}": f"COMPANY_{seg.sector}"}

    # Nombres de outputs con los 3 ejes
    def out(base: str) -> str:
        return dataset_name(base, window, seg, horizon)

    output_names = {
        "market_prepared": out("market_prepared"),
        "enriched_prepared": out("enriched_prepared"),
        "market_preprocessor": out("market_preprocessor"),
        "enriched_preprocessor": out("enriched_preprocessor"),
        "feature_configuration": out("feature_configuration"),
        "split_configuration": out("split_configuration"),
    }

    node_name = f"compute_model_features__{seg.dataset_token}__{window}__{horizon}"

    inputs = {
        **company_inputs,
        "feature_config": "params:compute_model_features.feature_engineering",
        "split_config": "params:compute_model_features.temporal_split",
        # Defaults globales de ventana (max_calendar_gap_days) + rama de window
        # (sequence_length). Se fusionan en el wrapper.
        "window_defaults": "params:compute_model_features.windows",
        "window_branch": f"params:compute_model_features.window_experiments.{window}",
        "price_quality": "params:price_quality",
        "horizon_config": f"params:compute_model_features.horizons.{horizon}",
        "segmentation_params": "params:compute_model_features.segmentation",
    }

    return pipeline(
        [
            node(
                func=_make_wrapper(seg.kind),
                inputs=inputs,
                outputs=output_names,
                name=node_name,
                tags=[window, seg.dataset_token, horizon, seg.kind],
            ),
        ],
        tags=["compute_model_features"],
        # Namespace-free: los nombres ya son unicos por los 3 ejes.
    )


def _make_wrapper(segmentation_kind: str):
    """Crea un wrapper de nodo con la segmentacion fijada en el closure.

    El segmentation_kind se conoce en tiempo de construccion del pipeline (no
    en runtime), por lo que se fija aqui de forma determinista en lugar de
    inferirlo del numero de datasets.
    """

    def _wrap(
        feature_config: dict,
        split_config: dict,
        window_defaults: dict,
        window_branch: dict,
        horizon_config: dict,
        price_quality: dict | None = None,
        segmentation_params: dict | None = None,
        **company_datasets,
    ) -> dict:
        datasets = {}
        prefix = "company_"
        for key, value in company_datasets.items():
            if key.startswith(prefix):
                datasets[key[len(prefix):]] = value

        # Config de ventana: gap global + sequence_length de la rama window.
        window_config = {
            "max_calendar_gap_days": window_defaults.get("max_calendar_gap_days", 10),
            "sequence_length": window_branch["sequence_length"],
        }

        add_sector_features = bool(
            (segmentation_params or {}).get("add_sector_features", False)
        )

        # Override holding_period_sessions con el del horizonte
        fe = {**feature_config}
        fe["holding_period_sessions"] = horizon_config["holding_period_sessions"]

        return compute_model_features(
            company_datasets=datasets,
            feature_config=fe,
            split_config=split_config,
            window_config=window_config,
            price_quality=price_quality,
            segmentation_kind=segmentation_kind,
            add_sector_features=add_sector_features,
        )

    return _wrap
