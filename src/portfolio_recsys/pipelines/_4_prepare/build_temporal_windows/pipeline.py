"""Pipeline build_temporal_windows.

Construye ventanas temporales deslizantes para entrenamiento de modelos
recurrentes a partir de los datasets preprocesados.

Datasets (identificadores con 3 ejes: window / segmentation / horizon):
  - Input:  market_prepared__{seg}__{window}__{horizon}, enriched_prepared__...,
            preprocessors
  - Output: window stores (MemoryDataset), indices (Parquet) y ventanas
            materializadas (.npz)

El sequence_length de las ventanas lo determina la rama de `window`
(window_experiments.{window}.sequence_length), fusionado con el resto de
defaults de `windows`.
"""

from kedro.pipeline import Pipeline, node, pipeline

from portfolio_recsys.paths import Segmentation, dataset_name

from .nodes import (
    build_windows_from_prepared,
    extract_window_index,
    materialize_windows,
)


def _make_window_builder():
    """Wrapper que fusiona defaults de ventana + sequence_length de la rama."""

    def _build(prepared_df, preprocessor, window_defaults, window_branch, price_quality):
        pq = price_quality or {}
        window_config = {
            "max_calendar_gap_days": window_defaults.get("max_calendar_gap_days", 10),
            "sequence_length": window_branch["sequence_length"],
            # Filtro de ventanas anomalas (Medida 4 de calidad). Se consume la
            # columna 'is_anomalous_return' marcada en compute_model_features.
            "filter_anomalous_windows": bool(pq.get("filter_anomalous_windows", False)),
            "anomaly_column": "is_anomalous_return",
        }
        return build_windows_from_prepared(
            prepared_df=prepared_df,
            preprocessor=preprocessor,
            window_config=window_config,
        )

    return _build


def create_pipeline(
    window: str = "w60",
    segmentation: str = "unified",
    sector: str | None = None,
    horizon: str | None = None,
    **kwargs,
) -> Pipeline:
    """Crea el pipeline de ventanas temporales para una combinacion de ejes.

    Args:
        window: Clave de ventana (ej: "w60").
        segmentation: "unified" o "by_sector".
        sector: Nombre canonico del sector (si segmentation="by_sector").
        horizon: Horizonte de prediccion. Requerido.
    """
    if horizon is None:
        raise ValueError("build_temporal_windows requiere 'horizon'.")

    seg = (
        Segmentation.unified()
        if segmentation == "unified"
        else Segmentation.for_sector(sector)
    )

    def ds(base: str) -> str:
        return dataset_name(base, window, seg, horizon)

    token = f"{seg.dataset_token}__{window}__{horizon}"

    window_params = {
        "window_defaults": "params:compute_model_features.windows",
        "window_branch": f"params:compute_model_features.window_experiments.{window}",
        "price_quality": "params:price_quality",
    }

    return pipeline(
        [
            # Market windows
            node(
                func=_make_window_builder(),
                inputs={
                    "prepared_df": ds("market_prepared"),
                    "preprocessor": ds("market_preprocessor"),
                    **window_params,
                },
                outputs=ds("market_window_store"),
                name=f"build_market_windows__{token}",
                tags=[window, seg.dataset_token, horizon, seg.kind],
            ),
            node(
                func=extract_window_index,
                inputs=ds("market_window_store"),
                outputs=ds("market_window_index"),
                name=f"extract_market_window_index__{token}",
                tags=[window, seg.dataset_token, horizon, seg.kind],
            ),
            node(
                func=materialize_windows,
                inputs=ds("market_window_store"),
                outputs=ds("market_window"),
                name=f"materialize_market_windows__{token}",
                tags=[window, seg.dataset_token, horizon, seg.kind],
            ),
            # Enriched windows
            node(
                func=_make_window_builder(),
                inputs={
                    "prepared_df": ds("enriched_prepared"),
                    "preprocessor": ds("enriched_preprocessor"),
                    **window_params,
                },
                outputs=ds("enriched_window_store"),
                name=f"build_enriched_windows__{token}",
                tags=[window, seg.dataset_token, horizon, seg.kind],
            ),
            node(
                func=extract_window_index,
                inputs=ds("enriched_window_store"),
                outputs=ds("enriched_window_index"),
                name=f"extract_enriched_window_index__{token}",
                tags=[window, seg.dataset_token, horizon, seg.kind],
            ),
            node(
                func=materialize_windows,
                inputs=ds("enriched_window_store"),
                outputs=ds("enriched_window"),
                name=f"materialize_enriched_windows__{token}",
                tags=[window, seg.dataset_token, horizon, seg.kind],
            ),
        ],
        tags=["build_temporal_windows"],
    )
