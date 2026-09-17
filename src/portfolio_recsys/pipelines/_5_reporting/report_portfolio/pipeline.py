"""Pipeline report_portfolio — Backtest de carteras Markowitz.

Este pipeline consume predicciones del modelo y precios historicos para
ejecutar un backtest de carteras optimizadas por Markowitz sobre el
periodo de evaluacion (test por defecto).

Se ejecuta una vez por combinacion (window, segmentation, horizon). Los
identificadores de dataset se construyen con portfolio_recsys.paths.dataset_name().

Uso: se invoca via CLI (pr-backtest-all) inyectando los ejes por
runtime_params, o directamente con kedro run seleccionando por tags.
"""

from kedro.pipeline import Pipeline, node, pipeline

from portfolio_recsys.paths import Segmentation, dataset_name

from .nodes import generate_portfolio_report, run_markowitz_backtest


def create_pipeline(
    window: str = "w60",
    segmentation: str = "unified",
    sector: str | None = None,
    horizon: str | None = None,
    dataset_type: str | None = None,
    **kwargs,
) -> Pipeline:
    """Crea el pipeline de backtesting de carteras Markowitz.

    Args:
        window: Clave de ventana (ej: "w60").
        segmentation: "unified" o "by_sector".
        sector: Nombre canonico del sector (si segmentation="by_sector").
        horizon: Horizonte de prediccion. Requerido.
        dataset_type: "market"/"enriched" para construir la cartera a partir de
            las predicciones de ese dataset concreto (consume
            prediction_table_test_{dataset_type} y produce
            portfolio_backtest_results_{dataset_type} / portfolio_report_{dataset_type}).
            Si es None, usa el layout clasico (prediction_table_test, etc.).
    """
    if horizon is None:
        raise ValueError("report_portfolio requiere 'horizon'.")

    seg = (
        Segmentation.unified()
        if segmentation == "unified"
        else Segmentation.for_sector(sector)
    )

    def ds(base: str) -> str:
        return dataset_name(base, window, seg, horizon)

    # Sufijo de tipo de dataset en los nombres base (market / enriched) o vacio.
    suffix = f"_{dataset_type}" if dataset_type else ""
    token = f"{seg.dataset_token}__{window}__{horizon}"
    if dataset_type:
        token = f"{dataset_type}__{token}"

    return pipeline(
        [
            node(
                func=run_markowitz_backtest,
                inputs={
                    # Predicciones sobre el split de evaluacion out-of-sample
                    # (TEST): periodo posterior a validation_end_date con el
                    # modelo congelado. Generada por el entrenamiento/pr-predict-test.
                    "prediction_table": ds(f"prediction_table_test{suffix}"),
                    "stock_prices_eur": "stock_prices_eur",
                    "benchmark_prices_eur": "benchmark_prices_eur",
                    "split_configuration": ds("split_configuration"),
                    "markowitz_params": "params:markowitz_portfolio",
                    "horizon_key": f"params:compute_model_features.horizons.{horizon}.horizon_key",
                    "horizon_config": f"params:compute_model_features.horizons.{horizon}",
                    "evaluation_split": "params:report_portfolio.evaluation_split",
                },
                outputs=ds(f"portfolio_backtest_results{suffix}"),
                name=f"run_markowitz_backtest__{token}",
                tags=[window, seg.dataset_token, horizon, seg.kind],
            ),
            node(
                func=generate_portfolio_report,
                inputs={
                    "backtest_results": ds(f"portfolio_backtest_results{suffix}"),
                    "markowitz_params": "params:markowitz_portfolio",
                },
                outputs=ds(f"portfolio_report{suffix}"),
                name=f"generate_portfolio_report__{token}",
                tags=[window, seg.dataset_token, horizon, seg.kind],
            ),
        ],
        tags=["report_portfolio"],
    )
