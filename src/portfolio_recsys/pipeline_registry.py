"""Registro de pipelines del proyecto Portfolio Recsys.

Organizacion por etapas:
  1_ingest     → Ingesta de datos desde fuentes externas
  2_transform  → Limpieza, normalizacion, validacion, muestreo
  3_consolidate → Join de datasets en vista unificada
  4_prepare    → Formateo para modelos (ventanas temporales)
"""

# --- Etapa 1: Ingesta ---
from .pipelines._1_ingest.parse_financials_from_html.pipeline import (
    create_pipeline as parse_financials_from_html_gen,
)
from .pipelines._1_ingest.parse_financial_statements.pipeline import (
    create_pipeline as parse_financial_statements_gen,
)
from .pipelines._1_ingest.extract_currencies_from_filenames.pipeline import (
    create_pipeline as extract_currencies_from_filenames_gen,
)
from .pipelines._1_ingest.fetch_exchange_rates.pipeline import (
    create_pipeline as fetch_exchange_rates_gen,
)
from .pipelines._1_ingest.fetch_stock_prices.pipeline import (
    create_pipeline as fetch_stock_prices_gen,
)
from .pipelines._1_ingest.fetch_benchmark_prices.pipeline import (
    create_pipeline as fetch_benchmark_prices_gen,
)

# --- Etapa 2: Transformacion ---
from .pipelines._2_transform.stratified_random_sampling.pipeline import (
    create_pipeline as stratified_random_sampling_gen,
)
from .pipelines._2_transform.enrich_sectors_with_currencies.pipeline import (
    create_pipeline as enrich_sectors_with_currencies_gen,
)
from .pipelines._2_transform.clean_financial_statements.pipeline import (
    create_pipeline as clean_financial_statements_gen,
)
from .pipelines._2_transform.normalize_currencies.pipeline import (
    create_pipeline as normalize_currencies_gen,
)
from .pipelines._2_transform.validate_price_currencies.pipeline import (
    create_pipeline as validate_price_currencies_gen,
)

# --- Etapa 3: Consolidacion ---
from .pipelines._3_consolidate.build_company_dataset.pipeline import (
    create_pipeline as build_company_dataset_gen,
)

# --- Etapa 4: Preparacion para modelo ---
from .pipelines._4_prepare.compute_model_features.pipeline import (
    create_pipeline as compute_model_features_gen,
)
from .pipelines._4_prepare.build_temporal_windows.pipeline import (
    create_pipeline as build_temporal_windows_gen,
)
from .pipelines._4_prepare.impute_missing_values.pipeline import (
    create_pipeline as impute_missing_values_gen,
)

# --- Etapa 5: Reporting ---
from .pipelines._5_reporting.report_sampling.pipeline import (
    create_pipeline as report_sampling_gen,
)
from .pipelines._5_reporting.report_matching.pipeline import (
    create_pipeline as report_matching_gen,
)
from .pipelines._5_reporting.report_currencies.pipeline import (
    create_pipeline as report_currencies_gen,
)
from .pipelines._5_reporting.report_exchange_rates.pipeline import (
    create_pipeline as report_exchange_rates_gen,
)
from .pipelines._5_reporting.report_stock_prices.pipeline import (
    create_pipeline as report_stock_prices_gen,
)
from .pipelines._5_reporting.report_normalization.pipeline import (
    create_pipeline as report_normalization_gen,
)
from .pipelines._5_reporting.report_data_quality.pipeline import (
    create_pipeline as report_data_quality_gen,
)
from .pipelines._5_reporting.report_company_dataset.pipeline import (
    create_pipeline as report_company_dataset_gen,
)
from .pipelines._5_reporting.report_portfolio.pipeline import (
    create_pipeline as report_portfolio_gen,
)
from .pipelines._5_reporting.observability.pipeline import (
    create_pipeline as observability_gen,
)

from .pipelines.sector_mapping import SECTORS_WITH_FS


def register_pipelines():
    # Pipelines implementados
    parse_financials_from_html = parse_financials_from_html_gen()
    stratified_random_sampling = stratified_random_sampling_gen()

    # Resto de pipelines de ingesta, transformacion y consolidacion
    parse_financial_statements = parse_financial_statements_gen()
    extract_currencies_from_filenames = extract_currencies_from_filenames_gen()
    enrich_sectors_with_currencies = enrich_sectors_with_currencies_gen()
    clean_financial_statements = clean_financial_statements_gen()
    fetch_exchange_rates = fetch_exchange_rates_gen()
    fetch_stock_prices = fetch_stock_prices_gen()
    fetch_benchmark_prices = fetch_benchmark_prices_gen()
    normalize_currencies = normalize_currencies_gen()
    validate_price_currencies = validate_price_currencies_gen()
    build_company_dataset = build_company_dataset_gen()
    # --- Etapa 4: Preparacion para modelo ---
    # Los pipelines de modelado son multi-eje (window/segmentation/horizon) y se
    # construyen por combinacion (ver seccion siguiente), no como instancia unica.
    impute_missing_values = impute_missing_values_gen()
    report_sampling = report_sampling_gen()
    report_matching = report_matching_gen()
    report_currencies = report_currencies_gen()
    report_exchange_rates = report_exchange_rates_gen()
    report_stock_prices = report_stock_prices_gen()
    report_normalization = report_normalization_gen()
    report_data_quality = report_data_quality_gen()
    report_company_dataset = report_company_dataset_gen()
    observability = observability_gen()

    # --- Pipelines por combinacion (window, segmentation, horizon) ---
    #
    # Enfoque B: en lugar de registrar las ~192 combinaciones posibles, se
    # registra una plantilla parametrizada por combinacion SOLO para las que
    # el CLI necesita. El orquestador pr-prepare-datasets construye los
    # nombres de pipeline con la convencion:
    #
    #   prepare__{segmentation_token}__{window}__{horizon}
    #   report_portfolio__{segmentation_token}__{window}__{horizon}
    #
    # y los ejecuta programaticamente. Ver build_prepare_pipeline() /
    # build_report_portfolio_pipeline() como fabricas reutilizables.
    horizons = ["1d", "1w", "2w", "1m", "2m", "3m", "6m", "1y"]
    windows = ["w60", "w20"]

    # Pipeline completo de ingesta/transformacion/consolidacion (sin la capa de
    # modelado, que ahora es multi-eje y se orquesta por CLI).
    full_pipeline = (
        parse_financials_from_html
        + stratified_random_sampling
        + parse_financial_statements
        + extract_currencies_from_filenames
        + enrich_sectors_with_currencies
        + clean_financial_statements
        + fetch_exchange_rates
        + fetch_stock_prices
        + fetch_benchmark_prices
        + normalize_currencies
        + validate_price_currencies
        + build_company_dataset
        + report_sampling
        + report_matching
    )

    registry = {
        "__default__": full_pipeline,
        # Etapa 1
        "parse_financials_from_html": parse_financials_from_html,
        "parse_financial_statements": parse_financial_statements,
        "extract_currencies_from_filenames": extract_currencies_from_filenames,
        "fetch_exchange_rates": fetch_exchange_rates,
        "fetch_stock_prices": fetch_stock_prices,
        "fetch_benchmark_prices": fetch_benchmark_prices,
        # Etapa 2
        "stratified_random_sampling": stratified_random_sampling,
        "enrich_sectors_with_currencies": enrich_sectors_with_currencies,
        "clean_financial_statements": clean_financial_statements,
        "normalize_currencies": normalize_currencies,
        "validate_price_currencies": validate_price_currencies,
        # Etapa 3
        "build_company_dataset": build_company_dataset,
        # Etapa 4
        "impute_missing_values": impute_missing_values,
        # Etapa 5: Reporting
        "report_sampling": report_sampling,
        "report_matching": report_matching,
        "report_currencies": report_currencies,
        "report_exchange_rates": report_exchange_rates,
        "report_stock_prices": report_stock_prices,
        "report_normalization": report_normalization,
        "report_data_quality": report_data_quality,
        "report_company_dataset": report_company_dataset,
        "observability": observability,
    }

    # --- Registro de pipelines de modelado por combinacion de ejes ---
    #
    # Se registran los pipelines "prepare" (compute_model_features +
    # build_temporal_windows) y "report_portfolio" por cada combinacion
    # (window, segmentation, horizon). El identificador de pipeline sigue la
    # convencion prepare__{seg_token}__{window}__{horizon}.
    #
    # unified: una entrada por (window, horizon).
    # by_sector: una entrada por (window, sector, horizon).
    for window in windows:
        for horizon in horizons:
            # --- unified ---
            seg_token = "unified"
            prepare_u = (
                compute_model_features_gen(
                    window=window, segmentation="unified", horizon=horizon
                )
                + build_temporal_windows_gen(
                    window=window, segmentation="unified", horizon=horizon
                )
            )
            registry[f"prepare__{seg_token}__{window}__{horizon}"] = prepare_u
            registry[f"report_portfolio__{seg_token}__{window}__{horizon}"] = (
                report_portfolio_gen(
                    window=window, segmentation="unified", horizon=horizon
                )
            )
            # Carteras por tipo de dataset (market / enriched) en unified.
            for dataset_type in ("market", "enriched"):
                registry[
                    f"report_portfolio_{dataset_type}__{seg_token}__{window}__{horizon}"
                ] = report_portfolio_gen(
                    window=window, segmentation="unified", horizon=horizon,
                    dataset_type=dataset_type,
                )
            # --- by_sector (uno por sector) ---
            for sector in SECTORS_WITH_FS:
                seg_token = f"sector_{sector}"
                prepare_s = (
                    compute_model_features_gen(
                        window=window, segmentation="by_sector",
                        sector=sector, horizon=horizon,
                    )
                    + build_temporal_windows_gen(
                        window=window, segmentation="by_sector",
                        sector=sector, horizon=horizon,
                    )
                )
                registry[f"prepare__{seg_token}__{window}__{horizon}"] = prepare_s
                registry[f"report_portfolio__{seg_token}__{window}__{horizon}"] = (
                    report_portfolio_gen(
                        window=window, segmentation="by_sector",
                        sector=sector, horizon=horizon,
                    )
                )

    return registry
