"""Nodos del pipeline compute_model_features.

Implementa toda la cadena de feature engineering + split temporal +
escalado robusto para producir datasets listos para ventanas temporales.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import polars as pl

from portfolio_recsys.models.config import (
    FeatureEngineeringConfig,
    TemporalSplitConfig,
    WindowConfig,
)
from portfolio_recsys.models.preprocessing import (
    fit_robust_preprocessor,
    sanitize_numeric_columns,
    transform_with_preprocessor,
)
from portfolio_recsys.pipelines.sector_mapping import SECTORS_WITH_FS

logger = logging.getLogger(__name__)


def sector_one_hot_columns() -> list[str]:
    """Devuelve los nombres de las columnas one-hot de sector, en orden estable.

    El vocabulario es SECTORS_WITH_FS (fuente unica de verdad), de modo que el
    conjunto y el orden de columnas es identico entre train/validation/test y
    entre ejecuciones. Formato: 'sector_ohe_{Canonical}'.
    """
    return [f"sector_ohe_{sector}" for sector in SECTORS_WITH_FS]


def add_sector_one_hot(dataframe: pl.DataFrame) -> pl.DataFrame:
    """Agrega columnas one-hot de sector con vocabulario fijo (SECTORS_WITH_FS).

    A diferencia de un one-hot dinamico (que dependeria de los sectores
    presentes en el DataFrame), este fija las 11 columnas siempre, para que el
    modelo reciba el mismo espacio de features independientemente del split o
    del subconjunto de datos. Las columnas son binarias (Int8, 0/1).

    Args:
        dataframe: DataFrame con una columna 'sector' (nombre canonico).

    Returns:
        DataFrame con las columnas 'sector_ohe_{Canonical}' añadidas.
    """
    if "sector" not in dataframe.columns:
        raise ValueError(
            "add_sector_one_hot requiere la columna 'sector' en el DataFrame."
        )
    expressions = [
        (pl.col("sector") == sector).cast(pl.Int8).alias(f"sector_ohe_{sector}")
        for sector in SECTORS_WITH_FS
    ]
    return dataframe.with_columns(expressions)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────────────────


def _signed_log1p_expr(source_column: str, output_column: str) -> pl.Expr:
    """Calcula signed log1p: sign(x) * log1p(|x|)."""
    column = pl.col(source_column)
    return (
        pl.when(column.is_null())
        .then(None)
        .when(column >= 0)
        .then(column.log1p())
        .otherwise(-(-column).log1p())
        .alias(output_column)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Nodo principal: compute_model_features
# ─────────────────────────────────────────────────────────────────────────────


def compute_model_features(
    company_datasets: dict[str, pl.DataFrame],
    feature_config: dict[str, Any],
    split_config: dict[str, Any],
    window_config: dict[str, Any],
    price_quality: dict[str, Any] | None = None,
    segmentation_kind: str = "unified",
    add_sector_features: bool = False,
) -> dict[str, Any]:
    """Genera features, targets y preprocesa los datos para modelado.

    Segun `segmentation_kind`:
      - "unified": concatena todos los COMPANY_{sector} recibidos en un unico
        dataset. Si `add_sector_features` es True, añade one-hot de sector
        (11 columnas binarias con vocabulario fijo SECTORS_WITH_FS) como features
        del modelo. El escalado robusto se ajusta sobre el conjunto global.
      - "by_sector": recibe UN unico COMPANY_{sector} (el dict tendra una sola
        entrada). El escalado robusto se ajusta con las estadisticas propias de
        ese sector (escalado por-sector). No se añade one-hot (redundante: todas
        las filas son del mismo sector).

    Args:
        company_datasets: Dict {sector: DataFrame}. En "unified" contiene todos
            los sectores; en "by_sector" contiene exactamente uno.
        feature_config: Parametros de FeatureEngineeringConfig (dict YAML).
        split_config: Parametros de TemporalSplitConfig (dict YAML).
        window_config: Parametros de WindowConfig (dict YAML).
        price_quality: Dict de configuracion de calidad de precios. Clave
            relevante: clip_returns (bool), max_abs_log_return (float). El limite
            se escala por el numero de sesiones de cada retorno (acumulacion de
            log-retornos). Si es None, no se aplica clipping.
        segmentation_kind: "unified" o "by_sector".
        add_sector_features: Si True y segmentation_kind == "unified", añade
            one-hot de sector como features binarias. Ignorado en "by_sector".

    Returns:
        Diccionario con todos los artefactos:
        - market_prepared: DataFrame preprocesado (solo mercado).
        - enriched_prepared: DataFrame preprocesado (mercado + fundamentales).
        - market_preprocessor: Dict del preprocesador de mercado.
        - enriched_preprocessor: Dict del preprocesador enriquecido.
        - feature_configuration: Metadata de las columnas de features.
        - split_configuration: Metadata del split temporal.
    """
    if segmentation_kind not in ("unified", "by_sector"):
        raise ValueError(
            f"segmentation_kind no valido: {segmentation_kind!r}. "
            "Use 'unified' o 'by_sector'."
        )

    fe_config = FeatureEngineeringConfig.from_dict(feature_config)
    ts_config = TemporalSplitConfig.from_dict(split_config)
    win_config = WindowConfig.from_dict(window_config)

    # Concatenar los sectores recibidos (en by_sector sera uno solo)
    all_dfs = []
    for sector, df in sorted(company_datasets.items()):
        if not df.is_empty():
            all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No hay datasets de empresa disponibles.")

    if segmentation_kind == "by_sector" and len(all_dfs) != 1:
        raise ValueError(
            "segmentation_kind='by_sector' requiere exactamente un dataset de "
            f"sector; se recibieron {len(all_dfs)}."
        )

    df = pl.concat(all_dfs, how="diagonal_relaxed")
    df = df.sort(["ticker", "date"])

    # One-hot de sector: solo tiene sentido en unified (en by_sector todas las
    # filas comparten sector, seria una constante).
    use_sector_ohe = bool(add_sector_features) and segmentation_kind == "unified"
    if use_sector_ohe:
        df = add_sector_one_hot(df)

    logger.info(
        "[%s] Dataset: %d filas, %d tickers, %d columnas (one-hot sector=%s)",
        segmentation_kind, df.height, df["ticker"].n_unique(), df.shape[1],
        use_sector_ohe,
    )

    # Configuracion de clipping de retornos (winsorizacion de saltos corruptos)
    clip_returns = bool(price_quality.get("clip_returns", False)) if price_quality else False
    max_abs_daily = float(price_quality.get("max_abs_log_return", 0.40)) if price_quality else 0.40

    # ─── 1. Features de mercado ───────────────────────────────────────────

    # Retornos multi-horizonte
    return_expressions = []
    for feature_name, num_sessions in fe_config.return_horizons.items():
        previous_price = pl.col("close_eur").shift(num_sessions).over("ticker")
        return_expr = (
            pl.when(
                previous_price.is_not_null()
                & (previous_price > 0)
                & (pl.col("close_eur") > 0)
            )
            .then((pl.col("close_eur") / previous_price).log())
            .otherwise(None)
        )
        if clip_returns:
            # El limite escala con el horizonte: los log-retornos se acumulan de
            # forma aditiva, por lo que un retorno a N sesiones admite N veces el
            # limite diario. Esto recorta saltos espurios sin penalizar tendencias
            # legitimas a largo plazo.
            limit = max_abs_daily * num_sessions
            return_expr = return_expr.clip(-limit, limit)
        return_expressions.append(return_expr.alias(feature_name))

    df = df.with_columns(return_expressions)

    # Log del precio
    df = df.with_columns(
        pl.col("close_eur").log().alias("market_log_close_eur")
    )

    # Log del tipo de cambio y FX return diario
    previous_rate = pl.col("rate_to_eur").shift(1).over("ticker")
    df = df.with_columns(
        pl.when(pl.col("rate_to_eur").is_not_null() & (pl.col("rate_to_eur") > 0))
        .then(pl.col("rate_to_eur").log())
        .otherwise(None)
        .alias("market_log_rate_to_eur"),

        pl.when(
            pl.col("rate_to_eur").is_not_null()
            & previous_rate.is_not_null()
            & (pl.col("rate_to_eur") > 0)
            & (previous_rate > 0)
        )
        .then((pl.col("rate_to_eur") / previous_rate).log())
        .otherwise(None)
        .alias("market_fx_return_1d"),
    )

    # Columnas one-hot de sector (vacio si no se usan)
    sector_ohe_cols = sector_one_hot_columns() if use_sector_ohe else []

    # ── Feature de retorno de mercado: SIEMPRE el log-retorno DIARIO ──────────
    # Todos los modelos, sea cual sea su horizonte de prediccion, reciben como
    # unica variable de mercado el log-retorno diario (cierre de hoy frente al
    # cierre de ayer, market_return_1d = log(p_t / p_{t-1})). Es un retorno YA
    # REALIZADO y conocido al cierre de la sesion de senal (mira hacia atras), no
    # una rentabilidad futura. La ventana temporal de la RNN aporta asi la
    # historia de los ultimos N retornos diarios reales, homogenea entre
    # horizontes. Lo unico que cambia por horizonte es el TARGET (el retorno
    # futuro al periodo de holding), no la feature de entrada.
    #
    # No se incluyen el nivel de precio (market_log_close_eur), el nivel/retorno
    # del tipo de cambio (market_log_rate_to_eur, market_fx_return_1d) ni los
    # retornos a otras escalas: la propia ventana aporta la historia relevante.
    daily_return_col = next(
        (name for name, sessions in fe_config.return_horizons.items()
         if int(sessions) == 1),
        None,
    )
    if daily_return_col is None:
        raise ValueError(
            "No hay una feature de retorno diario (num_sessions == 1) en "
            "return_horizons. Es imprescindible para la variable de mercado."
        )

    market_feature_cols = [daily_return_col] + sector_ohe_cols

    # ─── 2. Trading targets ───────────────────────────────────────────────

    entry_shift = fe_config.execution_lag_sessions
    exit_shift = fe_config.execution_lag_sessions + fe_config.holding_period_sessions

    df = df.with_columns(
        pl.col("date").shift(-entry_shift).over("ticker").alias("trade_entry_date"),
        pl.col("close_eur").shift(-entry_shift).over("ticker").alias("trade_entry_close_eur"),
        pl.col("date").shift(-exit_shift).over("ticker").alias("trade_exit_date"),
        pl.col("close_eur").shift(-exit_shift).over("ticker").alias("trade_exit_close_eur"),
    )

    target_expr = (
        pl.col("trade_exit_close_eur") / pl.col("trade_entry_close_eur")
    ).log()
    if clip_returns:
        # Limite del target escalado por el periodo de holding (log-retornos aditivos).
        target_limit = max_abs_daily * fe_config.holding_period_sessions
        target_expr = target_expr.clip(-target_limit, target_limit)

    df = df.with_columns(target_expr.alias("target_log_return"))

    # Filtrar filas con target valido
    df = df.filter(
        pl.col("target_log_return").is_not_null()
        & pl.col("target_log_return").is_finite()
        & pl.col("trade_entry_date").is_not_null()
        & pl.col("trade_exit_date").is_not_null()
    )

    # ─── 3. Features de fundamentales ─────────────────────────────────────

    # Snapshots de fundamentales (una fila por ticker + fiscal_date)
    fundamental_value_cols = list(fe_config.fundamental_log_mapping.keys()) + list(
        fe_config.fundamental_ratio_mapping.keys()
    )
    # Solo usar columnas que existen en el DataFrame
    existing_fund_cols = [c for c in fundamental_value_cols if c in df.columns]

    if existing_fund_cols and "fiscal_date" in df.columns:
        fundamental_snapshots = (
            df.filter(pl.col("fiscal_date").is_not_null())
            .group_by(["ticker", "fiscal_date"])
            .agg([
                pl.col(c).drop_nulls().first().alias(c) for c in existing_fund_cols
            ])
            .sort(["ticker", "fiscal_date"])
            .with_columns(
                (pl.col("fiscal_date") + pl.duration(days=fe_config.fundamental_lag_days))
                .alias("fundamental_available_date")
            )
        )

        # Construir dataset de mercado (sin fundamentales)
        metadata_cols = [
            "date", "ticker", "sector", "currency_original", "close_eur",
            "trade_entry_date", "trade_entry_close_eur",
            "trade_exit_date", "trade_exit_close_eur",
            "target_log_return",
        ]

        market_model_df = df.select(
            [c for c in metadata_cols if c in df.columns] + market_feature_cols
        ).sort(["ticker", "date"])

        # Join asof para enriquecer con fundamentales
        daily_market_rows = df.select(
            [c for c in metadata_cols if c in df.columns] + market_feature_cols
        ).sort(["ticker", "date"])

        enriched_model_df = daily_market_rows.join_asof(
            fundamental_snapshots,
            left_on="date",
            right_on="fundamental_available_date",
            by="ticker",
            strategy="backward",
        ).sort(["ticker", "date"])

        # Capitalizacion bursatil aproximada = precio del dia (EUR) x numero de
        # acciones del ultimo estado financiero disponible. Combina informacion de
        # mercado (close_eur, diaria) con la fundamental (acciones, del cierre mas
        # reciente). Es la unica feature fundamental que varia a diario.
        shares_col = "Weighted Average Diluted Shares Outstanding"
        if shares_col in enriched_model_df.columns and "close_eur" in enriched_model_df.columns:
            enriched_model_df = enriched_model_df.with_columns(
                pl.when(
                    pl.col(shares_col).is_not_null()
                    & (pl.col(shares_col) > 0)
                    & pl.col("close_eur").is_not_null()
                    & (pl.col("close_eur") > 0)
                )
                .then(pl.col("close_eur") * pl.col(shares_col))
                .otherwise(None)
                .alias("_market_cap_eur")
            )

        # Signed log1p de fundamentales
        log_expressions = []
        for src, dst in fe_config.fundamental_log_mapping.items():
            if src in enriched_model_df.columns:
                log_expressions.append(_signed_log1p_expr(src, dst))
        # Capitalizacion en log con signo (misma transformacion que las magnitudes)
        if "_market_cap_eur" in enriched_model_df.columns:
            log_expressions.append(_signed_log1p_expr("_market_cap_eur", "fund_market_cap_slog"))

        if log_expressions:
            enriched_model_df = enriched_model_df.with_columns(log_expressions)

        # Ratios renombrados
        ratio_expressions = []
        for src, dst in fe_config.fundamental_ratio_mapping.items():
            if src in enriched_model_df.columns:
                ratio_expressions.append(pl.col(src).alias(dst))

        if ratio_expressions:
            enriched_model_df = enriched_model_df.with_columns(ratio_expressions)

        # Dias desde disponibilidad + flag binario
        enriched_model_df = enriched_model_df.with_columns(
            (pl.col("date") - pl.col("fundamental_available_date"))
            .dt.total_days()
            .alias("fund_days_since_available"),
            pl.col("fundamental_available_date")
            .is_not_null()
            .cast(pl.Int8)
            .alias("fund_is_available"),
        )

        # Definir features de fundamentales.
        #  - Magnitudes en log (slog), incluida la capitalizacion bursatil.
        #  - Ratios de rentabilidad.
        # Los metadatos de disponibilidad (fund_days_since_available,
        # fund_is_available) se calculan y conservan como columnas auxiliares,
        # pero YA NO entran como features del modelo: con cierres fiscales reales
        # y sin lag, apenas aportan senal y podian introducir un patron
        # estacional espurio.
        magnitude_feature_cols = [
            v for k, v in fe_config.fundamental_log_mapping.items() if k in df.columns
        ]
        if "fund_market_cap_slog" in enriched_model_df.columns:
            magnitude_feature_cols.append("fund_market_cap_slog")
        fundamental_feature_cols = (
            magnitude_feature_cols
            + [v for k, v in fe_config.fundamental_ratio_mapping.items() if k in df.columns]
        )
        enriched_feature_cols = market_feature_cols + fundamental_feature_cols
    else:
        market_model_df = df
        enriched_model_df = df
        fundamental_feature_cols = []
        enriched_feature_cols = market_feature_cols

    # ─── 4. Split temporal ────────────────────────────────────────────────

    if ts_config.train_end_date:
        train_end_date = date.fromisoformat(ts_config.train_end_date)
    else:
        # Calcular automaticamente
        available_dates = (
            market_model_df.select("trade_exit_date").drop_nulls().unique()
            .sort("trade_exit_date").get_column("trade_exit_date").to_list()
        )
        n = len(available_dates)
        train_end_date = available_dates[int(n * ts_config.train_fraction) - 1]

    if ts_config.validation_end_date:
        validation_end_date = date.fromisoformat(ts_config.validation_end_date)
    else:
        available_dates = (
            market_model_df.select("trade_exit_date").drop_nulls().unique()
            .sort("trade_exit_date").get_column("trade_exit_date").to_list()
        )
        n = len(available_dates)
        pos = int(n * (ts_config.train_fraction + ts_config.validation_fraction)) - 1
        validation_end_date = available_dates[pos]

    # Tope superior del test (opcional): los trades que salen despues de esta
    # fecha se descartan (split="excluded"), acotando el periodo de evaluacion.
    test_end_date = (
        date.fromisoformat(ts_config.test_end_date)
        if ts_config.test_end_date
        else None
    )
    if test_end_date is not None:
        logger.info(
            "Test acotado por test_end_date=%s: los trades con salida posterior "
            "se marcan como 'excluded' y no entran en ningun split.",
            test_end_date,
        )

    # Aplicar split y sample eligibility
    market_model_df = _add_split_and_eligibility(
        market_model_df, train_end_date, validation_end_date,
        win_config.sequence_length, test_end_date,
    )
    enriched_model_df = _add_split_and_eligibility(
        enriched_model_df, train_end_date, validation_end_date,
        win_config.sequence_length, test_end_date,
    )

    # ─── 4b. Marca de retornos anomalos por sector (Medida 4 de calidad) ──
    #
    # Marca cada fila cuyo log-retorno diario CRUDO (reconstruido desde close_eur,
    # sin el clipping fijo) cae fuera de las vallas de Tukey de SU sector. Los
    # umbrales se estiman SOLO con train+validation (sin leakage de test). La marca
    # se consume en build_temporal_window_store para descartar ventanas de
    # train/validation que la contengan (las de test se conservan).
    filter_windows = bool(price_quality.get("filter_anomalous_windows", False)) \
        if price_quality else False
    if filter_windows:
        tukey_k = float(price_quality.get("anomaly_tukey_k", 5.0))
        market_model_df = _add_anomaly_flag(market_model_df, tukey_k)
        # Propagar la marca a enriched por (ticker, date) para mantener coherencia.
        enriched_model_df = enriched_model_df.join(
            market_model_df.select(["ticker", "date", "is_anomalous_return"]),
            on=["ticker", "date"], how="left",
        ).with_columns(pl.col("is_anomalous_return").fill_null(False))

    # ─── 5. Sanitizar y escalar ───────────────────────────────────────────

    # Sanitizar Inf → null en features
    market_model_df = sanitize_numeric_columns(market_model_df, market_feature_cols)
    enriched_model_df = sanitize_numeric_columns(enriched_model_df, enriched_feature_cols)

    # Fit preprocesadores sobre datos de entrenamiento
    market_train = market_model_df.filter(
        (pl.col("split") == "train") & pl.col("sample_eligible")
    )
    enriched_train = enriched_model_df.filter(
        (pl.col("split") == "train") & pl.col("sample_eligible")
    )

    # Las columnas one-hot de sector son binarias (0/1): no se escalan.
    market_binary_cols = [c for c in sector_ohe_cols if c in market_feature_cols]
    market_preprocessor = fit_robust_preprocessor(
        train_dataframe=market_train,
        feature_columns=market_feature_cols,
        target_column="target_log_return",
        binary_feature_columns=market_binary_cols,
    )

    enriched_binary_cols = [
        c for c in (["fund_is_available"] + sector_ohe_cols)
        if c in enriched_feature_cols
    ]
    enriched_preprocessor = fit_robust_preprocessor(
        train_dataframe=enriched_train,
        feature_columns=enriched_feature_cols,
        target_column="target_log_return",
        binary_feature_columns=enriched_binary_cols,
    )

    # Transform
    market_prepared = transform_with_preprocessor(market_model_df, market_preprocessor)
    enriched_prepared = transform_with_preprocessor(enriched_model_df, enriched_preprocessor)

    # Feature configuration
    feature_configuration = {
        "market_feature_columns": market_preprocessor["model_feature_columns"],
        "enriched_feature_columns": enriched_preprocessor["model_feature_columns"],
        "market_dropped_columns": market_preprocessor["dropped_feature_columns"],
        "enriched_dropped_columns": enriched_preprocessor["dropped_feature_columns"],
        "segmentation_kind": segmentation_kind,
        "sector_one_hot_enabled": use_sector_ohe,
        "sector_one_hot_columns": sector_ohe_cols,
    }

    split_configuration = {
        "train_end_date": str(train_end_date),
        "validation_end_date": str(validation_end_date),
        "test_end_date": str(test_end_date) if test_end_date is not None else None,
        "execution_lag_sessions": fe_config.execution_lag_sessions,
        "holding_period_sessions": fe_config.holding_period_sessions,
        "sequence_length": win_config.sequence_length,
        "segmentation_kind": segmentation_kind,
    }

    logger.info(
        "Features de mercado: %d variables. Features enriquecidas: %d variables.",
        len(market_preprocessor["model_feature_columns"]),
        len(enriched_preprocessor["model_feature_columns"]),
    )

    return {
        "market_prepared": market_prepared,
        "enriched_prepared": enriched_prepared,
        "market_preprocessor": market_preprocessor,
        "enriched_preprocessor": enriched_preprocessor,
        "feature_configuration": feature_configuration,
        "split_configuration": split_configuration,
    }


def _add_split_and_eligibility(
    dataframe: pl.DataFrame,
    train_end_date: date,
    validation_end_date: date,
    sequence_length: int,
    test_end_date: date | None = None,
) -> pl.DataFrame:
    """Agrega split temporal y marca de elegibilidad para muestreo.

    Una fila es 'sample_eligible' solo si:
    - En train: siempre True.
    - En validation: date > train_end_date AND trade_entry_date > train_end_date.
    - En test: date > validation_end_date AND trade_entry_date > validation_end_date.

    Si se especifica test_end_date, los trades cuya salida (trade_exit_date) sea
    posterior a esa fecha se marcan como 'excluded' (no entran en ningun split ni
    son sample_eligible), acotando el periodo de evaluacion.

    Esto previene data leakage en las fronteras de los splits.
    """
    split_expr = (
        pl.when(pl.col("trade_exit_date") <= pl.lit(train_end_date))
        .then(pl.lit("train"))
        .when(pl.col("trade_exit_date") <= pl.lit(validation_end_date))
        .then(pl.lit("validation"))
    )
    if test_end_date is not None:
        # Tras validacion: es test solo si sale en o antes de test_end_date;
        # si sale despues, se excluye del dataset.
        split_expr = (
            split_expr
            .when(pl.col("trade_exit_date") <= pl.lit(test_end_date))
            .then(pl.lit("test"))
            .otherwise(pl.lit("excluded"))
        )
    else:
        split_expr = split_expr.otherwise(pl.lit("test"))

    dataframe = dataframe.with_columns(split_expr.alias("split"))

    dataframe = dataframe.with_columns(
        pl.when(pl.col("split") == "train")
        .then(pl.lit(True))
        .when(pl.col("split") == "validation")
        .then(
            (pl.col("date") > pl.lit(train_end_date))
            & (pl.col("trade_entry_date") > pl.lit(train_end_date))
        )
        .when(pl.col("split") == "test")
        .then(
            (pl.col("date") > pl.lit(validation_end_date))
            & (pl.col("trade_entry_date") > pl.lit(validation_end_date))
        )
        .otherwise(pl.lit(False))
        .alias("sample_eligible")
    )

    return dataframe.sort(["ticker", "date"])


def _add_anomaly_flag(dataframe: pl.DataFrame, tukey_k: float) -> pl.DataFrame:
    """Marca las filas con log-retorno diario anomalo respecto a su sector.

    La anomalia se define con las vallas de Tukey del log-retorno diario CRUDO
    (reconstruido desde close_eur, sin el clipping fijo de las features) de cada
    sector: fuera de [Q1 - k*IQR, Q3 + k*IQR]. Los cuantiles se estiman SOLO con
    las filas de train + validation, para no introducir informacion del conjunto
    de test en la etapa de preparacion (evita data leakage). La marca se aplica
    despues a TODAS las filas (el filtrado por split se hace al construir ventanas).

    Requiere columnas: ticker, date, sector, close_eur, split. Devuelve el
    dataframe con una columna booleana adicional 'is_anomalous_return'.
    """
    df = dataframe.sort(["ticker", "date"])

    # Log-retorno diario crudo por ticker (sin clipping)
    df = df.with_columns(
        pl.when(
            (pl.col("close_eur") > 0)
            & (pl.col("close_eur").shift(1).over("ticker") > 0)
        )
        .then((pl.col("close_eur") / pl.col("close_eur").shift(1).over("ticker")).log())
        .otherwise(None)
        .alias("_raw_logret")
    )

    # Vallas de Tukey por sector, estimadas SOLO con train+validation
    trainval = df.filter(
        pl.col("split").is_in(["train", "validation"])
        & pl.col("_raw_logret").is_not_null()
        & pl.col("_raw_logret").is_finite()
    )
    sector_bounds = (
        trainval.group_by("sector")
        .agg(
            pl.col("_raw_logret").quantile(0.25).alias("_q1"),
            pl.col("_raw_logret").quantile(0.75).alias("_q3"),
        )
        .with_columns((pl.col("_q3") - pl.col("_q1")).alias("_iqr"))
        .with_columns(
            (pl.col("_q1") - tukey_k * pl.col("_iqr")).alias("_lo"),
            (pl.col("_q3") + tukey_k * pl.col("_iqr")).alias("_hi"),
        )
        .select(["sector", "_lo", "_hi"])
    )

    df = df.join(sector_bounds, on="sector", how="left")

    # Una fila es anomala si su retorno cae fuera de las vallas de su sector.
    # Si no hay retorno (primera sesion) o el sector no tiene vallas, no es anomala.
    df = df.with_columns(
        (
            pl.col("_raw_logret").is_not_null()
            & pl.col("_lo").is_not_null()
            & (
                (pl.col("_raw_logret") < pl.col("_lo"))
                | (pl.col("_raw_logret") > pl.col("_hi"))
            )
        ).alias("is_anomalous_return")
    )

    return df.drop(["_raw_logret", "_lo", "_hi"])
