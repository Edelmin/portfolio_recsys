"""Nodos del pipeline build_company_dataset.

Consolida precios EUR + estados financieros (ya en EUR) en un dataset
unificado por sector. Cada sector produce un Parquet con:
- Precios diarios en EUR (date, ticker, close, close_eur, rate_to_eur)
- Metadatos de empresa (sector, currency_original)
- Metricas financieras anuales asignadas por join_asof (cada dia hereda
  los fundamentales del cierre fiscal mas reciente <= su fecha)

Las columnas monetarias de los FS se renombran con sufijo ``_eur`` para
dejar constancia de que han sido convertidas (o ratificadas) a euros.
Los ratios (%) y columnas de acciones (shares) se mantienen sin sufijo.
"""

import logging
from collections.abc import Callable

import polars as pl

from portfolio_recsys.pipelines.sector_mapping import SECTORS_WITH_FS

logger = logging.getLogger(__name__)

# Columnas de metadatos / estructura — nunca se renombran
_META_COLS = {"date", "ticker", "currency", "sector", "period_start", "period_end", "rate_to_eur"}

# Palabras clave que indican columnas adimensionales (shares, ratios %).
# Estas no representan importes monetarios y no llevan sufijo _eur.
_NON_MONETARY_KEYWORDS = {"shares", "share"}


def _is_monetary_column(col_name: str) -> bool:
    """Determina si una columna financiera es monetaria (y requiere sufijo _eur).

    Criterios para NO ser monetaria:
    - Está en META_COLS.
    - Termina en '%' (ratio/porcentaje).
    - Contiene 'Shares' o 'Share' (unidades de acciones).
    - Empieza por '%' (porcentaje como prefijo).
    """
    if col_name in _META_COLS:
        return False
    if col_name.endswith("%"):
        return False
    if col_name.startswith("%"):
        return False
    col_lower = col_name.lower()
    if any(kw in col_lower for kw in _NON_MONETARY_KEYWORDS):
        return False
    return True


def _load_fs_eur(partitions: dict[str, Callable[[], pl.DataFrame]]) -> pl.DataFrame:
    """Apila todas las particiones (empresas) de FS en EUR de un sector.

    Recibe un PartitionedDataset del catalogo: un diccionario
    ``{nombre_fichero: callable -> pl.DataFrame}``, una entrada por empresa.
    Cada DataFrame tiene formato: date (Datetime UTC), metricas (Float64),
    ticker, currency (=EUR), sector, period_start, period_end, rate_to_eur.

    Args:
        partitions: Diccionario de particiones del sector (o vacio si no hay).

    Returns:
        DataFrame apilado con todas las empresas del sector, o vacio.
    """
    if not partitions:
        return pl.DataFrame()

    dfs = []
    for partition_id, load_fn in sorted(partitions.items()):
        try:
            df = load_fn()
            if not df.is_empty():
                dfs.append(df)
        except Exception as e:  # noqa: BLE001
            logger.warning("Error leyendo particion %s: %s", partition_id, e)

    if not dfs:
        return pl.DataFrame()

    return pl.concat(dfs, how="diagonal_relaxed")


def _compute_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    """Calcula ratios y métricas derivadas a partir de los fundamentales y precios.

    Features calculadas:
    - gross_profit_ratio: Gross Profit / Total Revenues
    - ebitda_ratio: EBITDA / Total Revenues
    - operating_income_ratio: Operating Income / Total Revenues
    - net_income_ratio: Net Income to Company / Total Revenues
    - capitalization_eur: Weighted Average Diluted Shares Outstanding * close_eur
    - per: capitalization_eur / Net Income to Company

    Todas las divisiones usan proteccion frente a division por cero (null si
    el denominador es 0 o null).
    """
    cols = df.columns

    # Solo calcular si las columnas necesarias existen
    has_revenues = "Total Revenues_eur" in cols
    has_gross = "Gross Profit_eur" in cols
    has_ebitda = "EBITDA_eur" in cols
    has_op_income = "Operating Income_eur" in cols
    has_net_income = "Net Income to Company_eur" in cols
    has_shares = "Weighted Average Diluted Shares Outstanding" in cols
    has_close_eur = "close_eur" in cols

    expressions: list[pl.Expr] = []

    if has_revenues and has_gross:
        expressions.append(
            (pl.col("Gross Profit_eur") / pl.col("Total Revenues_eur"))
            .alias("gross_profit_ratio")
        )

    if has_revenues and has_ebitda:
        expressions.append(
            (pl.col("EBITDA_eur") / pl.col("Total Revenues_eur"))
            .alias("ebitda_ratio")
        )

    if has_revenues and has_op_income:
        expressions.append(
            (pl.col("Operating Income_eur") / pl.col("Total Revenues_eur"))
            .alias("operating_income_ratio")
        )

    if has_revenues and has_net_income:
        expressions.append(
            (pl.col("Net Income to Company_eur") / pl.col("Total Revenues_eur"))
            .alias("net_income_ratio")
        )

    if has_shares and has_close_eur:
        # TODO: Incorporar capitalization_eur como feature del modelo en iteración futura
        expressions.append(
            (pl.col("Weighted Average Diluted Shares Outstanding") * pl.col("close_eur"))
            .alias("capitalization_eur")
        )

    if has_shares and has_close_eur and has_net_income:
        # TODO: Incorporar PER como feature del modelo en iteración futura
        expressions.append(
            (
                (pl.col("Weighted Average Diluted Shares Outstanding") * pl.col("close_eur"))
                / pl.col("Net Income to Company_eur")
            ).alias("per")
        )

    if expressions:
        df = df.with_columns(expressions)

    return df


# Horizontes para log-returns, expresados en días de trading aproximados.
# Cada entrada: (sufijo para el nombre de columna, nº de sesiones de shift)
# TODO: Los horizontes de 2y y 3y no se utilizan actualmente como features.
#       Revisar la disponibilidad efectiva de datos a estos horizontes para
#       valorar su incorporación futura.
_LOG_RETURN_HORIZONS: list[tuple[str, int]] = [
    ("1d", 1),
    ("1w", 5),
    ("2w", 10),
    ("1m", 21),
    ("2m", 42),
    ("3m", 63),
    ("6m", 126),
    ("1y", 252),
    ("2y", 504),
    ("3y", 756),
]


def _compute_log_returns(df: pl.DataFrame) -> pl.DataFrame:
    """Calcula log-returns sobre close_eur para múltiples horizontes.

    Para cada horizonte h (en días de trading):
        log_return_{h} = ln(close_eur(t)) - ln(close_eur(t - h))

    El cálculo se hace por ticker (over("ticker")), asumiendo que el
    DataFrame viene ordenado por (ticker, date).

    Genera columnas: log_return_1d, log_return_1w, log_return_2w,
    log_return_1m, log_return_2m, log_return_3m, log_return_6m,
    log_return_1y, log_return_2y, log_return_3y.

    Las primeras `h` filas de cada ticker tendrán null (sin dato anterior).
    """
    if "close_eur" not in df.columns:
        return df

    # Pre-calcular ln(close_eur) una sola vez
    df = df.with_columns(pl.col("close_eur").log().alias("_ln_close_eur"))

    # Calcular cada horizonte como diferencia de logs
    expressions = [
        (
            pl.col("_ln_close_eur")
            - pl.col("_ln_close_eur").shift(sessions).over("ticker")
        ).alias(f"log_return_{suffix}")
        for suffix, sessions in _LOG_RETURN_HORIZONS
    ]

    df = df.with_columns(expressions)

    # Eliminar columna auxiliar
    df = df.drop("_ln_close_eur")

    return df


def build_company_dataset(
    stock_prices_eur: pl.DataFrame,
    ticker_mapping: dict,
    cap: str,
    **sector_inputs: object,
) -> dict[str, pl.DataFrame]:
    """Construye el dataset consolidado de empresas por sector.

    Para cada sector:
    1. Identifica los tickers de ese sector desde ENRICHED_*.
    2. Usa ticker_mapping para traducir entre ticker TIKR (en FS) y
       ticker yfinance (en stock_prices_eur).
    3. Filtra sus precios EUR de stock_prices_eur.
    4. Carga los FS en EUR de 02_3_2.
    5. Hace join_asof por ticker + fecha (cada dia hereda los fundamentales
       del cierre fiscal mas reciente <= su fecha).
    6. Produce un dataset por sector.

    Args:
        stock_prices_eur: Precios diarios en EUR (todas las empresas).
        ticker_mapping: Dict {sector_es: {ticker_tikr: ticker_yfinance}}.
        cap: Capitalizacion ("LargeCaps" o "SmallCaps").
        **sector_inputs: Inputs por sector inyectados por el pipeline:
            - ENRICHED_{cap}_{canonical} con clave "{cap}_{canonical}"
              (DataFrames, para mapear ticker -> sector).
            - FS_EUR_{cap}_{canonical} con clave "fs_eur__{canonical}"
              (PartitionedDataset: dict {fichero: callable -> pl.DataFrame}).

    Returns:
        Diccionario {sector_canonical: DataFrame} con un dataset por sector.
    """
    logger.info("Consolidando dataset de empresas para cap=%s", cap)

    if stock_prices_eur.is_empty():
        logger.warning("stock_prices_eur esta vacio, nada que consolidar")
        return {}

    # Separar los inputs por tipo segun su prefijo de clave.
    fs_eur_partitions: dict[str, dict] = {
        key[len("fs_eur__"):]: value
        for key, value in sector_inputs.items()
        if key.startswith("fs_eur__")
    }
    enriched_sectors: dict[str, pl.DataFrame] = {
        key: value
        for key, value in sector_inputs.items()
        if not key.startswith("fs_eur__")
    }

    # Construir mapeo bidireccional: tikr_ticker -> yf_ticker
    tikr_to_yf: dict[str, str] = {}
    for _sector_es, mapping in ticker_mapping.items():
        for tikr_t, yf_t in mapping.items():
            tikr_to_yf[tikr_t] = yf_t

    # Construir mapeo ticker_yf -> sector (canonical)
    ticker_to_sector: dict[str, str] = {}
    for key, df in enriched_sectors.items():
        if df.is_empty():
            continue
        parts = key.split("_", 1)
        sector = parts[1] if len(parts) > 1 else key

        for ticker in df.filter(pl.col("currency").is_not_null())["Simbolo"].to_list():
            yf_ticker = tikr_to_yf.get(ticker, ticker)
            if yf_ticker not in ticker_to_sector:
                ticker_to_sector[yf_ticker] = sector

    # Agrupar tickers yfinance por sector
    sector_tickers: dict[str, list[str]] = {}
    for yf_ticker, sector in ticker_to_sector.items():
        sector_tickers.setdefault(sector, []).append(yf_ticker)

    results = {}

    for sector in sorted(SECTORS_WITH_FS):
        yf_tickers = sector_tickers.get(sector, [])
        if not yf_tickers:
            logger.warning("Sector %s: sin tickers mapeados", sector)
            continue

        # Filtrar precios de este sector
        sector_prices = stock_prices_eur.filter(pl.col("ticker").is_in(yf_tickers))

        if sector_prices.is_empty():
            logger.warning("Sector %s: sin precios EUR disponibles", sector)
            continue

        # Cargar FS en EUR del sector (particiones inyectadas desde el catalogo)
        fs_eur = _load_fs_eur(fs_eur_partitions.get(sector) or {})

        if fs_eur.is_empty():
            dataset = sector_prices.with_columns(pl.lit(sector).alias("sector"))
            results[sector] = dataset
            logger.info(
                "Sector %s: %d registros (solo precios, sin fundamentales)",
                sector, dataset.shape[0],
            )
            continue

        # Traducir tickers TIKR en FS a tickers yfinance para el join
        fs_eur = fs_eur.with_columns(
            pl.col("ticker").replace_strict(tikr_to_yf, default=pl.col("ticker")).alias("ticker")
        )

        # Seleccionar solo columnas necesarias para el join:
        # date (fiscal), ticker, y todas las metricas financieras
        meta_cols = {"ticker", "currency", "sector", "period_start", "period_end", "rate_to_eur"}
        financial_cols = [c for c in fs_eur.columns if c not in meta_cols and c != "date"]
        fs_for_join = fs_eur.select(["date", "ticker"] + financial_cols)

        # Renombrar columnas monetarias con sufijo _eur para denotar conversion
        rename_map = {
            c: f"{c}_eur" for c in financial_cols if _is_monetary_column(c)
        }
        if rename_map:
            fs_for_join = fs_for_join.rename(rename_map)

        # Renombrar date a fiscal_date para claridad en el join
        fs_for_join = fs_for_join.rename({"date": "fiscal_date"}).sort(["ticker", "fiscal_date"])

        # Join asof: para cada (ticker, date) en precios, obtener los fundamentales
        # del fiscal_date mas reciente <= date
        sector_prices_sorted = sector_prices.sort("date")

        # Asegurar mismo tipo para la columna de join
        # stock_prices_eur.date es Datetime UTC, fs fiscal_date tambien
        joined = sector_prices_sorted.join_asof(
            fs_for_join,
            left_on="date",
            right_on="fiscal_date",
            by="ticker",
            strategy="backward",
        ).with_columns(pl.lit(sector).alias("sector"))

        # Descartar filas sin fundamental asignado (precio anterior al primer cierre fiscal)
        joined = joined.filter(pl.col("fiscal_date").is_not_null())

        if joined.is_empty():
            logger.warning("Sector %s: sin filas tras filtrar por fiscal_date", sector)
            continue

        # --- Calcular ratios y metricas derivadas ---
        joined = _compute_derived_features(joined)

        # --- Calcular log-returns para múltiples horizontes ---
        joined = joined.sort(["ticker", "date"])
        joined = _compute_log_returns(joined)

        results[sector] = joined
        logger.info(
            "Sector %s: %d registros, %d tickers, %d columnas",
            sector, joined.shape[0], joined["ticker"].n_unique(), joined.shape[1],
        )

    return results
