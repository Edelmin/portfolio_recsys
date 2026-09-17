"""Nodos del pipeline validate_price_currencies.

Convierte los valores monetarios de los estados financieros limpios a EUR
usando los tipos de cambio historicos (join_asof, strategy="backward").

Para cada fecha fiscal de cada empresa, se busca el tipo de cambio XXX/EUR
cuya fecha sea <= la fecha fiscal (ultimo rate conocido a esa fecha).
Los campos que son ratios (%) no se convierten.
"""

import logging
from pathlib import Path

import polars as pl

from portfolio_recsys.pipelines.sector_mapping import SECTORS_WITH_FS

logger = logging.getLogger(__name__)

# Columnas que NO son metricas monetarias (no se multiplican por el rate)
META_COLS = {"date", "ticker", "currency", "sector", "period_start", "period_end"}


def convert_financial_statements_to_eur(
    clean_fs_base_path: str,
    output_base_path: str,
    cap: str,
    exchange_rates: pl.DataFrame,
) -> dict[str, object]:
    """Convierte los valores monetarios de los FS limpios a EUR.

    Para cada empresa cuya moneda != EUR:
    1. Carga el Parquet limpio.
    2. Hace join_asof con exchange_rates para obtener rate_to_eur por fecha fiscal.
    3. Multiplica las columnas monetarias por rate_to_eur.
    4. Escribe el resultado como Parquet en la carpeta de salida.

    Las empresas en EUR se copian directamente (rate_to_eur = 1.0).
    Las empresas sin par de conversion se registran en cuarentena.

    Args:
        clean_fs_base_path: Ruta de los FS limpios (output de clean_financial_statements).
        output_base_path: Ruta de salida para los FS en EUR.
        cap: Capitalizacion ("LargeCaps" o "SmallCaps").
        exchange_rates: DataFrame con columnas (date, currency, rate_to_eur).

    Returns:
        Dict con estadisticas y cuarentena:
        - "stats": Dict[sector, {"converted": int, "quarantine": int}]
        - "quarantine": Lista de dicts {"ticker", "currency", "sector", "reason"}
    """
    clean_base = Path(clean_fs_base_path) / cap
    output_base = Path(output_base_path) / cap

    # Divisas disponibles en exchange_rates
    available_currencies: set[str] = set()
    if not exchange_rates.is_empty():
        # Compatibilidad: si el Parquet todavía tiene "rate" (nombre antiguo), renombrar
        if "rate" in exchange_rates.columns and "rate_to_eur" not in exchange_rates.columns:
            exchange_rates = exchange_rates.rename({"rate": "rate_to_eur"})
        available_currencies = set(exchange_rates["currency"].unique().to_list())

    stats: dict[str, dict[str, int]] = {}
    quarantine_records: list[dict] = []

    for sector in SECTORS_WITH_FS:
        sector_path = clean_base / sector
        if not sector_path.exists():
            stats[sector] = {"converted": 0, "quarantine": 0}
            continue

        output_sector_path = output_base / sector
        output_sector_path.mkdir(parents=True, exist_ok=True)

        converted_count = 0
        quarantine_count = 0

        for parquet_file in sorted(sector_path.glob("*.parquet")):
            df = pl.read_parquet(parquet_file)
            if df.is_empty():
                continue

            ticker = df["ticker"][0]
            currency = df["currency"][0]

            if currency == "EUR":
                # No necesita conversion, añadir rate_to_eur = 1.0
                df_eur = df.with_columns(pl.lit(1.0).alias("rate_to_eur"))
                df_eur.write_parquet(output_sector_path / parquet_file.name)
                converted_count += 1

            elif currency in available_currencies:
                # Conversion via join_asof
                rates_for_currency = (
                    exchange_rates
                    .filter(pl.col("currency") == currency)
                    .select(["date", "rate_to_eur"])
                    .sort("date")
                )

                df_sorted = df.sort("date")
                joined = df_sorted.join_asof(
                    rates_for_currency,
                    on="date",
                    strategy="backward",
                )

                # Verificar que hay rates
                n_missing = joined.filter(pl.col("rate_to_eur").is_null()).shape[0]
                if n_missing == joined.shape[0]:
                    # Todas las fechas son anteriores al primer rate disponible
                    quarantine_records.append({
                        "ticker": ticker,
                        "currency": currency,
                        "sector": sector,
                        "reason": f"Todas las fechas fiscales anteriores al primer rate ({rates_for_currency['date'].min()})",
                    })
                    quarantine_count += 1
                    continue

                # Identificar columnas monetarias (las que no son meta ni ratios %)
                monetary_cols = [
                    c for c in joined.columns
                    if c not in META_COLS
                    and c != "rate_to_eur"
                    and not c.endswith("%")
                ]

                # Multiplicar columnas monetarias por rate_to_eur
                if monetary_cols:
                    joined = joined.with_columns([
                        (pl.col(c) * pl.col("rate_to_eur")).alias(c)
                        for c in monetary_cols
                    ])

                # Actualizar moneda a EUR
                joined = joined.with_columns(
                    pl.lit("EUR").alias("currency")
                )

                joined.write_parquet(output_sector_path / parquet_file.name)
                converted_count += 1

                if n_missing > 0:
                    logger.warning(
                        "Ticker %s (%s): %d/%d fechas sin rate (anteriores al historico)",
                        ticker, currency, n_missing, joined.shape[0],
                    )

            else:
                # Sin par de conversion disponible → cuarentena
                quarantine_records.append({
                    "ticker": ticker,
                    "currency": currency,
                    "sector": sector,
                    "reason": "Sin par de conversion disponible",
                })
                quarantine_count += 1

        stats[sector] = {"converted": converted_count, "quarantine": quarantine_count}
        logger.info(
            "Sector %s: %d convertidos, %d en cuarentena",
            sector, converted_count, quarantine_count,
        )

    # Resumen global
    total_converted = sum(s["converted"] for s in stats.values())
    total_quarantine = sum(s["quarantine"] for s in stats.values())
    logger.info(
        "Conversion FS a EUR completada: %d empresas convertidas, %d en cuarentena",
        total_converted, total_quarantine,
    )

    return {
        "stats": stats,
        "quarantine": quarantine_records,
    }
