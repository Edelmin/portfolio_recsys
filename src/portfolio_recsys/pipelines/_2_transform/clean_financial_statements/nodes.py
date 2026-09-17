"""Nodos del pipeline clean_financial_statements.

Transforma los Parquets crudos de estados financieros a formato limpio:
- Filtra campos seleccionados (por hoja) desde parameters.yaml.
- Transpone de formato ancho (métricas como filas) a formato largo (fechas como filas).
- Une las 3 hojas por fecha para cada empresa.
- Produce un Parquet por empresa.
"""

import logging
from pathlib import Path

import polars as pl

from portfolio_recsys.pipelines._1_ingest.parse_financial_statements.nodes import (
    parse_filename,
)
from portfolio_recsys.pipelines.sector_mapping import SECTORS_WITH_FS

logger = logging.getLogger(__name__)

SHEETS = ["Income_Statement", "Cash_Flow", "Ratios"]

# Mapeo del nombre de la primera columna de cada hoja al nombre limpio
SHEET_METRIC_COL = {
    "Income_Statement": "Income Statement",
    "Cash_Flow": "Cash Flow",
    "Ratios": "Ratios",
}


def _process_sheet(
    df: pl.DataFrame,
    sheet: str,
    selected_fields: list[str],
    meta: dict,
) -> pl.DataFrame | None:
    """Procesa una hoja: filtra campos, transpone, añade metadatos.

    Returns:
        DataFrame con columnas: date + campos seleccionados, o None si vacío.
    """
    if df.is_empty() or df.shape[1] < 2:
        return None

    metric_col = df.columns[0]
    clean_name = SHEET_METRIC_COL.get(sheet, sheet)

    # Renombrar primera columna a nombre limpio
    df = df.rename({metric_col: clean_name})

    # Drop columnas de metadatos y LTM si existen
    cols_to_drop = [c for c in ["ticker", "currency", "sector", "period_start", "period_end", "LTM"] if c in df.columns]
    if cols_to_drop:
        df = df.drop(cols_to_drop)

    # Filtrar campos seleccionados (preservando orden)
    if not selected_fields:
        return None

    temp_df = pl.DataFrame({clean_name: selected_fields}).with_row_index("_orden")

    filtered = (
        temp_df
        .join(df, on=clean_name, how="left", coalesce=True)
        .sort("_orden")
        .drop("_orden")
    )

    # Transponer: filas=fechas, columnas=métricas
    transposed = (
        filtered
        .transpose(column_names=clean_name, include_header=True)
        .rename({"column": "date"})
    )

    # Castear columnas de métricas a Float64
    metric_cols = [c for c in transposed.columns if c != "date"]
    for col in metric_cols:
        transposed = transposed.with_columns(
            pl.col(col).cast(pl.Float64, strict=False)
        )

    return transposed


def _align_to_calendar_year_end(df: pl.DataFrame, numeric_cols: list[str]) -> pl.DataFrame:
    """[OBSOLETA — ya no se usa] Alinea las fechas fiscales al 31/12 del año natural.

    Esta funcion interpolaba los estados financieros al 31/12 usando el cierre
    del ano siguiente, lo que introducia informacion futura (data leakage) en las
    features fundamentales. Se ha dejado de usar: ahora se conservan los cierres
    fiscales reales de cada empresa sin interpolar (ver clean_financial_statements).
    Se mantiene el codigo por referencia historica; no debe invocarse.

    Para empresas con cierre fiscal que no cae en 31/12 (ej: 31/03, 30/06),
    interpola linealmente entre cierres consecutivos para estimar el valor
    al 31/12 de cada año cubierto.

    Logica:
    - Ordenar por fecha.
    - Para cada año cubierto, generar un punto al 31/12.
    - Si ya existe un dato exacto al 31/12, se usa directamente.
    - Si no, se interpola entre el cierre anterior y el siguiente.

    Args:
        df: DataFrame con columnas date (Datetime UTC) + numeric_cols + metadatos.
        numeric_cols: Lista de columnas numéricas a interpolar.

    Returns:
        DataFrame con fechas alineadas al 31/12 de cada año.
    """
    from datetime import datetime, timezone

    df = df.sort("date")

    # Extraer años cubiertos
    years = df.with_columns(
        pl.col("date").dt.year().alias("_year")
    )["_year"].unique().sort().to_list()

    if not years:
        return df

    # Generar fechas objetivo: 31/12 de cada año
    target_dates = [
        datetime(y, 12, 31, tzinfo=timezone.utc) for y in years
    ]

    # Verificar si ya todas las fechas caen en 31/12
    existing_month_days = df.with_columns([
        pl.col("date").dt.month().alias("_month"),
        pl.col("date").dt.day().alias("_day"),
    ])
    all_dec_31 = existing_month_days.filter(
        (pl.col("_month") == 12) & (pl.col("_day") == 31)
    ).shape[0] == df.shape[0]

    if all_dec_31:
        # Ya está alineado, no hacer nada
        return df

    # Obtener metadatos (constantes para la empresa)
    meta_cols = ["ticker", "currency", "sector", "period_start", "period_end"]
    meta_values = {col: df[col][0] for col in meta_cols if col in df.columns}

    # Construir el DataFrame interpolado
    # Usamos los puntos originales + los targets, luego interpolamos
    original_dates = df["date"].to_list()
    original_values = {col: df[col].to_list() for col in numeric_cols}

    result_rows = []
    for target_dt in target_dates:
        # Buscar el intervalo [anterior, siguiente] que contiene target_dt
        before_idx = None
        after_idx = None
        for i, d in enumerate(original_dates):
            if d == target_dt:
                # Dato exacto
                before_idx = i
                after_idx = i
                break
            elif d < target_dt:
                before_idx = i
            elif d > target_dt:
                after_idx = i
                break

        if before_idx is None and after_idx is None:
            continue

        row = {"date": target_dt}

        if before_idx == after_idx:
            # Dato exacto en esa fecha
            for col in numeric_cols:
                row[col] = original_values[col][before_idx]
        elif before_idx is not None and after_idx is not None:
            # Interpolar linealmente
            d_before = original_dates[before_idx]
            d_after = original_dates[after_idx]
            total_days = (d_after - d_before).total_seconds() / 86400
            elapsed_days = (target_dt - d_before).total_seconds() / 86400

            if total_days > 0:
                frac = elapsed_days / total_days
            else:
                frac = 0.0

            for col in numeric_cols:
                v_before = original_values[col][before_idx]
                v_after = original_values[col][after_idx]
                if v_before is not None and v_after is not None:
                    row[col] = v_before + (v_after - v_before) * frac
                elif v_before is not None:
                    row[col] = v_before
                elif v_after is not None:
                    row[col] = v_after
                else:
                    row[col] = None
        elif before_idx is not None:
            # Solo hay dato anterior (target > ultima fecha) → usar ultimo valor
            for col in numeric_cols:
                row[col] = original_values[col][before_idx]
        else:
            # Solo hay dato posterior (target < primera fecha) → no generar
            continue

        result_rows.append(row)

    if not result_rows:
        return pl.DataFrame()

    # Construir DataFrame resultado
    result = pl.DataFrame(result_rows)

    # Asegurar que date es Datetime UTC
    if result["date"].dtype != pl.Datetime("us", "UTC"):
        result = result.with_columns(
            pl.col("date").cast(pl.Datetime("us", "UTC"))
        )

    # Añadir metadatos
    for col, val in meta_values.items():
        result = result.with_columns(pl.lit(val).alias(col))

    return result


def clean_financial_statements(
    fs_base_path: str,
    output_base_path: str,
    cap: str,
    selected_fields: dict,
) -> dict[str, int]:
    """Limpia y transpone los estados financieros por empresa.

    Para cada empresa (ticker) de cada sector:
    1. Carga las 3 hojas.
    2. Filtra campos seleccionados y transpone.
    3. Une las 3 hojas por fecha.
    4. Escribe un Parquet por empresa.

    Args:
        fs_base_path: Ruta de los Parquets crudos.
        output_base_path: Ruta de salida para los Parquets limpios.
        cap: Capitalización ("LargeCaps" o "SmallCaps").
        selected_fields: Dict {sheet_name: [field1, field2, ...]}.

    Returns:
        Dict con estadísticas: {sector: n_empresas_procesadas}.
    """
    base = Path(fs_base_path) / cap
    output_base = Path(output_base_path) / cap
    stats: dict[str, int] = {}

    for sector in SECTORS_WITH_FS:
        sector_path = base / sector
        if not sector_path.exists():
            logger.warning("Sector %s: carpeta no encontrada en %s", sector, sector_path)
            stats[sector] = 0
            continue

        output_sector_path = output_base / sector
        output_sector_path.mkdir(parents=True, exist_ok=True)

        # Agrupar ficheros por ticker
        ticker_files: dict[str, dict[str, Path]] = {}
        for sheet in SHEETS:
            for f in sorted(sector_path.glob(f"*-{sheet}.parquet")):
                meta = parse_filename(f.stem.replace(f"-{sheet}", "") + ".xlsx")
                if not meta:
                    continue
                ticker = meta["ticker"]
                ticker_files.setdefault(ticker, {"meta": meta})[sheet] = f

        empresas_ok = 0
        for ticker, files_info in sorted(ticker_files.items()):
            meta = files_info.get("meta", {})

            # Procesar cada hoja
            sheet_dfs = []
            for sheet in SHEETS:
                sheet_path = files_info.get(sheet)
                if not sheet_path:
                    continue

                fields_for_sheet = selected_fields.get(sheet, [])
                if not fields_for_sheet:
                    continue

                try:
                    raw_df = pl.read_parquet(sheet_path)
                except Exception as e:
                    logger.warning("Error leyendo %s: %s", sheet_path.name, e)
                    continue

                processed = _process_sheet(raw_df, sheet, fields_for_sheet, meta)
                if processed is not None and not processed.is_empty():
                    sheet_dfs.append(processed)

            if not sheet_dfs:
                continue

            # Unir hojas por fecha
            combined = sheet_dfs[0]
            for extra_df in sheet_dfs[1:]:
                combined = combined.join(extra_df, on="date", how="outer", coalesce=True)

            # Añadir metadatos
            combined = combined.with_columns([
                pl.lit(meta.get("ticker", "")).alias("ticker"),
                pl.lit(meta.get("currency", "")).alias("currency"),
                pl.lit(sector).alias("sector"),
                pl.lit(meta.get("period_start", "")).alias("period_start"),
                pl.lit(meta.get("period_end", "")).alias("period_end"),
            ])

            # Castear tipos finales
            # date:         "DD/MM/YY"  → pl.Date (barras, año 2 dígitos)
            # period_start: "DD.MM.YY"  → pl.Date (puntos, año 2 dígitos)
            # period_end:   "DD.MM.YY"  → pl.Date (puntos, año 2 dígitos)
            # Todas las fechas se convierten a Datetime UTC para consistencia
            combined = combined.with_columns([
                pl.col("date")
                .str.strptime(pl.Date, "%d/%m/%y", strict=False)
                .cast(pl.Datetime("us", "UTC"))
                .alias("date"),
                pl.col("period_start")
                .str.strptime(pl.Date, "%d.%m.%y", strict=False)
                .cast(pl.Datetime("us", "UTC"))
                .alias("period_start"),
                pl.col("period_end")
                .str.strptime(pl.Date, "%d.%m.%y", strict=False)
                .cast(pl.Datetime("us", "UTC"))
                .alias("period_end"),
            ])

            # Métricas numéricas: asegurar Float64
            numeric_cols = [
                c for c in combined.columns
                if c not in {"date", "ticker", "currency", "sector", "period_start", "period_end"}
            ]
            if numeric_cols:
                combined = combined.with_columns([
                    pl.col(c).cast(pl.Float64, strict=False) for c in numeric_cols
                ])

            # --- Descartar fechas anteriores a 2009 ---
            cutoff = pl.lit("2009-01-01").str.strptime(pl.Date, "%Y-%m-%d").cast(pl.Datetime("us", "UTC"))
            combined = combined.filter(pl.col("date") >= cutoff)

            if combined.is_empty():
                continue

            # --- Cierres fiscales reales (sin alinear al 31/12) ---
            # Se conservan las fechas de cierre fiscal tal como las reporta cada
            # empresa (p. ej. 31/03, 30/06, 31/12), sin interpolar al 31/12 del
            # año natural. Motivo: la interpolacion al fin de año estimaba el
            # valor de una fecha usando el cierre del año SIGUIENTE, lo que
            # introduce informacion futura (data leakage sutil) en las features.
            # Se asume que cada estado financiero esta disponible en su propia
            # fecha de cierre. Se ordena por fecha y se eliminan duplicados por
            # fecha (conservando la ultima observacion de cada cierre).
            combined = (
                combined.sort("date")
                .unique(subset=["date"], keep="last", maintain_order=True)
            )

            if combined.is_empty():
                continue

            # Escribir Parquet
            output_path = output_sector_path / f"{ticker}.parquet"
            combined.write_parquet(output_path)
            empresas_ok += 1

        stats[sector] = empresas_ok
        logger.info("Sector %s: %d empresas procesadas", sector, empresas_ok)

    return stats
