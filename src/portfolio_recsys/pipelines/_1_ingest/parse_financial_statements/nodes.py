"""Nodos del pipeline parse_financial_statements.

Lee los ficheros Excel de estados financieros historicos (multiples hojas por fichero)
y los convierte a Parquet sin modificar los datos. Solo asegura la conversion de formato.

Usa pl.read_excel() con engine="calamine" para evitar el bug de openpyxl/pandas con NaN.
"""

import logging
import re
from pathlib import Path

import polars as pl

from portfolio_recsys.pipelines.sector_mapping import SECTORS, SECTORS_WITH_FS

logger = logging.getLogger(__name__)

# Hojas que se extraen de cada Excel
SHEETS = ["Income Statement", "Cash Flow", "Ratios"]


def parse_filename(filename: str) -> dict | None:
    """Extrae currency, ticker y periodo del nombre del fichero.

    Patron: '{currency} - {ticker} - Financials ({start} - {end})'
    """
    name = Path(filename).stem
    pattern = r"^(.+?)\s*-\s*(.+?)\s*-\s*Financials\s*\((.+?)\s*-\s*(.+?)\)$"
    match = re.match(pattern, name)
    if match:
        return {
            "currency": match.group(1).strip(),
            "ticker": match.group(2).strip(),
            "period_start": match.group(3).strip(),
            "period_end": match.group(4).strip(),
        }
    return None


def convert_sector_excel_to_parquet(
    raw_base_path: str,
    output_base_path: str,
    cap: str = "LargeCaps",
) -> dict[str, int]:
    """Convierte todos los Excel de estados financieros a Parquet.

    Para cada sector con carpeta FS, lee cada .xlsx con polars+calamine
    y escribe un .parquet por hoja en la carpeta de salida, organizada
    por capitalizacion y sector.

    Args:
        raw_base_path: Ruta base de los ficheros raw.
            Ej: 'data/01_raw/01.2 - FinancialStatementHistorical'
        output_base_path: Ruta base de salida para los Parquet.
            Ej: 'data/02_intermediate/02_3_0. RawFinancialStatements_Parquet'
        cap: Capitalizacion de los datos ('LargeCaps' o 'SmallCaps').

    Returns:
        Diccionario con estadisticas: {sector: n_ficheros_convertidos}
    """
    raw_base = Path(raw_base_path)
    output_base = Path(output_base_path)
    stats: dict[str, int] = {}

    for sector_canonical in SECTORS_WITH_FS:
        sector_info = SECTORS[sector_canonical]
        sector_folder = sector_info.folder_fs
        sector_path = raw_base / sector_folder

        if not sector_path.exists():
            logger.warning(
                "Carpeta no encontrada para sector %s: %s",
                sector_canonical,
                sector_path,
            )
            stats[sector_canonical] = 0
            continue

        output_sector_path = output_base / cap / sector_canonical
        output_sector_path.mkdir(parents=True, exist_ok=True)

        ficheros_ok = 0
        for xlsx_file in sorted(sector_path.glob("*.xlsx")):
            meta = parse_filename(xlsx_file.name)
            if not meta:
                logger.warning("No se pudo parsear filename: %s", xlsx_file.name)
                continue

            # Nombre base para los parquet de salida
            base_name = xlsx_file.stem

            hojas_escritas = 0
            for sheet_name in SHEETS:
                try:
                    df = pl.read_excel(
                        xlsx_file,
                        sheet_name=sheet_name,
                        engine="calamine",
                    )
                except Exception as e:
                    logger.warning(
                        "Error leyendo hoja '%s' de %s: %s",
                        sheet_name,
                        xlsx_file.name,
                        e,
                    )
                    continue

                # Añadir metadatos como columnas
                df = df.with_columns([
                    pl.lit(meta["ticker"]).alias("ticker"),
                    pl.lit(meta["currency"]).alias("currency"),
                    pl.lit(sector_canonical).alias("sector"),
                    pl.lit(meta["period_start"]).alias("period_start"),
                    pl.lit(meta["period_end"]).alias("period_end"),
                ])

                # Nombre del fichero parquet: {base_name}-{sheet_name}.parquet
                safe_sheet = sheet_name.replace(" ", "_")
                parquet_name = f"{base_name}-{safe_sheet}.parquet"
                parquet_path = output_sector_path / parquet_name

                df.write_parquet(parquet_path)
                hojas_escritas += 1

            if hojas_escritas > 0:
                ficheros_ok += 1

        stats[sector_canonical] = ficheros_ok
        logger.info(
            "Sector %s: %d ficheros convertidos a Parquet",
            sector_canonical,
            ficheros_ok,
        )

    return stats
