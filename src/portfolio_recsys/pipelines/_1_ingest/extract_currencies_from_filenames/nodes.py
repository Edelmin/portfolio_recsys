"""Nodos del pipeline extract_currencies_from_filenames.

Parsea los nombres de fichero de los PartitionedDatasets de estados financieros
para extraer la moneda y el ticker de cada empresa, junto con las fechas de
cobertura temporal.

Convencion de fechas:
  Las columnas period_start y period_end se almacenan en formato ISO 8601 UTC
  (Datetime[us, UTC]). Esta convencion es compatible con los estandares
  financieros internacionales (MiFID II, FIX Protocol, ISO 20022).
"""

import re

import polars as pl


def parse_filename(filename: str) -> dict | None:
    """Extrae currency, ticker y periodo del nombre del fichero.

    Patron: '{currency} - {ticker} - Financials ({start} - {end})'
    Ejemplo: 'USD - BRK.A - Financials (31.12.05 - 31.12.24)'
    """
    name = filename.replace(".xlsx", "").split("/")[-1]
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


def _parse_period_dates(df: pl.DataFrame) -> pl.DataFrame:
    """Convierte period_start y period_end de string (dd.MM.yy o dd.MM.yyyy) a Datetime UTC.

    Sigue la convencion ISO 8601 para almacenamiento de fechas financieras.
    """
    for col in ("period_start", "period_end"):
        df = df.with_columns(
            pl.when(pl.col(col).str.len_chars() <= 8)
            .then(pl.col(col).str.to_datetime("%d.%m.%y", time_zone="UTC", strict=False))
            .otherwise(pl.col(col).str.to_datetime("%d.%m.%Y", time_zone="UTC", strict=False))
            .alias(col)
        )
    return df


def extract_currencies(fs_sector: dict, sector: str) -> pl.DataFrame:
    """Extrae moneda, ticker y periodo de los nombres de fichero de un sector.

    Args:
        fs_sector: Diccionario {filename: callable} del PartitionedDataset.
                   Solo se usan las keys (no se cargan los DataFrames).
        sector: Nombre canonico del sector para incluir en el output.

    Returns:
        polars DataFrame con columnas:
          - ticker (Utf8)
          - currency (Utf8)
          - sector (Utf8)
          - period_start (Datetime[us, UTC])
          - period_end (Datetime[us, UTC])
    """
    records = []
    for filename in fs_sector.keys():
        parsed = parse_filename(filename)
        if parsed:
            parsed["sector"] = sector
            records.append(parsed)

    if not records:
        return pl.DataFrame(
            schema={
                "ticker": pl.Utf8,
                "currency": pl.Utf8,
                "sector": pl.Utf8,
                "period_start": pl.Datetime("us", "UTC"),
                "period_end": pl.Datetime("us", "UTC"),
            }
        )

    df = pl.DataFrame(records)
    return _parse_period_dates(df)
