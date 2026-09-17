"""Nodos del pipeline report_matching.

Informe de discrepancias del matching sector-divisa (dominio de reporting
`matching`). El contenido se genera aqui, en la capa de reporting, a partir de
los datasets enriquecidos (ENRICHED_{cap}_{canonical}) ya persistidos por el
pipeline de transformacion enrich_sectors_with_currencies.
"""

from datetime import datetime

import polars as pl


def generate_matching_report(**kwargs: pl.DataFrame) -> str:
    """Genera un informe de discrepancias del matching sector-divisas.

    Identifica:
    - Empresas del SECTOR que no tienen divisa (sin match en currencies).
    - Tickers en currencies que no aparecen en ningun SECTOR.

    Args:
        **kwargs: Pares de keyword arguments. Las keys con prefijo 'sector_'
                  son DataFrames de SECTOR, las keys con prefijo 'currencies_'
                  son DataFrames de currencies, las keys con prefijo 'enriched_'
                  son los DataFrames enriquecidos resultantes.

    Returns:
        String Markdown con el informe de discrepancias.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# Informe de Matching: Sectores ↔ Divisas",
        "",
        f"**Generado**: {timestamp}",
        "",
        "---",
        "",
    ]

    # Separar los DataFrames por tipo
    enriched_dfs = {
        k.replace("enriched_", ""): v
        for k, v in kwargs.items()
        if k.startswith("enriched_")
    }

    total_matched = 0
    total_unmatched = 0

    for key in sorted(enriched_dfs.keys()):
        df = enriched_dfs[key]
        matched = df.filter(pl.col("currency").is_not_null()).shape[0]
        unmatched = df.filter(pl.col("currency").is_null()).shape[0]
        total_matched += matched
        total_unmatched += unmatched

        if unmatched > 0:
            lines.append(f"## {key} — {unmatched} sin emparejar")
            lines.append("")
            unmatched_df = df.filter(pl.col("currency").is_null())
            for row in unmatched_df.iter_rows(named=True):
                lines.append(f"- **{row.get('Simbolo', '')}** — {row.get('Empresa', '')}")
            lines.append("")

    # Resumen
    lines.append("---")
    lines.append("")
    lines.append(f"**Total emparejados: {total_matched}**")
    lines.append(f"**Total sin emparejar: {total_unmatched}**")
    total = total_matched + total_unmatched
    if total > 0:
        lines.append(f"**Tasa de matching: {total_matched / total:.1%}**")
    lines.append("")

    return "\n".join(lines)
