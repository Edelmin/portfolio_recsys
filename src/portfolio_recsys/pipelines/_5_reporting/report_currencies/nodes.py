"""Nodos del pipeline report_currencies.

Genera un resumen de la distribución de divisas en el universo de empresas.
"""

import json

import polars as pl


def generate_currency_report(**sector_currencies: pl.DataFrame) -> str:
    """Genera un JSON con el resumen de divisas por sector.

    Args:
        **sector_currencies: Keyword arguments donde cada key es el nombre
            canónico del sector y el value es el DataFrame de currencies.

    Returns:
        String JSON con la estructura:
        {
            "total_tickers": int,
            "total_currencies": int,
            "currencies": {"USD": 150, "EUR": 80, ...},
            "by_sector": {
                "Financials": {"total": 50, "currencies": {"USD": 30, ...}},
                ...
            }
        }
    """
    all_dfs = []
    by_sector = {}

    for sector, df in sorted(sector_currencies.items()):
        if df.is_empty():
            by_sector[sector] = {"total": 0, "currencies": {}}
            continue

        all_dfs.append(df)

        # Conteo por divisa en este sector
        counts = (
            df.group_by("currency")
            .len()
            .sort("len", descending=True)
        )
        by_sector[sector] = {
            "total": df.shape[0],
            "currencies": dict(
                zip(
                    counts["currency"].to_list(),
                    counts["len"].to_list(),
                )
            ),
        }

    # Totales
    if all_dfs:
        combined = pl.concat(all_dfs)
        total_counts = (
            combined.group_by("currency")
            .len()
            .sort("len", descending=True)
        )
        currencies_summary = dict(
            zip(
                total_counts["currency"].to_list(),
                total_counts["len"].to_list(),
            )
        )
        total_tickers = combined.shape[0]
        total_currencies = combined["currency"].n_unique()
    else:
        currencies_summary = {}
        total_tickers = 0
        total_currencies = 0

    report = {
        "total_tickers": total_tickers,
        "total_currencies": total_currencies,
        "currencies": currencies_summary,
        "by_sector": by_sector,
    }

    return json.dumps(report, indent=2, ensure_ascii=False)
