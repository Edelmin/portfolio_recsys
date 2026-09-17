"""Nodos del pipeline report_exchange_rates.

Valida la completitud de las descargas de pares de divisa y genera
un informe con el estado de cada par, cobertura temporal y alertas.
"""

import json
from datetime import datetime

import polars as pl


def generate_exchange_rates_report(
    fetch_status: pl.DataFrame,
    rates: pl.DataFrame,
    start_date: str,
    end_date: str,
) -> str:
    """Genera un informe JSON de validacion de los pares de divisa descargados.

    Verifica:
    - Pares con estado OK, NO_DATA o ERROR.
    - Cobertura temporal: ¿el date_min y date_max cubren el periodo solicitado?
    - Gaps significativos en las series (> 5 dias habiles consecutivos).

    Args:
        fetch_status: DataFrame con el estado de cada descarga.
        rates: DataFrame consolidado de tipos de cambio.
        start_date: Fecha de inicio configurada.
        end_date: Fecha de fin configurada.

    Returns:
        String JSON con el informe de validacion.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Resumen por estado
    if fetch_status.is_empty():
        status_counts = {}
    else:
        status_counts = dict(
            zip(
                fetch_status.group_by("status").len()["status"].to_list(),
                fetch_status.group_by("status").len()["len"].to_list(),
            )
        )

    # Analisis de cobertura por par
    coverage_issues = []
    if not fetch_status.is_empty():
        for row in fetch_status.filter(pl.col("status") == "OK").iter_rows(named=True):
            pair = row["pair"]
            date_min = row.get("date_min", "")
            date_max = row.get("date_max", "")

            issues = []
            if date_min and date_min > start_date:
                issues.append(f"inicio real ({date_min}) posterior al solicitado ({start_date})")
            if date_max and date_max < end_date:
                issues.append(f"fin real ({date_max}) anterior al solicitado ({end_date})")

            if issues:
                coverage_issues.append({
                    "pair": pair,
                    "currency": row.get("currency", ""),
                    "date_min": date_min,
                    "date_max": date_max,
                    "issues": issues,
                })

    # Analisis de gaps (> 5 dias entre registros consecutivos)
    gap_alerts = []
    if not rates.is_empty():
        for currency in rates["currency"].unique().to_list():
            currency_data = rates.filter(pl.col("currency") == currency).sort("date")
            if currency_data.shape[0] < 2:
                continue

            dates = currency_data["date"].to_list()
            max_gap_days = 0
            max_gap_date = None
            for i in range(1, len(dates)):
                gap = (dates[i] - dates[i - 1]).days
                if gap > max_gap_days:
                    max_gap_days = gap
                    max_gap_date = str(dates[i])[:10]

            if max_gap_days > 5:
                gap_alerts.append({
                    "currency": currency,
                    "max_gap_days": max_gap_days,
                    "gap_after_date": max_gap_date,
                })

    # Pares fallidos
    failed_pairs = []
    if not fetch_status.is_empty():
        for row in fetch_status.filter(pl.col("status") != "OK").iter_rows(named=True):
            failed_pairs.append({
                "pair": row["pair"],
                "currency": row.get("currency", ""),
                "status": row["status"],
                "error": row.get("error", ""),
            })

    report = {
        "generated_at": timestamp,
        "requested_period": {"start": start_date, "end": end_date},
        "summary": {
            "total_pairs": fetch_status.shape[0] if not fetch_status.is_empty() else 0,
            "status_counts": status_counts,
            "total_records": rates.shape[0] if not rates.is_empty() else 0,
        },
        "failed_pairs": failed_pairs,
        "coverage_issues": coverage_issues,
        "gap_alerts": gap_alerts,
    }

    return json.dumps(report, indent=2, ensure_ascii=False)
