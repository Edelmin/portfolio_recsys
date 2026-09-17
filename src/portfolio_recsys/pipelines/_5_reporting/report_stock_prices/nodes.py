"""Nodos del pipeline report_stock_prices.

Genera un informe JSON con metricas de la descarga de precios:
- Tasa de éxito por sector y capitalización.
- Tickers fallidos.
- Discrepancias de moneda (esperada vs reportada).
- Cobertura temporal.
"""

import json
from datetime import datetime

import polars as pl


def generate_stock_prices_report(
    fetch_status: pl.DataFrame,
    start_date: str,
    end_date: str,
) -> str:
    """Genera un informe JSON con metricas de la descarga de precios.

    Args:
        fetch_status: DataFrame con el estado de cada ticker descargado.
        start_date: Fecha de inicio del periodo solicitado.
        end_date: Fecha de fin del periodo solicitado.

    Returns:
        String JSON con el informe.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    if fetch_status.is_empty():
        return json.dumps({"generated_at": timestamp, "error": "Sin datos de fetch_status"})

    total = fetch_status.shape[0]
    ok = fetch_status.filter(pl.col("status") == "OK")
    failed = fetch_status.filter(pl.col("status") != "OK")

    # Resumen general
    status_counts = dict(
        zip(
            fetch_status.group_by("status").len()["status"].to_list(),
            fetch_status.group_by("status").len()["len"].to_list(),
        )
    )

    # Tasa de éxito por sector
    by_sector = (
        fetch_status.group_by(["sector", "status"])
        .len()
        .pivot(on="status", index="sector", values="len")
        .fill_null(0)
    )
    sector_summary = {}
    for row in by_sector.iter_rows(named=True):
        sector = row.pop("sector")
        sector_summary[sector] = row

    # Tasa de éxito por capitalización
    by_cap = (
        fetch_status.group_by(["cap", "status"])
        .len()
        .pivot(on="status", index="cap", values="len")
        .fill_null(0)
    )
    cap_summary = {}
    for row in by_cap.iter_rows(named=True):
        cap = row.pop("cap")
        cap_summary[cap] = row

    # Discrepancias de moneda
    currency_mismatches = []
    if not ok.is_empty():
        mismatches = ok.filter(
            pl.col("currency_expected") != pl.col("currency_reported")
        )
        for row in mismatches.iter_rows(named=True):
            currency_mismatches.append({
                "ticker": row["ticker"],
                "sector": row["sector"],
                "expected": row["currency_expected"],
                "reported": row["currency_reported"],
            })

    # Cobertura temporal
    coverage_issues = []
    if not ok.is_empty():
        for row in ok.iter_rows(named=True):
            issues = []
            date_min = row.get("date_min", "")
            date_max = row.get("date_max", "")
            if date_min and date_min > start_date:
                issues.append(f"inicio real ({date_min}) > solicitado ({start_date})")
            if date_max and date_max < end_date:
                issues.append(f"fin real ({date_max}) < solicitado ({end_date})")
            if issues:
                coverage_issues.append({
                    "ticker": row["ticker"],
                    "date_min": date_min,
                    "date_max": date_max,
                    "issues": issues,
                })

    # Tickers fallidos
    failed_tickers = []
    for row in failed.iter_rows(named=True):
        failed_tickers.append({
            "ticker": row["ticker"],
            "sector": row["sector"],
            "cap": row["cap"],
            "status": row["status"],
            "error": row.get("error", ""),
        })

    report = {
        "generated_at": timestamp,
        "requested_period": {"start": start_date, "end": end_date},
        "summary": {
            "total_tickers": total,
            "status_counts": status_counts,
            "success_rate": round(ok.shape[0] / total * 100, 1) if total > 0 else 0,
            "total_records": int(ok["records"].sum()) if not ok.is_empty() else 0,
        },
        "by_sector": sector_summary,
        "by_cap": cap_summary,
        "currency_mismatches": currency_mismatches,
        "coverage_issues_count": len(coverage_issues),
        "coverage_issues": coverage_issues[:50],  # Limitar a 50 para legibilidad
        "failed_tickers": failed_tickers,
    }

    return json.dumps(report, indent=2, ensure_ascii=False)
