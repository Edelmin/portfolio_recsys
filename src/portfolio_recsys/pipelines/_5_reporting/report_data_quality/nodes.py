"""Nodos del pipeline report_data_quality.

Analiza los Parquets de estados financieros para generar un informe de:
- Hojas disponibles por empresa (Income Statement, Cash Flow, Ratios).
- Campos/metricas presentes en cada hoja.
- Cobertura temporal por empresa (años fiscales cubiertos).
- Nulls y gaps en metricas clave.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import polars as pl

from portfolio_recsys.pipelines._1_ingest.parse_financial_statements.nodes import (
    parse_filename,
)
from portfolio_recsys.pipelines.sector_mapping import SECTORS_WITH_FS

logger = logging.getLogger(__name__)

SHEETS = ["Income_Statement", "Cash_Flow", "Ratios"]


def generate_data_quality_report(
    fs_base_path: str,
    cap: str,
    start_date: str,
    end_date: str,
) -> str:
    """Genera un informe JSON de cobertura y calidad de los estados financieros.

    Args:
        fs_base_path: Ruta base de los Parquets de FS.
        cap: Capitalizacion ("LargeCaps" o "SmallCaps").
        start_date: Inicio del periodo de estudio (para evaluar cobertura).
        end_date: Fin del periodo de estudio.

    Returns:
        JSON con el informe de calidad.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    base = Path(fs_base_path) / cap

    # Parsear año de inicio/fin para evaluar cobertura
    study_start_year = int(start_date[:4])
    study_end_year = int(end_date[:4])

    sector_reports = {}
    total_tickers = 0
    total_with_full_coverage = 0

    for sector in SECTORS_WITH_FS:
        sector_path = base / sector
        if not sector_path.exists():
            sector_reports[sector] = {"status": "NO_FOLDER", "tickers": []}
            continue

        # Agrupar ficheros por ticker
        ticker_files: dict[str, dict[str, Path]] = {}  # ticker -> {sheet: path}
        for f in sorted(sector_path.glob("*.parquet")):
            # Determinar hoja del nombre: {base}-{Sheet_Name}.parquet
            stem = f.stem
            sheet = None
            for s in SHEETS:
                if stem.endswith(f"-{s}"):
                    sheet = s
                    break
            if not sheet:
                continue

            # Extraer ticker del nombre base
            base_name = stem.replace(f"-{sheet}", "")
            meta = parse_filename(base_name + ".xlsx")
            if not meta:
                continue
            ticker = meta["ticker"]

            ticker_files.setdefault(ticker, {})[sheet] = f

        ticker_details = []
        for ticker, sheets_dict in sorted(ticker_files.items()):
            total_tickers += 1

            detail = {
                "ticker": ticker,
                "sheets_available": sorted(sheets_dict.keys()),
                "sheets_missing": [s for s in SHEETS if s not in sheets_dict],
                "coverage": {},
                "metrics_count": {},
                "null_ratio": {},
            }

            # Analizar cada hoja disponible
            for sheet_name, sheet_path in sheets_dict.items():
                try:
                    df = pl.read_parquet(sheet_path)
                except Exception:
                    continue

                if df.is_empty() or df.shape[1] < 2:
                    continue

                # Las columnas (excepto la primera) son fechas: dd/MM/yy o dd/MM/yyyy
                date_cols = [c for c in df.columns[1:] if c != "LTM"]
                years = set()
                for dc in date_cols:
                    try:
                        parts = dc.split("/")
                        if len(parts) == 3:
                            year = parts[2]
                            if len(year) == 2:
                                year = int(year)
                                year = 2000 + year if year < 50 else 1900 + year
                            else:
                                year = int(year)
                            years.add(year)
                    except (ValueError, IndexError):
                        pass

                # Metricas (primera columna)
                metric_col = df.columns[0]
                metrics = [
                    m for m in df[metric_col].to_list()
                    if m and not m.endswith(":")  # Excluir headers de seccion
                ]

                # Null ratio: proporcion de celdas vacias en las columnas de datos
                data_cells = df.shape[0] * len(date_cols)
                if data_cells > 0:
                    null_count = sum(
                        df[c].null_count() for c in date_cols
                    )
                    null_pct = round(null_count / data_cells * 100, 1)
                else:
                    null_pct = 0.0

                # Cobertura dentro del periodo de estudio
                covered_years = sorted([y for y in years if study_start_year <= y <= study_end_year])

                detail["coverage"][sheet_name] = {
                    "years_available": sorted(years),
                    "years_in_study_period": covered_years,
                    "coverage_years": len(covered_years),
                    "expected_years": study_end_year - study_start_year,
                }
                detail["metrics_count"][sheet_name] = len(metrics)
                detail["null_ratio"][sheet_name] = null_pct

            # Evaluar cobertura completa (todas las hojas cubren el periodo)
            has_full_coverage = all(
                detail["coverage"].get(s, {}).get("coverage_years", 0)
                >= (study_end_year - study_start_year) * 0.8  # 80% como umbral
                for s in SHEETS
                if s in sheets_dict
            )
            detail["full_coverage"] = has_full_coverage
            if has_full_coverage:
                total_with_full_coverage += 1

            ticker_details.append(detail)

        sector_reports[sector] = {
            "status": "OK",
            "total_tickers": len(ticker_details),
            "with_full_coverage": sum(1 for t in ticker_details if t["full_coverage"]),
            "with_all_sheets": sum(1 for t in ticker_details if not t["sheets_missing"]),
            "tickers": ticker_details,
        }

    # Resumen global
    report = {
        "generated_at": timestamp,
        "parameters": {
            "fs_base_path": fs_base_path,
            "cap": cap,
            "study_period": f"{start_date} → {end_date}",
        },
        "summary": {
            "total_sectors": len(SECTORS_WITH_FS),
            "total_tickers": total_tickers,
            "with_full_coverage": total_with_full_coverage,
            "coverage_rate": round(total_with_full_coverage / total_tickers * 100, 1) if total_tickers > 0 else 0,
        },
        "by_sector": sector_reports,
    }

    return json.dumps(report, indent=2, ensure_ascii=False)
