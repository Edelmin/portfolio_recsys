"""Nodos del pipeline report_sampling.

Informe de muestra estratificada para descarga manual (human-in-the-loop).

El contenido (Markdown con checkboxes) se genera aqui, en la capa de reporting,
a partir de las muestras ya persistidas (SAMPLE_{cap}_{canonical}). La conversion
a PDF es una accion manual y puntual que vive fuera del DAG, en la tool CLI
`pr-report-pdf` (portfolio_recsys.tools.report_to_pdf).
"""

from datetime import datetime

import polars as pl


def generate_sample_report(**samples: pl.DataFrame) -> str:
    """Genera un informe Markdown con la lista de empresas seleccionadas.

    Este informe esta pensado para uso humano: el operador lo consulta
    para saber que estados financieros debe descargar manualmente de TIKR.
    Incluye checkboxes para facilitar el seguimiento de descargas.

    Args:
        **samples: Keyword arguments donde cada key es "{cap}_{canonical}"
                   y cada value es el DataFrame de la muestra correspondiente.

    Returns:
        String con el contenido Markdown del informe.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# Informe de Muestra Estratificada",
        "",
        f"**Generado**: {timestamp}",
        "",
        "Lista de empresas seleccionadas para descarga manual de estados financieros.",
        "Marcar con `[x]` las empresas ya descargadas.",
        "",
        "---",
        "",
    ]

    total_empresas = 0

    for key in sorted(samples.keys()):
        df = samples[key]
        cap, sector = key.split("_", 1)
        n = df.shape[0]
        total_empresas += n

        lines.append(f"## {cap} — {sector} ({n} empresas)")
        lines.append("")

        for row in df.iter_rows(named=True):
            simbolo = row.get("Simbolo", "")
            empresa = row.get("Empresa", "")
            industria = row.get("Industria", "")
            ubicacion = row.get("Ubicacion", "")
            lines.append(
                f"- [ ] **{simbolo}** — {empresa} | {industria} | {ubicacion}"
            )

        lines.append("")

    # Resumen al final
    lines.append("---")
    lines.append("")
    lines.append(f"**Total de empresas a descargar: {total_empresas}**")
    lines.append("")

    return "\n".join(lines)
