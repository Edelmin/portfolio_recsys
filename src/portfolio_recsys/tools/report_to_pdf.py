"""Convierte el informe de muestra estratificada de Markdown a PDF.

Usa md-to-pdf (Node.js) para renderizar el Markdown con checkboxes a PDF.

Uso:
    uv run python -m portfolio_recsys.tools.report_to_pdf

Prerequisitos:
    Node.js instalado (npx disponible en PATH).
    md-to-pdf se descarga automaticamente via npx en la primera ejecucion.
"""

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[3]
    report_dir = project_root / "data" / "06_reporting" / "stratified_sampling"
    md_path = report_dir / "sample_report.md"

    if not md_path.exists():
        print(f"Error: no se encontro {md_path}")
        print("Ejecuta primero: uv run kedro run --pipeline stratified_random_sampling")
        sys.exit(1)

    # Generar nombre con timestamp para el PDF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_name = f"sample_report_{timestamp}.pdf"
    pdf_path = report_dir / pdf_name

    # Verificar que npx esta disponible
    if not shutil.which("npx"):
        print("Error: npx no encontrado. Instala Node.js.")
        sys.exit(1)

    print(f"Convirtiendo {md_path.name} → {pdf_name} ...")

    result = subprocess.run(
        ["npx", "--yes", "md-to-pdf", str(md_path)],
        cwd=str(report_dir),
        capture_output=True,
        text=True,
        shell=True,
    )

    if result.returncode != 0:
        print(f"Error en md-to-pdf:\n{result.stderr}")
        sys.exit(1)

    # md-to-pdf genera el PDF con el mismo nombre base
    generated_pdf = md_path.with_suffix(".pdf")
    if generated_pdf.exists():
        generated_pdf.rename(pdf_path)
        print(f"PDF generado: {pdf_path.relative_to(project_root)}")
    else:
        print("Error: no se genero el PDF esperado.")
        sys.exit(1)


if __name__ == "__main__":
    main()
