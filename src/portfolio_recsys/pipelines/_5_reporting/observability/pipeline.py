"""Pipeline observability.

Genera un informe legible (Markdown) del historico de ejecuciones auditadas.

Datasets:
  - Input:  ninguno (lee el historico JSONL desde disco via el modulo de auditoria)
  - Output: observability_report (Markdown en data/06_reporting/observability/)
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import generate_observability_report


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=generate_observability_report,
                inputs=None,
                outputs="observability_report",
                name="generate_observability_report",
            ),
        ],
        tags=["observability"],
    )
