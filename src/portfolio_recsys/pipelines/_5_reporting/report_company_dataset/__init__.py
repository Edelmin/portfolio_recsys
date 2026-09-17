"""Pipeline report_company_dataset.

Genera un informe de calidad y métricas sobre los datasets consolidados
(COMPANY_{sector}) para todos los sectores disponibles.
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
