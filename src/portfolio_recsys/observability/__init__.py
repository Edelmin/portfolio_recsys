"""Observabilidad y auditoria de ejecuciones del proyecto.

Este subpaquete centraliza:
    - Las rutas donde se persisten los artefactos de auditoria (JSON por
      ejecucion + historico JSONL).
    - La logica de captura de metadatos de ejecucion (timestamps, duracion,
      parametros, commit de git, datasets generados, estado y errores).

Los hooks de Kedro (``portfolio_recsys.hooks.RunLoggingHooks``) usan este modulo
para registrar cada ejecucion de forma estructurada y auditable.
"""

from portfolio_recsys.observability.audit import (
    AUDIT_ROOT,
    RUNS_HISTORY_FILE,
    RUNS_ROOT,
    RunAudit,
    load_run_history,
)

__all__ = [
    "AUDIT_ROOT",
    "RUNS_HISTORY_FILE",
    "RUNS_ROOT",
    "RunAudit",
    "load_run_history",
]
