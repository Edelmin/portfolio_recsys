"""Hooks de ciclo de vida de Kedro para logging y auditoria de ejecuciones.

``RunLoggingHooks`` registra cada ejecucion de forma estructurada y persistida
(ver ``portfolio_recsys.observability.audit``): timestamps, duracion, parametros,
commit de git, tiempos por nodo, datasets generados y estado/errores.

Toda la logica de auditoria es tolerante a fallos: cualquier error en el registro
se loguea pero NUNCA interrumpe la ejecucion del pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

from kedro.framework.hooks import hook_impl

from portfolio_recsys.observability.audit import RunAudit

logger = logging.getLogger(__name__)


class RunLoggingHooks:
    """Captura metadatos de ejecucion y los persiste como auditoria."""

    def __init__(self) -> None:
        self._audit: RunAudit | None = None
        self._catalog: Any = None

    # ── Ciclo de vida del pipeline ───────────────────────────────────────────

    @hook_impl
    def before_pipeline_run(self, run_params, pipeline, catalog):
        pipeline_name = self._resolve_pipeline_name(run_params)
        logger.info("Starting Kedro pipeline run: %s", pipeline_name)

        self._catalog = catalog
        try:
            self._audit = RunAudit.start(
                pipeline_name=pipeline_name,
                env=run_params.get("env"),
                runtime_params=run_params.get("extra_params")
                or run_params.get("runtime_params")
                or {},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("No se pudo iniciar la auditoria de la ejecucion: %s", exc)
            self._audit = None

    @staticmethod
    def _resolve_pipeline_name(run_params) -> str | None:
        """Normaliza el nombre del pipeline desde run_params.

        Kedro pasa ``pipeline_names`` como lista (ej: ['observability']) o, en
        algunas versiones, ``pipeline_name`` como cadena. Devuelve una cadena
        legible o None si es el pipeline por defecto.
        """
        names = run_params.get("pipeline_names")
        if isinstance(names, (list, tuple, set)):
            names = [n for n in names if n]
            if names:
                return ", ".join(sorted(str(n) for n in names))
            return None
        if isinstance(names, str) and names:
            return names
        single = run_params.get("pipeline_name")
        return single if isinstance(single, str) and single else None

    @hook_impl
    def after_pipeline_run(self, run_params, run_result, pipeline, catalog):
        logger.info("Kedro pipeline run finished.")
        if self._audit is None:
            return
        try:
            generated = self._persisted_outputs(pipeline, catalog)
            self._audit.finish(status="ok", generated_datasets=generated)
        except Exception as exc:  # noqa: BLE001
            logger.warning("No se pudo cerrar la auditoria de la ejecucion: %s", exc)

    @hook_impl
    def on_pipeline_error(self, error, run_params, pipeline, catalog):
        logger.exception("Kedro pipeline run failed: %s", error)
        if self._audit is None:
            return
        try:
            generated = self._persisted_outputs(pipeline, catalog)
            self._audit.finish(
                status="error",
                generated_datasets=generated,
                error=f"{type(error).__name__}: {error}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "No se pudo registrar el error en la auditoria de la ejecucion: %s", exc
            )

    # ── Ciclo de vida de los nodos ───────────────────────────────────────────

    @hook_impl
    def before_node_run(self, node, catalog, inputs, is_async):
        if self._audit is not None:
            try:
                self._audit.node_started(node.name)
            except Exception:  # noqa: BLE001
                pass

    @hook_impl
    def after_node_run(self, node, catalog, inputs, outputs, is_async):
        if self._audit is not None:
            try:
                self._audit.node_finished(node.name)
            except Exception:  # noqa: BLE001
                pass

    @hook_impl
    def on_node_error(self, error, node, catalog, inputs, is_async):
        if self._audit is not None:
            try:
                self._audit.node_errored(
                    node.name, error=f"{type(error).__name__}: {error}"
                )
            except Exception:  # noqa: BLE001
                pass

    # ── Utilidades ───────────────────────────────────────────────────────────

    @staticmethod
    def _persisted_outputs(pipeline, catalog) -> list[str]:
        """Lista los outputs del pipeline que se materializan en disco.

        Excluye parametros y datasets en memoria (MemoryDataset), quedandose con
        los datasets que realmente se persisten (tienen filepath/path).
        """
        try:
            candidate_outputs = set(pipeline.all_outputs())
        except Exception:  # noqa: BLE001
            return []

        persisted: list[str] = []
        for name in candidate_outputs:
            if name.startswith("params:") or name == "parameters":
                continue
            try:
                dataset = catalog._get_dataset(name)  # noqa: SLF001
            except Exception:  # noqa: BLE001
                # Si no se puede resolver, lo incluimos igualmente como generado.
                persisted.append(name)
                continue
            type_name = type(dataset).__name__
            if type_name == "MemoryDataset":
                continue
            persisted.append(name)
        return persisted
