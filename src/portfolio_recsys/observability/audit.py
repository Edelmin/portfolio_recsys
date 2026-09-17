"""Registro de auditoria de ejecuciones (trazabilidad integral).

Fuente unica de verdad para las rutas y la logica de persistencia de la
auditoria de ejecuciones. Cada ejecucion de un pipeline de Kedro produce:

    - Un JSON detallado por ejecucion en
      ``data/06_reporting/observability/runs/{run_id}.json``.
    - Una linea append-only en el historico
      ``data/06_reporting/observability/runs.jsonl`` (una linea por ejecucion),
      pensada para consultas rapidas del tipo "que se lanzo, cuando y que genero".

El diseño es tolerante a fallos: si algo del registro de auditoria falla, se
registra el error pero NUNCA se interrumpe la ejecucion del pipeline.
"""

from __future__ import annotations

import json
import logging
import subprocess
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Rutas de auditoria (fuente unica de verdad)
# ─────────────────────────────────────────────────────────────────────────────

DATA_ROOT = Path("data")
AUDIT_ROOT = DATA_ROOT / "06_reporting" / "observability"
RUNS_ROOT = AUDIT_ROOT / "runs"
RUNS_HISTORY_FILE = AUDIT_ROOT / "runs.jsonl"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _git_info() -> dict[str, Any]:
    """Obtiene el commit actual y si el working tree tiene cambios sin confirmar.

    Devuelve claves con valor None si git no esta disponible o no es un repo.
    """
    info: dict[str, Any] = {"commit": None, "dirty": None, "branch": None}
    try:
        info["commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        info["branch"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode()
        info["dirty"] = bool(status.strip())
    except Exception:  # noqa: BLE001 - git opcional, no debe romper la auditoria
        pass
    return info


@dataclass
class NodeTiming:
    """Metricas de ejecucion de un nodo."""

    node: str
    start: datetime | None = None
    end: datetime | None = None
    duration_seconds: float | None = None
    status: str = "running"  # running | ok | error
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "start": _iso(self.start),
            "end": _iso(self.end),
            "duration_seconds": self.duration_seconds,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class RunAudit:
    """Acumula los metadatos de una ejecucion y los persiste al finalizar.

    Uso tipico desde los hooks de Kedro:
        audit = RunAudit.start(pipeline_name=..., env=..., runtime_params=...)
        audit.node_started("nodo")
        audit.node_finished("nodo", outputs={...})
        audit.finish(status="ok", generated_datasets=[...])
    """

    run_id: str
    pipeline_name: str | None
    env: str | None
    runtime_params: dict[str, Any]
    start_time: datetime
    git: dict[str, Any] = field(default_factory=dict)
    node_timings: dict[str, NodeTiming] = field(default_factory=dict)
    end_time: datetime | None = None
    status: str = "running"  # running | ok | error
    error: str | None = None
    generated_datasets: list[str] = field(default_factory=list)

    # --- Construccion ---

    @classmethod
    def start(
        cls,
        pipeline_name: str | None,
        env: str | None,
        runtime_params: dict[str, Any] | None,
    ) -> "RunAudit":
        return cls(
            run_id=_utc_now().strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8],
            pipeline_name=pipeline_name,
            env=env,
            runtime_params=dict(runtime_params or {}),
            start_time=_utc_now(),
            git=_git_info(),
        )

    # --- Eventos de nodo ---

    def node_started(self, node_name: str) -> None:
        self.node_timings[node_name] = NodeTiming(
            node=node_name, start=_utc_now(), status="running"
        )

    def node_finished(self, node_name: str) -> None:
        timing = self.node_timings.get(node_name) or NodeTiming(node=node_name)
        timing.end = _utc_now()
        if timing.start is not None:
            timing.duration_seconds = round(
                (timing.end - timing.start).total_seconds(), 4
            )
        timing.status = "ok"
        self.node_timings[node_name] = timing

    def node_errored(self, node_name: str, error: str) -> None:
        timing = self.node_timings.get(node_name) or NodeTiming(node=node_name)
        timing.end = _utc_now()
        if timing.start is not None:
            timing.duration_seconds = round(
                (timing.end - timing.start).total_seconds(), 4
            )
        timing.status = "error"
        timing.error = error
        self.node_timings[node_name] = timing

    # --- Cierre ---

    def finish(
        self,
        status: str,
        generated_datasets: Iterable[str] | None = None,
        error: str | None = None,
    ) -> None:
        self.end_time = _utc_now()
        self.status = status
        self.error = error
        if generated_datasets is not None:
            self.generated_datasets = sorted(set(generated_datasets))
        self._persist()

    # --- Serializacion ---

    @property
    def duration_seconds(self) -> float | None:
        if self.end_time is None:
            return None
        return round((self.end_time - self.start_time).total_seconds(), 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "pipeline_name": self.pipeline_name,
            "env": self.env,
            "status": self.status,
            "start_time": _iso(self.start_time),
            "end_time": _iso(self.end_time),
            "duration_seconds": self.duration_seconds,
            "git": self.git,
            "runtime_params": self.runtime_params,
            "generated_datasets": self.generated_datasets,
            "n_nodes": len(self.node_timings),
            "node_timings": [t.to_dict() for t in self.node_timings.values()],
            "error": self.error,
        }

    def _summary_dict(self) -> dict[str, Any]:
        """Version compacta para el historico JSONL (una linea por ejecucion)."""
        return {
            "run_id": self.run_id,
            "pipeline_name": self.pipeline_name,
            "env": self.env,
            "status": self.status,
            "start_time": _iso(self.start_time),
            "end_time": _iso(self.end_time),
            "duration_seconds": self.duration_seconds,
            "git_commit": self.git.get("commit"),
            "git_dirty": self.git.get("dirty"),
            "n_nodes": len(self.node_timings),
            "n_generated_datasets": len(self.generated_datasets),
            "generated_datasets": self.generated_datasets,
            "error": self.error,
        }

    def _persist(self) -> None:
        """Escribe el JSON detallado y añade la linea al historico.

        Tolerante a fallos: si la escritura falla, se registra pero no propaga.
        """
        try:
            RUNS_ROOT.mkdir(parents=True, exist_ok=True)
            detail_path = RUNS_ROOT / f"{self.run_id}.json"
            detail_path.write_text(
                json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            with RUNS_HISTORY_FILE.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(self._summary_dict(), ensure_ascii=False) + "\n")

            logger.info(
                "Auditoria de ejecucion registrada: %s (status=%s, %.2fs)",
                detail_path,
                self.status,
                self.duration_seconds or 0.0,
            )
        except Exception as exc:  # noqa: BLE001 - la auditoria nunca rompe el run
            logger.warning("No se pudo persistir la auditoria de la ejecucion: %s", exc)


def load_run_history(history_file: Path | None = None) -> list[dict[str, Any]]:
    """Carga el historico de ejecuciones desde el fichero JSONL.

    Args:
        history_file: Ruta al JSONL. Por defecto ``RUNS_HISTORY_FILE``.

    Returns:
        Lista de registros (dicts), uno por ejecucion. Vacia si no existe.
    """
    path = history_file or RUNS_HISTORY_FILE
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Linea de historico ilegible, se omite: %s", line[:120])
    return records
