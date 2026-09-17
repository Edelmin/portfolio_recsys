"""Nodos del pipeline observability.

Genera un informe legible (Markdown) del historico de ejecuciones auditadas,
a partir del JSONL que escriben los hooks (portfolio_recsys.observability.audit).
"""

from __future__ import annotations

from typing import Any

from portfolio_recsys.observability.audit import load_run_history


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m {secs:.0f}s"


def _fmt_commit(commit: str | None, dirty: Any) -> str:
    if not commit:
        return "-"
    short = commit[:8]
    if dirty is True:
        short += " (dirty)"
    return short


def generate_observability_report() -> str:
    """Construye un informe Markdown del historico de ejecuciones.

    Returns:
        Contenido Markdown con un resumen agregado y la tabla de ejecuciones
        (mas recientes primero). Si no hay historico, devuelve un aviso.
    """
    history = load_run_history()

    if not history:
        return (
            "# Informe de observabilidad de ejecuciones\n\n"
            "No hay ejecuciones registradas todavia. El historico se genera "
            "automaticamente al ejecutar cualquier pipeline (ver "
            "`data/06_reporting/observability/runs.jsonl`).\n"
        )

    # Orden: mas recientes primero (por start_time ISO, ordenable como texto).
    history_sorted = sorted(
        history, key=lambda r: r.get("start_time") or "", reverse=True
    )

    total = len(history_sorted)
    n_ok = sum(1 for r in history_sorted if r.get("status") == "ok")
    n_error = sum(1 for r in history_sorted if r.get("status") == "error")
    durations = [
        r["duration_seconds"]
        for r in history_sorted
        if isinstance(r.get("duration_seconds"), (int, float))
    ]
    avg_duration = sum(durations) / len(durations) if durations else None

    lines: list[str] = []
    lines.append("# Informe de observabilidad de ejecuciones")
    lines.append("")
    lines.append("## Resumen")
    lines.append("")
    lines.append(f"- Ejecuciones registradas: **{total}**")
    lines.append(f"- Correctas: **{n_ok}**  |  Con error: **{n_error}**")
    lines.append(f"- Duracion media: **{_fmt_duration(avg_duration)}**")
    last_start = history_sorted[0].get("start_time") or "-"
    lines.append(f"- Ultima ejecucion: **{last_start}**")
    lines.append("")

    # Agregado por pipeline.
    by_pipeline: dict[str, dict[str, Any]] = {}
    for r in history_sorted:
        name = r.get("pipeline_name") or "(default)"
        agg = by_pipeline.setdefault(
            name, {"runs": 0, "ok": 0, "error": 0, "durations": []}
        )
        agg["runs"] += 1
        if r.get("status") == "ok":
            agg["ok"] += 1
        elif r.get("status") == "error":
            agg["error"] += 1
        if isinstance(r.get("duration_seconds"), (int, float)):
            agg["durations"].append(r["duration_seconds"])

    lines.append("## Por pipeline")
    lines.append("")
    lines.append("| Pipeline | Ejecuciones | OK | Error | Duracion media |")
    lines.append("|----------|------------:|---:|------:|---------------:|")
    for name in sorted(by_pipeline):
        agg = by_pipeline[name]
        avg = (
            sum(agg["durations"]) / len(agg["durations"])
            if agg["durations"]
            else None
        )
        lines.append(
            f"| {name} | {agg['runs']} | {agg['ok']} | {agg['error']} "
            f"| {_fmt_duration(avg)} |"
        )
    lines.append("")

    # Historial detallado (ultimas 50 para no saturar).
    lines.append("## Historial de ejecuciones (mas recientes primero)")
    lines.append("")
    lines.append(
        "| Inicio (UTC) | Pipeline | Env | Estado | Duracion | Nodos "
        "| Datasets generados | Commit |"
    )
    lines.append(
        "|--------------|----------|-----|--------|---------:|-----:|"
        "-------------------:|--------|"
    )
    for r in history_sorted[:50]:
        estado = r.get("status", "-")
        emoji = {"ok": "✅ ok", "error": "❌ error"}.get(estado, estado)
        lines.append(
            "| {start} | {pipe} | {env} | {estado} | {dur} | {nodes} "
            "| {datasets} | {commit} |".format(
                start=r.get("start_time") or "-",
                pipe=r.get("pipeline_name") or "(default)",
                env=r.get("env") or "-",
                estado=emoji,
                dur=_fmt_duration(r.get("duration_seconds")),
                nodes=r.get("n_nodes", "-"),
                datasets=r.get("n_generated_datasets", "-"),
                commit=_fmt_commit(r.get("git_commit"), r.get("git_dirty")),
            )
        )
    lines.append("")

    if total > 50:
        lines.append(f"> Mostrando las 50 ejecuciones mas recientes de {total}.")
        lines.append("")

    return "\n".join(lines)
