"""Read-only Rich dashboard for persisted evaluation and judge evidence."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import sleep
from typing import Any

from .contracts import identity, load_manifest
from .control import judge_all_status, pipeline_status, status as worker_status
from .plan import resolve_target


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError):
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(line) if line.strip() else None
        except json.JSONDecodeError:
            continue  # an in-progress final line is not evidence yet
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _utc_seconds(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "—"
    value = int(seconds)
    hours, value = divmod(value, 3600)
    minutes, seconds = divmod(value, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:d}:{seconds:02d}"


@dataclass(frozen=True)
class TargetStatus:
    name: str
    device: str
    routes: tuple[str, ...]
    expected: int
    succeeded: int
    failed: int
    generated_tokens: int
    generation_state: str
    generation_current_route: str | None
    generation_rate_per_second: float | None
    generation_eta_seconds: float | None
    judged: int
    judge_mean_total_25: float | None
    judge_failures: int
    judge_state: str
    judge_current_route: str | None
    judge_rate_per_second: float | None
    judge_eta_seconds: float | None
    judge_stalled: bool
    output: Path
    worker_pid: int | None
    worker_running: bool

    @property
    def completed(self) -> int:
        return self.succeeded + self.failed

    @property
    def missing(self) -> int:
        return max(0, self.expected - self.completed)

    @property
    def generated_percent(self) -> float:
        return 0.0 if not self.expected else 100.0 * self.completed / self.expected

    @property
    def judged_percent(self) -> float:
        return 0.0 if not self.succeeded else 100.0 * self.judged / self.succeeded

    @property
    def phase(self) -> str:
        if not self.expected:
            return "EMPTY"
        if not self.completed:
            return "GEN INITIALIZING" if self.worker_running else "GEN PENDING"
        if self.missing:
            return "GENERATING"
        if self.failed:
            return "GENERATION ERRORS"
        if self.judge_state == "RUNNING":
            return "JUDGE STALLED" if self.judge_stalled else "JUDGING"
        if self.judge_state in {"NOT STARTED", "DRY_RUN"}:
            return "AWAITING JUDGE"
        if self.judge_state in {"INCOMPLETE", "INTERRUPTED", "FAILED"}:
            return f"JUDGE {self.judge_state}"
        return "COMPLETE" if self.judged == self.succeeded else "JUDGE INCOMPLETE"

    @property
    def generation_phase(self) -> str:
        if self.failed:
            return "GENERATION ERRORS"
        if self.completed == self.expected:
            return "COMPLETE"
        # A monitor can lack permission to signal a worker owned by another
        # process context even while its atomically persisted progress is
        # advancing. Do not misreport that live evidence as an interruption.
        if self.worker_running or self.generation_state == "GENERATING":
            return self.generation_state
        return "PENDING" if not self.completed else "INTERRUPTED"


def target_status(config: dict[str, Any], config_path: Path, target: str) -> TargetStatus:
    plan = resolve_target(config, config_path, target)
    manifest = load_manifest(plan["manifest"])
    routes = tuple(plan["routes"] or [entry["route"] for entry in manifest["routes"]])
    route_counts = {entry["route"]: int(entry["records"]) for entry in manifest["routes"]}
    expected = sum(route_counts[route] for route in routes)
    output = plan["output"]
    # A managed screen may deliberately use --limit-per-route. The worker
    # persists its resolved count before loading a model, whereas the manifest
    # always describes the full benchmark. Prefer the live plan when valid so
    # the displayed percentage and ETA describe the run actually in progress.
    generation_progress = _read_json(output / "generation-progress.json")
    progress_expected = generation_progress.get("expected")
    if isinstance(progress_expected, int) and progress_expected >= 0:
        expected = progress_expected

    generated_by_id: dict[str, dict[str, Any]] = {}
    for row in _read_json_lines(output / "generations.jsonl"):
        if row.get("status") in {"ok", "error"} and row.get("route") in routes and isinstance(row.get("sample_id"), str):
            generated_by_id[identity(row["route"], row["sample_id"])] = row
    completed = list(generated_by_id.values())
    succeeded = sum(row.get("status") == "ok" for row in completed)
    failed = sum(row.get("status") == "error" for row in completed)
    generated_tokens = sum(int(row.get("output_tokens", 0)) for row in completed if row.get("status") == "ok")
    latest_generation = completed[-1] if completed else {}
    generation_state = str(generation_progress.get("status", "COMPLETE" if len(completed) == expected else "PENDING"))
    generation_current_route = generation_progress.get("current_route") if isinstance(generation_progress.get("current_route"), str) else latest_generation.get("route") if isinstance(latest_generation.get("route"), str) else None
    generation_started = _utc_seconds(generation_progress.get("started_at_utc"))
    generation_updated = _utc_seconds(generation_progress.get("updated_at_utc"))
    generation_new = int(generation_progress.get("completed", len(completed)))
    generation_elapsed = generation_updated - generation_started if generation_started is not None and generation_updated is not None else None
    generation_rate = generation_new / generation_elapsed if generation_state == "GENERATING" and generation_elapsed and generation_elapsed > 0 else None
    generation_eta = (expected - len(completed)) / generation_rate if generation_rate else None

    judged_by_id: dict[str, dict[str, Any]] = {}
    for row in _read_json_lines(output / "judgments.jsonl"):
        if isinstance(row.get("route"), str) and isinstance(row.get("sample_id"), str) and isinstance(row.get("total"), int):
            judged_by_id[identity(row["route"], row["sample_id"])] = row
    judgments = list(judged_by_id.values())
    totals = [int(row["total"]) for row in judgments]
    judge_run = _read_json(output / "judge-run.json")
    judge_state = str(judge_run.get("status", "NOT STARTED"))
    progress_at = _utc_seconds(judge_run.get("updated_at_utc") or judge_run.get("started_at_utc"))
    now = datetime.now(timezone.utc).timestamp()
    judge_stalled = judge_state == "RUNNING" and progress_at is not None and now - progress_at > 300
    scored_at_start = judge_run.get("scored_at_start", 0)
    started_at = _utc_seconds(judge_run.get("started_at_utc"))
    new_scores = max(0, len(judgments) - int(scored_at_start) if isinstance(scored_at_start, int) else len(judgments))
    elapsed = now - started_at if started_at is not None else None
    rate = new_scores / elapsed if judge_state == "RUNNING" and elapsed and elapsed > 0 and new_scores else None
    eta = (succeeded - len(judgments)) / rate if rate else None
    current = judge_run.get("current_batch")
    current_route = current.get("route") if isinstance(current, dict) and isinstance(current.get("route"), str) else None
    worker = worker_status(config, config_path, target)
    runtime_device = generation_progress.get("device") if isinstance(generation_progress.get("device"), str) else str(plan["device"])
    return TargetStatus(
        name=target, device=runtime_device, routes=routes, expected=expected,
        succeeded=succeeded, failed=failed, generated_tokens=generated_tokens,
        generation_state=generation_state, generation_current_route=generation_current_route,
        generation_rate_per_second=generation_rate, generation_eta_seconds=generation_eta,
        judged=len(judgments), judge_mean_total_25=sum(totals) / len(totals) if totals else None,
        judge_failures=len(_read_json_lines(output / "judge-failures.jsonl")),
        judge_state=judge_state, judge_current_route=current_route,
        judge_rate_per_second=rate, judge_eta_seconds=eta, judge_stalled=judge_stalled,
        output=output, worker_pid=worker["pid"] if isinstance(worker.get("pid"), int) else None,
        worker_running=bool(worker.get("running")),
    )


def gpu_status() -> list[dict[str, str]]:
    fields = "index,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu"
    try:
        result = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    names = ("index", "name", "memory_used_mib", "memory_total_mib", "utilization_percent", "power_w", "temperature_c")
    return [dict(zip(names, (value.strip() for value in line.split(",")), strict=True)) for line in result.stdout.splitlines() if line.strip()]


def _dashboard(statuses: list[TargetStatus], gpus: list[dict[str, str]], judge_control: dict[str, Any], pipeline_control: dict[str, Any], refresh_seconds: float):
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    total_generated = sum(status.succeeded for status in statuses)
    total_expected = sum(status.expected for status in statuses)
    total_judged = sum(status.judged for status in statuses)
    weighted_scores = sum((status.judge_mean_total_25 or 0.0) * status.judged for status in statuses)
    mean = weighted_scores / total_judged if total_judged else None
    active = [status for status in statuses if status.judge_state == "RUNNING" and not status.judge_stalled]
    active_generation = [
        status for status in statuses
        if status.worker_running or (status.generation_state == "GENERATING" and status.missing)
    ]
    control_command = judge_control.get("control", {}).get("command", []) if isinstance(judge_control.get("control"), dict) else []
    try:
        parallelism = control_command[control_command.index("--parallelism") + 1]
    except (ValueError, IndexError):
        parallelism = "—"
    judge_coordinator = f"RUNNING • PID {judge_control.get('pid')} • parallelism {parallelism}" if judge_control.get("running") else f"STOPPED • {judge_control.get('reason', 'no control record')}"
    pipeline_state = _read_json(Path(str(pipeline_control.get("output", ""))) / "pipeline-state.json")
    pipeline_phase = str(pipeline_state.get("status", "NOT STARTED"))
    if pipeline_control.get("running"):
        pipeline_coordinator = f"RUNNING • PID {pipeline_control.get('pid')} • {pipeline_phase}"
    elif pipeline_phase == "RUNNING" and active_generation:
        pipeline_coordinator = f"RUNNING • persisted evidence • PID unverified • {pipeline_phase}"
    else:
        pipeline_coordinator = f"STOPPED • {pipeline_control.get('reason', 'no control record')} • {pipeline_phase}"
    details = Table.grid(expand=True)
    details.add_column(style="bold cyan", no_wrap=True); details.add_column()
    details.add_row("End-to-end pipeline", pipeline_coordinator)
    details.add_row("Judge coordinator", judge_coordinator)
    details.add_row("Generation coverage", f"{sum(status.completed for status in statuses)}/{total_expected} records • {sum(status.generated_tokens for status in statuses):,} output tokens")
    details.add_row("Generation workers", ", ".join(f"{status.name} (PID {status.worker_pid})" for status in active_generation) or "none")
    details.add_row("Judge coverage", f"{total_judged}/{total_generated} valid answers ({100 * total_judged / total_generated if total_generated else 0.0:.1f}%)")
    details.add_row("Live micro mean", f"{mean:.2f}/25" if mean is not None else "—")
    details.add_row("Active targets", ", ".join(status.name for status in active) or "none")
    details.add_row("Managed log", str(judge_control.get("log", "—")))

    table = Table(expand=True)
    for title, justify in (("Target", "left"), ("GPU", "left"), ("Generation state", "left"), ("Generation", "right"), ("Gen route", "left"), ("Gen rate / ETA", "right"), ("Judge state", "left"), ("Judging", "right"), ("Judge route", "left"), ("Mean /25", "right"), ("Judge rate / ETA", "right"), ("Failed batches", "right")):
        table.add_column(title, justify=justify, overflow="fold")
    for status in statuses:
        table.add_row(
            status.name, status.device, status.generation_phase,
            f"{status.completed}/{status.expected} ({status.generated_percent:.1f}%)",
            status.generation_current_route or "—",
            (f"{status.generation_rate_per_second * 60:.1f}/min • {_duration(status.generation_eta_seconds)}" if status.generation_rate_per_second else "—"),
            status.phase,
            f"{status.judged}/{status.succeeded} ({status.judged_percent:.1f}%)",
            status.judge_current_route or "—",
            f"{status.judge_mean_total_25:.2f}" if status.judge_mean_total_25 is not None else "—",
            (f"{status.judge_rate_per_second * 60:.1f}/min • {_duration(status.judge_eta_seconds)}" if status.judge_rate_per_second else "—"),
            str(status.judge_failures),
        )

    gpu_table = Table(expand=True)
    for title, justify in (("GPU", "left"), ("Name", "left"), ("Memory", "right"), ("Util", "right"), ("Power", "right"), ("Temp", "right")):
        gpu_table.add_column(title, justify=justify)
    if not gpus:
        gpu_table.add_row("—", "nvidia-smi unavailable", "—", "—", "—", "—")
    for gpu in gpus:
        gpu_table.add_row(gpu["index"], gpu["name"], f"{gpu['memory_used_mib']}/{gpu['memory_total_mib']} MiB", f"{gpu['utilization_percent']}%", f"{gpu['power_w']} W", f"{gpu['temperature_c']} °C")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    header = Text(f"TELCOBENCH CONTROL MONITOR  •  {now}  •  refresh {refresh_seconds:g}s", style="bold cyan")
    note = Text("Read-only persisted-evidence monitor. Ctrl+C closes only this view. Use start-all|status-all|stop-all to control the full pipeline.", style="dim")
    return Group(header, Panel(details, title="End-to-end evaluation control"), Panel(table, title="Target progress"), Panel(gpu_table, title="GPU telemetry"), note)


def monitor(config: dict[str, Any], config_path: Path, targets: list[str], refresh_seconds: float) -> None:
    try:
        from rich.live import Live
    except ImportError as exc:
        raise RuntimeError("install telcobench[tui] for the terminal monitor") from exc
    try:
        with Live(refresh_per_second=max(1, int(1 / max(refresh_seconds, 0.1))), screen=True) as live:
            while True:
                statuses = [target_status(config, config_path, target) for target in targets]
                live.update(_dashboard(statuses, gpu_status(), judge_all_status(config, config_path), pipeline_status(config, config_path), refresh_seconds), refresh=True)
                sleep(refresh_seconds)
    except KeyboardInterrupt:
        return
