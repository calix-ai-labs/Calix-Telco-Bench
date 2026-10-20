"""Dynamic-resource, evidence-gated generation and judging coordinator."""
from __future__ import annotations

import json
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import sleep
from typing import Any

from .control import start as start_worker, status as worker_status, stop as stop_worker
from .plan import resolve_target

_FREE_MEMORY_MIB = 1024
_FREE_UTILIZATION_PERCENT = 10
_MAX_CONCURRENT_JUDGES = 1  # conservative gateway protection; overlaps GPU generation


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _free_gpus(reserved: set[str]) -> list[str]:
    """Return physical GPUs that are actually idle now, not config-reserved."""
    command = ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    free: list[str] = []
    for line in result.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            index, memory, utilization = int(fields[0]), float(fields[1]), float(fields[2])
        except ValueError:
            continue
        device = f"cuda:{index}"
        if device not in reserved and memory <= _FREE_MEMORY_MIB and utilization <= _FREE_UTILIZATION_PERCENT:
            free.append(device)
    return free


def _coverage(plan: dict[str, Any]) -> dict[str, int] | None:
    value = _read_json(Path(plan["output"]) / "coverage.json")
    try:
        return {key: int(value.get(key, 0)) for key in ("expected", "records", "unique_records", "missing", "ok", "errors")}
    except (TypeError, ValueError):
        return None


def _generation_is_valid(plan: dict[str, Any]) -> tuple[bool, str]:
    coverage = _coverage(plan)
    if coverage is None:
        return False, "worker exited without coverage.json"
    if coverage["missing"]:
        return False, f"incomplete generation: {coverage['missing']} records missing"
    if coverage["errors"]:
        return False, f"generation emitted {coverage['errors']} error records"
    if coverage["unique_records"] != coverage["expected"]:
        return False, "generation coverage does not equal frozen manifest coverage"
    return True, "complete validated generation"


def _judge_is_complete(plan: dict[str, Any], route: str | None = None) -> tuple[bool, str]:
    suffix = f"--{route}" if route else ""
    status = str(_read_json(Path(plan["output"]) / f"judge-run{suffix}.json").get("status", ""))
    return (True, "complete") if status == "COMPLETED" else (False, f"judge ended with {status or 'no judge-run state'}")


def _start_target_judge(config_path: Path, target: str, output: Path, resume: bool, route: str | None = None) -> subprocess.Popen[str]:
    command = [sys.executable, "-m", "network_llm_eval.cli", "judge", "--config", str(config_path), "--target", target]
    if route:
        # Route coverage has already been verified by generation-progress.json.
        # The target as a whole is intentionally still incomplete.
        command.extend(["--routes", route, "--allow-incomplete", "--resume"])
    else:
        # Route jobs have already created judgments.jsonl, so the aggregate
        # pass must resume regardless of how generation itself was started.
        command.append("--resume")
    with (output / "judge-worker.log").open("a", encoding="utf-8") as log:
        return subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=config_path.parent, text=True)


def run_pipeline(config: dict[str, Any], config_path: Path, *, targets: list[str], limit_per_route: int | None, resume: bool) -> dict[str, Any]:
    """Dispatch to free GPUs and overlap validated per-target judging."""
    plans = {target: resolve_target(config, config_path, target) for target in targets}
    root = Path(next(iter(plans.values()))["output"]).parent
    state_path = root / "pipeline-state.json"
    generation_pending = list(targets)
    judge_pending: list[tuple[str, str | None]] = []
    active_generation: dict[str, str] = {}
    active_judges: dict[tuple[str, str | None], subprocess.Popen[str]] = {}
    generation_complete: list[str] = []
    judge_complete: list[str] = []
    failed: dict[str, str] = {}
    queued_route_judges: set[tuple[str, str]] = set()
    stopped = False

    def persist(stage: str, message: str | None = None) -> None:
        _write(state_path, {
            "schema_version": 2, "status": stage, "updated_at_utc": _now(), "targets": targets,
            "generation_pending": list(generation_pending), "active_generation": dict(active_generation),
            "generation_complete": list(generation_complete), "judge_pending": [f"{target}:{route or 'final'}" for target, route in judge_pending],
            "active_judges": {f"{target}:{route or 'final'}": process.pid for (target, route), process in active_judges.items()}, "judge_complete": list(judge_complete),
            "failed": dict(failed), "message": message,
            "scheduler": {"gpu_policy": "first currently free GPU", "free_memory_mib_max": _FREE_MEMORY_MIB,
                          "free_utilization_percent_max": _FREE_UTILIZATION_PERCENT,
                          "max_concurrent_judges": _MAX_CONCURRENT_JUDGES},
        })

    def terminate(_signum: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True

    old_term = signal.signal(signal.SIGTERM, terminate)
    old_int = signal.signal(signal.SIGINT, terminate)
    try:
        persist("QUEUED", "targets will be assigned to genuinely free GPUs")
        while (generation_pending or active_generation or judge_pending or active_judges) and not stopped:
            # No static GPU reservation: launch each pending target on the next
            # currently idle device, with at most one evaluator model per GPU.
            for device in _free_gpus(set(active_generation)):
                if not generation_pending:
                    break
                target = generation_pending.pop(0)
                try:
                    start_worker(config, config_path, target, limit_per_route=limit_per_route, resume=resume, device=device)
                except Exception as exc:
                    failed[target] = f"could not start on {device}: {type(exc).__name__}: {exc}"
                    continue
                active_generation[device] = target

            # A route becomes eligible as soon as its final answer is flushed.
            # This overlaps network-bound judging with GPU generation of later routes.
            for target in active_generation.values():
                progress = _read_json(Path(plans[target]["output"]) / "generation-progress.json")
                for route in progress.get("completed_routes", []):
                    key = (target, str(route))
                    if key not in queued_route_judges:
                        queued_route_judges.add(key); judge_pending.append(key)
            while judge_pending and len(active_judges) < _MAX_CONCURRENT_JUDGES:
                target, route = judge_pending.pop(0)
                try:
                    active_judges[(target, route)] = _start_target_judge(config_path, target, Path(plans[target]["output"]), resume, route)
                except Exception as exc:
                    failed[target] = f"could not start judge for {route or 'final'}: {type(exc).__name__}: {exc}"

            persist("RUNNING")
            sleep(1)
            for device, target in list(active_generation.items()):
                if worker_status(config, config_path, target)["running"]:
                    continue
                valid, reason = _generation_is_valid(plans[target])
                del active_generation[device]
                if valid:
                    generation_complete.append(target)
                    # The final resume pass writes the aggregate score-summary.json.
                    # It has no pending items when every route job has succeeded.
                    judge_pending.append((target, None))
                else:
                    failed[target] = reason
                persist("RUNNING", f"generation {target}: {reason}")
            for (target, route), process in list(active_judges.items()):
                if process.poll() is None:
                    continue
                del active_judges[(target, route)]
                valid, reason = _judge_is_complete(plans[target], route)
                if valid and route is None:
                    judge_complete.append(target)
                elif not valid:
                    failed[target] = f"judge {route or 'final'}: {reason}"
                persist("RUNNING", f"judge {target}/{route or 'final'}: {reason}")

        if stopped:
            for target in active_generation.values(): stop_worker(config, config_path, target)
            for process in active_judges.values(): process.terminate()
            persist("STOPPED", "stop signal received; active generation and judge workers were terminated")
        elif failed:
            persist("FAILED", "one or more target stages failed; no aggregate report was produced")
        elif set(judge_complete) == set(targets):
            persist("COMPLETE", "all generation and per-target judge stages completed")
        else:
            persist("FAILED", "pipeline exited before every target was judged")
        return _read_json(state_path)
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
