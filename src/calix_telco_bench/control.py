"""Local lifecycle control for config-driven generation workers.

The controller deliberately manages only processes started by this package.  It
uses a PID record in the target's evidence directory so that start, status, and
stop work across terminals without requiring a service manager or MOPD.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import verify_target_artifacts
from .plan import resolve_target


_CONTROL_FILE = ".calix-telco-bench-control.json"
_JUDGE_ALL_CONTROL_FILE = ".calix-telco-bench-judge-all-control.json"
_PIPELINE_CONTROL_FILE = ".calix-telco-bench-pipeline-control.json"


def _control_path(output: Path) -> Path:
    return output / _CONTROL_FILE


def _read_control(output: Path) -> dict[str, Any] | None:
    return _read_control_file(_control_path(output))


def _read_control_file(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _judge_all_root(config: dict[str, Any], config_path: Path) -> Path:
    return resolve_target(config, config_path, next(iter(config["targets"])))["output"].parent


def _process_start_ticks(pid: int) -> str | None:
    """Linux process start time, used to prevent PID-reuse false positives."""
    try:
        # ``comm`` may contain spaces/parentheses, so split after its final ')'.
        remainder = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(") ", 1)[1].split()
        return remainder[19]  # field 22 (starttime); remainder starts at field 3.
    except (FileNotFoundError, IndexError, OSError):
        return None


def _managed_status(control: dict[str, Any] | None) -> tuple[bool, str]:
    """Check that the live process is the process this evaluator launched."""
    if not control:
        return False, "no control record"
    pid = control.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False, "invalid PID"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False, "process exited"
    except PermissionError:
        return False, "cannot verify managed process identity"
    started = _process_start_ticks(pid)
    if started is None:
        return False, "process exited"
    expected_started = control.get("process_start_ticks")
    if isinstance(expected_started, str) and started != expected_started:
        return False, "PID was reused"
    try:
        command_line = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", errors="replace").replace("\0", " ")
    except OSError:
        return False, "cannot read managed process command"
    command = control.get("command")
    if isinstance(command, list) and command and not all(str(part) in command_line for part in command):
        return False, "PID command does not match control record"
    return True, "managed process is running"


def pipeline_status(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Return status for the managed end-to-end coordinator."""
    root = _judge_all_root(config, config_path)
    control = _read_control_file(root / _PIPELINE_CONTROL_FILE)
    pid = control.get("pid") if control else None
    running, reason = _managed_status(control)
    return {
        "output": str(root), "log": str(root / "pipeline-all.log"), "pid": pid,
        "running": running, "reason": reason, "control": control,
    }


def start_pipeline(
    config: dict[str, Any], config_path: Path, *, targets: list[str],
    limit_per_route: int | None, resume: bool,
) -> dict[str, Any]:
    """Start one coordinator which dispatches at most one model to each free GPU."""
    current = pipeline_status(config, config_path)
    if current["running"]:
        raise RuntimeError(f"end-to-end pipeline is already running as PID {current['pid']}")
    active_targets = [target for target in targets if status(config, config_path, target)["running"]]
    if active_targets:
        raise RuntimeError(f"refusing fresh pipeline while target worker(s) are running: {', '.join(active_targets)}")
    # Verify every immutable input before clearing evidence or launching the
    # coordinator. The digest cache avoids rehashing the shared base per target.
    for target in targets:
        verify_target_artifacts(config, config_path, target)
    root = Path(current["output"])
    root.mkdir(parents=True, exist_ok=True)
    if not resume:
        # A plain start-all is deliberately a *fresh* frozen evaluation.  Only
        # configured target directories are removed; root-level evidence such
        # as the adapter-effect check is outside this scope and is retained.
        for target in targets:
            output = resolve_target(config, config_path, target)["output"]
            if output.exists():
                shutil.rmtree(output)
        for filename in (
            _PIPELINE_CONTROL_FILE, _JUDGE_ALL_CONTROL_FILE,
            "pipeline-state.json", "pipeline-all.log", "judge-all.log",
        ):
            path = root / filename
            if path.exists():
                path.unlink()
    command = [
        sys.executable, "-m", "calix_telco_bench.cli", "pipeline-run",
        "--config", str(config_path), "--targets", ",".join(targets),
    ]
    if limit_per_route is not None:
        command.extend(["--limit-per-route", str(limit_per_route)])
    if resume:
        command.append("--resume")
    log_path = root / "pipeline-all.log"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, cwd=config_path.parent,
        )
    control = {
        "schema_version": 1, "kind": "end_to_end_pipeline", "pid": process.pid,
        "process_start_ticks": _process_start_ticks(process.pid),
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "command": command,
        "log": str(log_path), "targets": targets,
    }
    (root / _PIPELINE_CONTROL_FILE).write_text(
        json.dumps(control, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "pipeline-state.json").write_text(json.dumps({
        "schema_version": 1, "status": "STARTING", "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "targets": targets, "queues": {}, "active": {}, "generation_complete": [], "generation_failed": {},
        "message": "fresh target evidence was cleared; coordinator is starting",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return pipeline_status(config, config_path)


def stop_pipeline(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    current = pipeline_status(config, config_path)
    pid = current["pid"]
    coordinator_stopped = False
    coordinator_reason: str | None = None
    if current["running"]:
        assert isinstance(pid, int)
        try:
            os.killpg(pid, signal.SIGTERM)
            coordinator_stopped = True
        except ProcessLookupError:
            coordinator_reason = "pipeline exited before stop signal"
    else:
        coordinator_reason = "no running managed end-to-end pipeline"

    # Generation workers and the judge have their own process sessions, so
    # terminate them explicitly as well.  Do this even when the coordinator
    # has already died or cannot be verified: otherwise a stale coordinator
    # record prevents ``stop-all`` from cleaning up its independently managed
    # children.
    worker_results: dict[str, dict[str, Any]] = {}
    for target in config["targets"]:
        worker_results[target] = stop(config, config_path, target)
    judge_result = stop_judge_all(config, config_path)
    # The dynamic pipeline starts per-target judge processes (rather than the
    # legacy judge-all worker), each in its own process session.
    root = _judge_all_root(config, config_path)
    state = _read_control_file(root / "pipeline-state.json") or {}
    active_judges = state.get("active_judges", {})
    if isinstance(active_judges, dict):
        for judge_pid in active_judges.values():
            if isinstance(judge_pid, int) and judge_pid > 0:
                try:
                    os.killpg(judge_pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
    children_stopped = any(result.get("stopped") for result in worker_results.values()) or bool(judge_result.get("stopped"))
    if coordinator_stopped or children_stopped:
        return {
            **current, "stopped": True, "signal": "SIGTERM",
            "workers": worker_results, "judge": judge_result,
        }
    return {
        **current, "stopped": False, "reason": coordinator_reason,
        "workers": worker_results, "judge": judge_result,
    }


def judge_all_status(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    root = _judge_all_root(config, config_path)
    control = _read_control_file(root / _JUDGE_ALL_CONTROL_FILE)
    pid = control.get("pid") if control else None
    running, reason = _managed_status(control)
    return {"output": str(root), "log": str(root / "judge-all.log"), "pid": pid, "running": running, "reason": reason, "control": control}


def start_judge_all(
    config: dict[str, Any], config_path: Path, *, resume: bool, batch_size: int,
    retries: int, parallelism: int,
) -> dict[str, Any]:
    current = judge_all_status(config, config_path)
    if current["running"]:
        raise RuntimeError(f"judge-all is already running as PID {current['pid']}")
    root = Path(current["output"]); root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "calix_telco_bench.cli", "judge-all", "--config", str(config_path),
        "--batch-size", str(batch_size), "--retries", str(retries), "--parallelism", str(parallelism),
    ]
    if resume:
        command.append("--resume")
    log_path = root / "judge-all.log"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=config_path.parent)
    control = {
        "schema_version": 1, "pid": process.pid, "process_start_ticks": _process_start_ticks(process.pid),
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "command": command, "log": str(log_path),
    }
    (root / _JUDGE_ALL_CONTROL_FILE).write_text(json.dumps(control, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return judge_all_status(config, config_path)


def stop_judge_all(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    current = judge_all_status(config, config_path)
    pid = current["pid"]
    if not current["running"]:
        return {**current, "stopped": False, "reason": "no running managed judge-all worker"}
    assert isinstance(pid, int)
    os.killpg(pid, signal.SIGTERM)
    return {**current, "stopped": True, "signal": "SIGTERM"}


def status(config: dict[str, Any], config_path: Path, target: str) -> dict[str, Any]:
    plan = resolve_target(config, config_path, target)
    control = _read_control(plan["output"])
    pid = control.get("pid") if control else None
    running, reason = _managed_status(control)
    return {
        "target": target,
        "output": str(plan["output"]),
        "log": str(plan["output"] / "worker.log"),
        "pid": pid,
        "running": running,
        "reason": reason,
        "control": control,
    }


def start(
    config: dict[str, Any], config_path: Path, target: str, *,
    limit_per_route: int | None, resume: bool, device: str | None = None,
) -> dict[str, Any]:
    plan = resolve_target(config, config_path, target)
    verify_target_artifacts(config, config_path, target)
    current = status(config, config_path, target)
    if current["running"]:
        raise RuntimeError(f"target {target!r} is already running as PID {current['pid']}")
    output = plan["output"]
    generations = output / "generations.jsonl"
    if generations.exists() and not resume:
        raise FileExistsError(f"{generations} already exists; use --resume or choose a new --output-suffix")
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "calix_telco_bench.cli", "run-config",
        "--config", str(config_path), "--target", target,
    ]
    if device is not None:
        command.extend(["--device", device])
    if limit_per_route is not None:
        command.extend(["--limit-per-route", str(limit_per_route)])
    if resume:
        command.append("--resume")
    log_path = output / "worker.log"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=config_path.parent,
        )
    control = {
        "schema_version": 1, "process_start_ticks": _process_start_ticks(process.pid),
        "target": target,
        "assigned_device": device,
        "pid": process.pid,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "log": str(log_path),
    }
    _control_path(output).write_text(json.dumps(control, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return status(config, config_path, target)


def stop(config: dict[str, Any], config_path: Path, target: str) -> dict[str, Any]:
    current = status(config, config_path, target)
    pid = current["pid"]
    if not current["running"]:
        return {**current, "stopped": False, "reason": "no running managed worker"}
    assert isinstance(pid, int)
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return {**current, "stopped": False, "reason": "worker exited before stop signal"}
    return {**current, "stopped": True, "signal": "SIGTERM"}
