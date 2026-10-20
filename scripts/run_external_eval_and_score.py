#!/usr/bin/env python3
"""Run external-model generation, then automatically judge complete outputs."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / ".venv" / "bin" / "telcobench"
POLL_SECONDS = 30


def cli_json(*args: str) -> dict[str, Any]:
    result = subprocess.run(
        [str(CLI), *args], cwd=ROOT, capture_output=True, text=True
    )
    if result.returncode:
        raise RuntimeError(
            f"CLI command failed ({result.returncode}): {' '.join(args)}\n"
            f"{result.stdout}{result.stderr}"
        )
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"CLI returned invalid JSON for {' '.join(args)}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"CLI returned unexpected data for {' '.join(args)}")
    return value


def coverage_results(targets: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for target, state in targets.items():
        output = Path(str(state.get("output", "")))
        path = output / "coverage.json"
        try:
            coverage = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failures.append(f"{target}: no readable coverage.json")
            continue
        if (
            coverage.get("missing") != 0
            or coverage.get("errors") != 0
            or coverage.get("unique_records") != coverage.get("expected")
            or coverage.get("expected") != 2850
        ):
            failures.append(f"{target}: incomplete/failed coverage {coverage}")
            continue
        generations_path = output / "generations.jsonl"
        try:
            generations = [
                json.loads(line)
                for line in generations_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError):
            failures.append(f"{target}: no readable generations.jsonl")
            continue
        invalid = [
            row for row in generations
            if row.get("status") != "ok"
            or not str(row.get("answer", "")).strip()
            or row.get("finish_reason") != "stop"
        ]
        if invalid:
            examples = ", ".join(
                f"{row.get('route')}/{row.get('sample_id')}:{row.get('finish_reason') or 'missing finish_reason'}"
                for row in invalid[:8]
            )
            failures.append(
                f"{target}: {len(invalid)} empty/non-stop answers; examples: {examples}"
            )
    return not failures, failures


def scoring_results(targets: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for target, state in targets.items():
        output = Path(str(state.get("output", "")))
        path = output / "judge-run.json"
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failures.append(f"{target}: no readable judge-run.json")
            continue
        if run.get("status") != "COMPLETED" or run.get("scored") != 2850 or run.get("remaining") != 0:
            failures.append(
                f"{target}: judge status={run.get('status')} scored={run.get('scored')} "
                f"remaining={run.get('remaining')}"
            )
    return not failures, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="path to a local provider configuration; do not commit credentials or internal endpoints",
    )
    parser.add_argument("--parallelism", type=int, default=4)
    args = parser.parse_args()
    config = args.config.resolve()

    print(f"Starting generation from {config}", flush=True)
    started = cli_json("start-all", "--config", str(config))
    print(json.dumps(started, indent=2, sort_keys=True), flush=True)

    # Give the detached workers time to publish their control records.
    time.sleep(10)
    while True:
        state = cli_json("status-all", "--config", str(config))
        targets = state.get("targets", {})
        active = [name for name, value in targets.items() if value.get("running")]
        summary = ", ".join(f"{name}={'running' if name in active else 'stopped'}" for name in targets)
        print(f"Generation: {summary}", flush=True)
        if not active:
            complete, failures = coverage_results(targets)
            if not complete:
                print("Generation did not complete cleanly; judging was not started:", flush=True)
                print("\n".join(failures), flush=True)
                return 1
            break
        time.sleep(POLL_SECONDS)

    print("All four generations have complete, error-free coverage; starting judging.", flush=True)
    judged = cli_json(
        "judge-all-start", "--config", str(config), "--resume",
        "--parallelism", str(args.parallelism),
    )
    print(json.dumps(judged, indent=2, sort_keys=True), flush=True)

    time.sleep(5)
    while True:
        status = cli_json("judge-all-status", "--config", str(config))
        print(
            f"Judging coordinator: {'running' if status.get('running') else status.get('reason')}",
            flush=True,
        )
        targets_state = cli_json("status-all", "--config", str(config)).get("targets", {})
        if not status.get("running"):
            complete, failures = scoring_results(targets_state)
            if not complete:
                print("Judging stopped before all targets completed:", flush=True)
                print("\n".join(failures), flush=True)
                return 1
            print("All four targets are fully scored.", flush=True)
            return 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Coordinator interrupted; managed generation/judging workers may still be running.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"Coordinator error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
