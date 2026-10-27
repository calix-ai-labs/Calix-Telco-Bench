"""Resumable LLM-as-judge execution for frozen generation evidence."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import identity, sha256
from .env import load_judge_env
from .rubrics import load_rubrics, missing_rubric, render_rubric, resolve_rubric
from .scoring import DIMENSIONS, aggregate


class JudgeError(RuntimeError):
    """The configured judge cannot safely produce a rubric record."""


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_rubric_contexts(
    rubric_path: Path | None, pending: list[dict[str, Any]]
) -> dict[str, dict[str, str]]:
    """Load per-route rubric context maps for the pending items.

    Each entry is ``{"factual": ..., "terminology": ...}`` and is passed
    un-rendered to ``missing_rubric`` (preflight) and ``resolve_rubric`` (the
    prompt), so no pre-rendering happens here. A route absent from the config is
    simply absent from the returned map; the preflight treats that as a missing
    rubric and refusing to judge. ``rubric_path`` is optional so the judge
    works without a config.
    """
    contexts: dict[str, dict[str, str]] = {}
    if rubric_path is not None and rubric_path.is_file():
        try:
            rubrics, _default = load_rubrics(rubric_path)
        except RuntimeError as exc:
            raise JudgeError(f"cannot load rubric config: {exc}") from exc
        for row in pending:
            route = row["route"]
            if route in rubrics:
                contexts.setdefault(route, rubrics[route])
    return contexts


def _extract_json(text: str) -> Any:
    value = text.strip()
    if "```json" in value:
        value = value.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in value:
        value = value.split("```", 1)[1].split("```", 1)[0]
    return json.loads(value.strip())


def _validate_vector(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        raise JudgeError("judge response item is not an object")
    result: dict[str, int] = {}
    for name in DIMENSIONS:
        score = value.get(name)
        if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 5:
            raise JudgeError(f"invalid {name!r} score: {score!r}")
        result[name] = score
    return result


def _prompt(
    protocol: str,
    items: list[dict[str, Any]],
    rubric_contexts: dict[str, str],
) -> str:
    rendered = []
    for number, item in enumerate(items, start=1):
        route = item["route"]
        rubric = resolve_rubric(rubric_contexts, route)
        if rubric is None:
            raise JudgeError(
                f"no rubric defined for route {route!r}; judging is refused"
            )
        rendered.append(
            f"[ITEM {number}]\nROUTE: {route}\n{rubric}\n\n"
            f"QUESTION:\n{item['prompt']}\n\n"
            f"MODEL ANSWER:\n{item['answer']}"
        )
    rendered_items = "\n\n---\n\n".join(rendered)
    return f"""{protocol}

You are judging generated answers for a networking-language-model benchmark.
Each item carries its own route-specific rubric above; use it verbatim. Apply
the five rubric dimensions independently as integers from 0 through 5. Return
ONLY a JSON array, in the same order as the items. Each object must contain
exactly these integer keys: factual, completeness, terminology, hallucinations,
usefulness. Do not include prose, Markdown, totals, IDs, or extra keys.

{rendered_items}
"""


def _anthropic_text(*, api_key: str, model: str, prompt: str) -> str:
    try:
        from anthropic import Anthropic
        import httpx2
    except ImportError as exc:
        raise JudgeError("Anthropic support is not installed; run `uv sync --extra judge`") from exc
    options: dict[str, Any] = {"api_key": api_key}
    if base_url := os.getenv("ANTHROPIC_BASE_URL"):
        options["base_url"] = base_url
    # The organization-hosted Anthropic-compatible endpoint uses a private
    # certificate. This is the same explicit transport contract used by MOPD.
    http_client = httpx2.Client(verify=False)
    try:
        # Keep this call contract byte-for-byte aligned in shape with the
        # proven MOPD evaluator transport.
        response = Anthropic(**options, http_client=http_client, timeout=60.0).messages.create(
            model=model,
            max_tokens=2000,
            # The installed Anthropic SDK carries provider-specific generation
            # controls through extra_body, rather than accepting temperature as
            # a direct Messages.create keyword.
            extra_body={"temperature": 0.0},
            timeout=60.0,
            messages=[{"role": "user", "content": prompt}],
        )
    finally:
        http_client.close()
    if not response.content or not hasattr(response.content[0], "text"):
        raise JudgeError("judge returned no text content")
    return str(response.content[0].text)


def _judge_batch(
    *, provider: str, model: str, api_key: str, protocol: str, items: list[dict[str, Any]],
    rubric_contexts: dict[str, str],
) -> tuple[list[dict[str, int]], str]:
    if provider != "anthropic":
        raise JudgeError(f"unsupported judge provider {provider!r}; only 'anthropic' is configured")
    raw = _anthropic_text(api_key=api_key, model=model, prompt=_prompt(protocol, items, rubric_contexts))
    parsed = _extract_json(raw)
    if not isinstance(parsed, list) or len(parsed) != len(items):
        raise JudgeError(f"judge returned {len(parsed) if isinstance(parsed, list) else 0} scores for {len(items)} items")
    return [_validate_vector(value) for value in parsed], raw


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def judge(
    *, generations: Path, output: Path, protocol_path: Path, rubric_path: Path | None,
    env_file: Path | None,
    batch_size: int, retries: int, resume: bool, allow_incomplete: bool,
    max_items: int | None, dry_run: bool, routes: list[str] | None = None,
) -> dict[str, Any]:
    if batch_size < 1 or retries < 1:
        raise ValueError("batch_size and retries must be positive")
    if not generations.is_file() or not protocol_path.is_file():
        raise FileNotFoundError("generations and judge protocol must exist")
    selected_routes = set(routes) if routes else None
    all_rows = [row for row in _rows(generations) if selected_routes is None or row.get("route") in selected_routes]
    successful = [row for row in all_rows if row.get("status") == "ok"]
    errors = [row for row in all_rows if row.get("status") == "error"]
    coverage_path = generations.with_name("coverage.json")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8")) if coverage_path.is_file() else {}
    if not allow_incomplete and (coverage.get("missing", 0) or errors):
        raise JudgeError("generation is incomplete or contains errors; wait for a clean frozen result or pass --allow-incomplete")
    metadata = load_judge_env(env_file)
    provider, model = metadata["provider"], metadata["model"]
    if not provider or not model:
        raise JudgeError("CALIX_TELCO_BENCH_JUDGE_PROVIDER and CALIX_TELCO_BENCH_JUDGE_MODEL are required in .env")
    api_key = os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key and not dry_run:
        raise JudgeError("ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY is required in .env")
    output.mkdir(parents=True, exist_ok=True)
    suffix = "--" + "-".join(sorted(selected_routes)) if selected_routes else ""
    run_path = output / f"judge-run{suffix}.json"
    summary_path = output / f"score-summary{suffix}.json"
    judgments_path = output / "judgments.jsonl"
    raw_path = output / "judge-raw.jsonl"
    failures_path = output / "judge-failures.jsonl"
    if judgments_path.exists() and not resume:
        raise FileExistsError(f"{judgments_path} exists; use --resume")
    done = {identity(row["route"], row["sample_id"]) for row in _rows(judgments_path)} if judgments_path.exists() else set()
    pending = [row for row in successful if identity(row["route"], row["sample_id"]) not in done]
    if max_items is not None:
        pending = pending[:max_items]
    rubric_contexts = _load_rubric_contexts(rubric_path, pending)
    missing = missing_rubric(rubric_contexts, {row["route"] for row in pending})
    if missing:
        raise JudgeError(
            "rubric defined for no pending route; refusing to judge: "
            + ", ".join(sorted(missing))
        )
    plan = {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "generations": str(generations), "generations_sha256": sha256(generations),
        "judge_protocol": str(protocol_path), "judge_protocol_sha256": sha256(protocol_path),
        "provider": provider, "model": model, "model_revision": metadata["model_revision"],
        "temperature": 0.0, "batch_size": batch_size, "retries": retries,
        "generated_ok": len(successful), "generated_errors": len(errors),
        "pending_at_start": len(pending), "scored_at_start": len(done), "scored": len(done),
        "remaining": len(pending), "completed_batches": 0, "failed_batches_this_run": 0,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(), "allow_incomplete": allow_incomplete,
        "dry_run": dry_run, "selected_routes": sorted(selected_routes) if selected_routes else None,
        "status": "DRY_RUN" if dry_run else "RUNNING",
    }
    _write_json(run_path, plan)
    if dry_run:
        return {**plan, "dry_run": True, "judgments_path": str(judgments_path)}
    protocol = protocol_path.read_text(encoding="utf-8")
    with judgments_path.open("a", encoding="utf-8") as judgments, raw_path.open("a", encoding="utf-8") as raw_log, failures_path.open("a", encoding="utf-8") as failures:
        for offset in range(0, len(pending), batch_size):
            items = pending[offset:offset + batch_size]
            key_rows = [{"route": row["route"], "sample_id": row["sample_id"]} for row in items]
            plan["current_batch"] = {"route": items[0]["route"], "size": len(items), "first_sample_id": items[0]["sample_id"]}
            plan["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            _write_json(run_path, plan)
            last_error: str | None = None
            for attempt in range(1, retries + 1):
                try:
                    vectors, raw = _judge_batch(provider=str(provider), model=str(model), api_key=str(api_key), protocol=protocol, items=items, rubric_contexts=rubric_contexts)
                    raw_log.write(json.dumps({"items": key_rows, "attempt": attempt, "response": raw}, ensure_ascii=False) + "\n"); raw_log.flush()
                    for item, vector in zip(items, vectors, strict=True):
                        result = {"route": item["route"], "sample_id": item["sample_id"], **vector, "total": sum(vector.values())}
                        judgments.write(json.dumps(result, sort_keys=True) + "\n")
                    judgments.flush()
                    plan["scored"] = int(plan["scored"]) + len(items)
                    plan["remaining"] = max(0, len(successful) - int(plan["scored"]))
                    plan["completed_batches"] = int(plan["completed_batches"]) + 1
                    plan["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                    _write_json(run_path, plan)
                    last_error = None
                    break
                except KeyboardInterrupt:
                    plan["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
                    plan["status"] = "INTERRUPTED"
                    _write_json(run_path, plan)
                    summary = aggregate(generations, judgments_path, selected_routes)
                    summary["judge"] = {key: plan[key] for key in ("provider", "model", "model_revision", "temperature", "batch_size", "retries", "judge_protocol_sha256")}
                    _write_json(summary_path, summary)
                    return summary
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    if attempt < retries:
                        time.sleep(float(attempt))
            if last_error:
                failures.write(json.dumps({"items": key_rows, "error": last_error, "attempts": retries}, sort_keys=True) + "\n"); failures.flush()
                plan["failed_batches_this_run"] = int(plan["failed_batches_this_run"]) + 1
                plan["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                _write_json(run_path, plan)
    summary = aggregate(generations, judgments_path, selected_routes)
    summary["judge"] = {key: plan[key] for key in ("provider", "model", "model_revision", "temperature", "batch_size", "retries", "judge_protocol_sha256")}
    _write_json(summary_path, summary)
    plan["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    plan.pop("current_batch", None)
    plan["scored"] = summary["scored"]
    plan["remaining"] = summary["unscored_generated"]
    plan["updated_at_utc"] = plan["finished_at_utc"]
    plan["status"] = "COMPLETED" if summary["unscored_generated"] == 0 else "INCOMPLETE"
    _write_json(run_path, plan)
    return summary
