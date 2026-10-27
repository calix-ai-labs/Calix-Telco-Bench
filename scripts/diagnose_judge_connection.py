#!/usr/bin/env python3
"""Diagnose the configured LLM-as-judge endpoint without exposing credentials.

Run from a cloned ``calix-telco-bench`` checkout using its local virtual
environment.  The script does not import MOPD and never print credentials.  A
request test deliberately uses the same Anthropic/httpx2 call contract as the
evaluator, so it can distinguish a missing endpoint/token from a gateway
concurrency limit.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _env(path: Path) -> dict[str, Any]:
    from dotenv import load_dotenv

    if path.is_file():
        load_dotenv(path, override=False)
    token = os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY")
    return {
        "env_file": str(path),
        "env_file_exists": path.is_file(),
        "base_url": os.getenv("ANTHROPIC_BASE_URL"),
        "credential_variable": (
            "ANTHROPIC_AUTH_TOKEN" if os.getenv("ANTHROPIC_AUTH_TOKEN") else
            "ANTHROPIC_API_KEY" if os.getenv("ANTHROPIC_API_KEY") else None
        ),
        "credential_present": bool(token),
        "credential_length": len(token) if token else 0,
        "judge_provider": os.getenv("CALIX_TELCO_BENCH_JUDGE_PROVIDER"),
        "judge_model": os.getenv("CALIX_TELCO_BENCH_JUDGE_MODEL"),
    }


def _root_error(exc: BaseException) -> str:
    root: BaseException = exc
    while root.__cause__ is not None:
        root = root.__cause__
    text = str(root)
    for name in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"):
        value = os.getenv(name)
        if value:
            text = text.replace(value, "[REDACTED]")
    return f"{type(root).__name__}: {text}"


def _request(model: str) -> dict[str, Any]:
    """One minimal real provider request using the evaluator transport."""
    from anthropic import Anthropic
    import httpx2

    token = os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY")
    if not token:
        raise RuntimeError("no ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY is configured")
    options: dict[str, Any] = {"api_key": token}
    if base_url := os.getenv("ANTHROPIC_BASE_URL"):
        options["base_url"] = base_url
    started = time.monotonic()
    client = httpx2.Client(verify=False)
    try:
        response = Anthropic(**options, http_client=client, timeout=60.0).messages.create(
            model=model,
            max_tokens=32,
            extra_body={"temperature": 0.0},
            timeout=60.0,
            messages=[{"role": "user", "content": "Return exactly: OK"}],
        )
    finally:
        client.close()
    text = "".join(str(getattr(item, "text", "")) for item in response.content)
    return {"ok": True, "elapsed_seconds": round(time.monotonic() - started, 3), "response_characters": len(text)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--requests", type=int, default=0, help="number of real minimal requests; default: configuration-only")
    parser.add_argument("--parallelism", type=int, default=1, help="simultaneous requests for the real-request probe")
    args = parser.parse_args()
    if args.requests < 0 or args.parallelism < 1:
        parser.error("--requests must be non-negative and --parallelism must be positive")

    import anthropic
    import httpx2

    environment = _env(args.env_file.resolve())
    report: dict[str, Any] = {
        "schema_version": 1,
        "python": sys.version.split()[0],
        "anthropic_version": anthropic.__version__,
        "httpx2_version": httpx2.__version__,
        "environment": environment,
        "requested_probes": args.requests,
        "parallelism": args.parallelism,
        "results": [],
    }
    configuration_ok = bool(environment["credential_present"] and environment["base_url"] and environment["judge_model"])
    if args.requests and not configuration_ok:
        report["configuration_ok"] = False
        report["error"] = "real probe requires ANTHROPIC_BASE_URL, a credential, and CALIX_TELCO_BENCH_JUDGE_MODEL"
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2
    report["configuration_ok"] = configuration_ok
    if args.requests:
        model = str(environment["judge_model"])
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.parallelism, args.requests)) as pool:
            futures = [pool.submit(_request, model) for _ in range(args.requests)]
            for index, future in enumerate(futures, start=1):
                try:
                    report["results"].append({"probe": index, **future.result()})
                except Exception as exc:  # report failure details, never throw away evidence
                    report["results"].append({"probe": index, "ok": False, "error": _root_error(exc)})
    report["successful_probes"] = sum(result["ok"] for result in report["results"])
    report["failed_probes"] = len(report["results"]) - report["successful_probes"]
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if configuration_ok and report["failed_probes"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
