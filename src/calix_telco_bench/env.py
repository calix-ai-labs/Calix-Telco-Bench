"""Local, untracked runtime configuration for evaluator integrations."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def load_runtime_env(env_file: Path | None = None) -> Path | None:
    """Load an untracked `.env` without overriding exported variables."""
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("install calix-telco-bench to load .env configuration") from exc
    resolved = (env_file or Path.cwd() / ".env").resolve()
    if resolved.is_file():
        load_dotenv(resolved, override=False)
        return resolved
    return None


def load_judge_env(env_file: Path | None = None) -> dict[str, str | None]:
    """Load judge metadata from `.env` without overriding exported variables.

    The utility intentionally returns identity metadata only. Provider API keys
    remain in the process environment and are never copied to evaluation
    provenance, YAML configuration, logs, or the terminal monitor.
    """
    resolved = load_runtime_env(env_file)
    return {
        "provider": os.getenv("CALIX_TELCO_BENCH_JUDGE_PROVIDER") or None,
        "model": os.getenv("CALIX_TELCO_BENCH_JUDGE_MODEL") or None,
        "model_revision": os.getenv("CALIX_TELCO_BENCH_JUDGE_MODEL_REVISION") or None,
        "env_file": str(resolved) if resolved is not None else None,
    }
