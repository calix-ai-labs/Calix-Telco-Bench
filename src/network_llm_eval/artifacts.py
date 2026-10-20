"""Immutable artifact verification for managed evaluation launches."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .contracts import sha256


_CACHE_FILE = ".telcobench-digest-cache.json"
_CACHE_SCHEMA_VERSION = 1


def _cache_path(config_path: Path) -> Path:
    return config_path.parent / _CACHE_FILE


def _fingerprint(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _read_cache(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    if value.get("schema_version") != _CACHE_SCHEMA_VERSION or not isinstance(value.get("artifacts"), dict):
        return {}
    return value["artifacts"]


def _write_cache(path: Path, artifacts: dict[str, Any]) -> None:
    """Write cache atomically when possible; cache failures never bypass checks."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps({
            "schema_version": _CACHE_SCHEMA_VERSION,
            "artifacts": artifacts,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass


def _verify(artifact: Path, expected: object, cache: dict[str, Any]) -> None:
    if not isinstance(expected, str) or len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected.lower()):
        raise ValueError(f"{artifact}: expected SHA-256 must be a 64-character hexadecimal digest")
    if not artifact.is_file():
        raise FileNotFoundError(f"required artifact does not exist: {artifact}")
    resolved = str(artifact.resolve())
    fingerprint = _fingerprint(artifact)
    cached = cache.get(resolved)
    if isinstance(cached, dict) and cached.get("expected_sha256") == expected and cached.get("fingerprint") == fingerprint:
        return
    actual = sha256(artifact)
    if actual != expected:
        raise ValueError(f"artifact SHA-256 mismatch for {artifact}: expected {expected}, got {actual}")
    cache[resolved] = {"expected_sha256": expected, "fingerprint": fingerprint}


def _verify_base(base: dict[str, Any], cache: dict[str, Any]) -> None:
    """Verify either a consolidated model or an explicitly pinned shard set."""
    root = Path(base["path"])
    shards = base.get("model_weight_files")
    if shards is None:
        _verify(root / "model.safetensors", base.get("model_safetensors_sha256"), cache)
        return
    if not isinstance(shards, list) or not shards:
        raise ValueError("release.cpt_base.model_weight_files must be a non-empty list")
    for shard in shards:
        if not isinstance(shard, dict) or not isinstance(shard.get("path"), str):
            raise ValueError("each release.cpt_base.model_weight_files entry requires path and sha256")
        _verify(root / shard["path"], shard.get("sha256"), cache)


def verify_target_artifacts(config: dict[str, Any], config_path: Path, target_name: str) -> None:
    """Verify the configured base plus one target adapter before launching it.

    Cache keys use resolved path, filesystem device/inode, size, and nanosecond
    mtime. An unchanged 30-GB model is not rehashed for later targets or
    resumes, while replacing or modifying an artifact invalidates the cache.
    """
    target = config["targets"].get(target_name)
    if not isinstance(target, dict):
        raise ValueError(f"unknown evaluation target: {target_name}")
    provider = config.get("provider")
    if isinstance(provider, dict) and provider.get("kind") == "openai_compatible":
        # Endpoint models have no local artifact to hash. The exact endpoint
        # and model ID are persisted in the run metadata instead.
        return
    base_name = target.get("base_model", "cpt_base")
    if not isinstance(base_name, str) or not base_name:
        raise ValueError(f"target {target_name!r} base_model must be a non-empty release model name")
    base = config["release"].get(base_name)
    if not isinstance(base, dict) or not isinstance(base.get("path"), str):
        raise ValueError(f"release.{base_name}.path is required for target {target_name!r}")
    cache_path = _cache_path(config_path)
    cache = _read_cache(cache_path)
    try:
        _verify_base(base, cache)
        adapter = target.get("adapter")
        if adapter is not None:
            _verify(Path(str(adapter)) / "adapter_model.safetensors", target.get("adapter_revision"), cache)
    finally:
        # Retain any successful base verification even if a later adapter
        # check fails, so correcting that adapter does not rehash the base.
        _write_cache(cache_path, cache)
