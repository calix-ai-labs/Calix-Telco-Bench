"""Immutable benchmark identities and release-manifest validation."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any

REQUIRED_RECORD_FIELDS = ("prompt", "sample_id", "teacher_routing_key")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    """Return the UTF-8 SHA-256 for an exact rendered-text artifact."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not isinstance(value.get("routes"), list):
        raise ValueError("manifest must have schema_version=1 and a routes list")
    return value

def identity(route: str, sample_id: str) -> str:
    """Stable ID: source-local sample IDs are only unique within a route."""
    return f"{route}:{sample_id}"

def validate_manifest(path: Path) -> dict[str, int]:
    manifest, seen, total = load_manifest(path), set(), 0
    routes = manifest["routes"]
    if len({entry.get("route") for entry in routes}) != len(routes):
        raise ValueError("manifest contains duplicate route names")
    for entry in routes:
        route, relative = entry.get("route"), entry.get("path")
        if not isinstance(route, str) or not isinstance(relative, str):
            raise ValueError("each route needs string route and path fields")
        dataset = path.parent / relative
        if not dataset.is_file() or sha256(dataset) != entry.get("sha256"):
            raise ValueError(f"missing or altered dataset: {dataset}")
        count = 0
        for line_number, raw in enumerate(dataset.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip(): continue
            record = json.loads(raw)
            missing = [key for key in REQUIRED_RECORD_FIELDS if not isinstance(record.get(key), str) or not record[key].strip()]
            if missing: raise ValueError(f"{dataset}:{line_number}: missing {', '.join(missing)}")
            if record["teacher_routing_key"] != route: raise ValueError(f"{dataset}:{line_number}: route label differs from manifest")
            key = identity(route, record["sample_id"])
            if key in seen: raise ValueError(f"duplicate benchmark identity: {key}")
            seen.add(key); count += 1
        if count != entry.get("records"): raise ValueError(f"{dataset}: count {count} differs from manifest")
        total += count
    return {"routes": len(routes), "prompts": total}
