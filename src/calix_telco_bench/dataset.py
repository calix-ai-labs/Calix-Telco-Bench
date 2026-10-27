"""Read benchmark records only after their release manifest validates."""
from __future__ import annotations
import json
from pathlib import Path
from collections.abc import Collection
from typing import Any, Iterator
from .contracts import load_manifest

def iter_records(
    manifest_path: Path,
    limit_per_route: int | None = None,
    routes: Collection[str] | None = None,
) -> Iterator[dict[str, Any]]:
    selected = set(routes) if routes is not None else None
    for entry in load_manifest(manifest_path)["routes"]:
        if selected is not None and entry["route"] not in selected:
            continue
        emitted = 0
        for raw in (manifest_path.parent / entry["path"]).read_text(encoding="utf-8").splitlines():
            if not raw.strip() or (limit_per_route is not None and emitted >= limit_per_route): continue
            record = json.loads(raw); record["route"] = entry["route"]
            yield record; emitted += 1
