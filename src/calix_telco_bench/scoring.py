"""Strict aggregation of externally generated, five-dimension rubric scores."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .contracts import identity

DIMENSIONS = ("factual", "completeness", "terminology", "hallucinations", "usefulness")

def aggregate(generations_path: Path, judgments_path: Path, routes: set[str] | None = None) -> dict[str, Any]:
    generations = [json.loads(line) for line in generations_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    generated = {identity(item["route"], item["sample_id"]): item for item in generations if item.get("status") == "ok" and (routes is None or item["route"] in routes)}
    grouped: dict[str, list[float]] = {}; seen: set[str] = set()
    for line in judgments_path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        score = json.loads(line)
        if routes is not None and score["route"] not in routes: continue
        key = identity(score["route"], score["sample_id"])
        if key not in generated or key in seen: raise ValueError(f"unknown or duplicate judgment: {key}")
        values = [score.get(name) for name in DIMENSIONS]
        if any(not isinstance(value, int) or not 0 <= value <= 5 for value in values): raise ValueError(f"invalid rubric dimensions: {key}")
        total = sum(values)
        if score.get("total", total) != total: raise ValueError(f"inconsistent total: {key}")
        grouped.setdefault(score["route"], []).append(float(total)); seen.add(key)
    routes = {route: {"scored": len(values), "mean_total_25": sum(values)/len(values)} for route, values in sorted(grouped.items())}
    values = [v for group in grouped.values() for v in group]
    return {"schema_version": 1, "generated_ok": len(generated), "scored": len(seen), "unscored_generated": len(generated)-len(seen), "macro_route_mean_total_25": sum(v["mean_total_25"] for v in routes.values())/len(routes) if routes else None, "micro_mean_total_25": sum(values)/len(values) if values else None, "routes": routes}
