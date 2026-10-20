"""Comparison reporting for completed, provenance-compatible judge outputs."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .scoring import aggregate


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_target(name: str, output: Path) -> dict[str, Any]:
    generations, judgments = output / "generations.jsonl", output / "judgments.jsonl"
    if not judgments.is_file():
        raise FileNotFoundError(f"{name}: missing judgments: {judgments}")
    summary = aggregate(generations, judgments)
    if summary["scored"] != summary["generated_ok"] or summary["unscored_generated"]:
        raise ValueError(f"{name}: judge coverage is incomplete ({summary['scored']}/{summary['generated_ok']})")
    recorded = output / "score-summary.json"
    if not recorded.is_file():
        raise FileNotFoundError(f"{name}: missing finalized score summary: {recorded}")
    judge = json.loads(recorded.read_text(encoding="utf-8")).get("judge")
    if not isinstance(judge, dict):
        raise ValueError(f"{name}: score summary lacks judge provenance")
    return {"name": name, "summary": summary, "judge": judge}


def _combine_teachers(parts: list[dict[str, Any]]) -> dict[str, Any]:
    routes: dict[str, dict[str, Any]] = {}
    for part in parts:
        for route, values in part["summary"]["routes"].items():
            if route in routes:
                raise ValueError(f"teacher route overlap: {route}")
            routes[route] = values
    scored = sum(int(item["scored"]) for item in routes.values())
    total = sum(float(item["mean_total_25"]) * int(item["scored"]) for item in routes.values())
    return {
        "generated_ok": sum(int(part["summary"]["generated_ok"]) for part in parts),
        "scored": scored,
        "unscored_generated": sum(int(part["summary"]["unscored_generated"]) for part in parts),
        "micro_mean_total_25": total / scored if scored else None,
        "macro_route_mean_total_25": sum(float(item["mean_total_25"]) for item in routes.values()) / len(routes) if routes else None,
        "routes": dict(sorted(routes.items())),
        "components": [part["name"] for part in parts],
    }


def build_comparison_report(*, outputs: dict[str, Path], output: Path) -> dict[str, Any]:
    required = {"cpt_base", "mopd_student_step_2000"}
    if not required.issubset(outputs):
        raise ValueError("comparison requires cpt_base and mopd_student_step_2000")
    loaded = {name: _load_target(name, path) for name, path in outputs.items()}
    provenance = {json.dumps(value["judge"], sort_keys=True) for value in loaded.values()}
    if len(provenance) != 1:
        raise ValueError("cannot compare targets with different judge provenance")
    teacher_parts = [value for name, value in loaded.items() if name not in required]
    teacher = _combine_teachers(teacher_parts)
    cpt, student = loaded["cpt_base"]["summary"], loaded["mopd_student_step_2000"]["summary"]
    if set(cpt["routes"]) != set(student["routes"]):
        raise ValueError("CPT and MOPD student route coverage differs")
    route_deltas = {
        route: float(student["routes"][route]["mean_total_25"]) - float(cpt["routes"][route]["mean_total_25"])
        for route in cpt["routes"]
    }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "judge": loaded["cpt_base"]["judge"],
        "models": {"cpt_base": cpt, "mopd_student_step_2000": student, "routed_teacher_aggregate": teacher},
        "mopd_minus_cpt": {
            "micro_mean_total_25": float(student["micro_mean_total_25"]) - float(cpt["micro_mean_total_25"]),
            "macro_route_mean_total_25": float(student["macro_route_mean_total_25"]) - float(cpt["macro_route_mean_total_25"]),
            "route_mean_total_25": route_deltas,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "comparison-report.json", report)
    rows = ["# Neutral evaluation comparison", "", "| Model | Judged | Micro /25 | Macro /25 |", "|---|---:|---:|---:|"]
    for name, summary in report["models"].items():
        rows.append(f"| {name} | {summary['scored']} | {summary['micro_mean_total_25']:.4f} | {summary['macro_route_mean_total_25']:.4f} |")
    rows.extend(["", f"MOPD − CPT micro: {report['mopd_minus_cpt']['micro_mean_total_25']:+.4f}/25.", f"MOPD − CPT macro: {report['mopd_minus_cpt']['macro_route_mean_total_25']:+.4f}/25.", ""])
    (output / "comparison-report.md").write_text("\n".join(rows), encoding="utf-8")
    return report
