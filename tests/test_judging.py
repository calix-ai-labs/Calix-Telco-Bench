from pathlib import Path

import pytest

from network_llm_eval.contracts import load_manifest
from network_llm_eval.judging import (
    JudgeError,
    _extract_json,
    _prompt,
    _load_rubric_contexts,
    _validate_vector,
)
from network_llm_eval.rubrics import (
    load_rubrics,
    missing_rubric,
    render_rubric,
    resolve_rubric,
)
from network_llm_eval.scoring import aggregate


def test_judge_json_and_rubric_vector_are_validated() -> None:
    value = _extract_json('```json\n[{"factual": 5, "completeness": 4, "terminology": 3, "hallucinations": 2, "usefulness": 1}]\n```')
    assert _validate_vector(value[0]) == {
        "factual": 5, "completeness": 4, "terminology": 3,
        "hallucinations": 2, "usefulness": 1,
    }


def test_judge_vector_rejects_invalid_dimensions() -> None:
    with pytest.raises(JudgeError, match="invalid"):
        _validate_vector({"factual": 6, "completeness": 4, "terminology": 3, "hallucinations": 2, "usefulness": 1})


ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "datasets" / "MANIFEST.json"


def test_rubric_config_covers_every_manifest_route() -> None:
    rubrics, default = load_rubrics(ROOT / "prompts" / "rubric_config.yaml")
    routes = [r["route"] for r in load_manifest(MANIFEST)["routes"]]
    for route in routes:
        assert route in rubrics, f"missing rubric scope for {route}"
    assert default in rubrics
    assert missing_rubric(rubrics, routes) == []


def test_rubric_config_is_scope_not_reference() -> None:
    """Route entries name scope (behavior + terminology), never a single RFC.

    The redesign deliberately avoids citing a single reference document because
    no single RFC fully defines a protocol; the config must only describe the
    range of correct behavior and terminology.
    """
    rubrics, _ = load_rubrics(ROOT / "prompts" / "rubric_config.yaml")
    blob = "\n".join(
        f"{r} {c.get('factual', '')} {c.get('terminology', '')}"
        for r, c in rubrics.items()
    )
    assert "rfc" not in blob.lower(), "rubric scope must not cite a single RFC"
    assert "all " in blob.lower()


def test_render_rubric_is_generic_and_route_aware() -> None:
    generic = render_rubric({"factual": "", "terminology": ""})
    assert "factual" in generic.lower() and "5:" in generic  # anchor lines present
    assert "{{FACTUAL_DOMAIN}}" not in generic  # no unsubstituted placeholders

    bgp_context = {"factual": "BGP facts", "terminology": "BGP terms"}
    rendered = render_rubric(bgp_context)
    assert "BGP facts" in rendered and "BGP terms" in rendered
    assert "no {{" not in rendered  # both placeholders substituted


def test_resolve_rubric_returns_none_for_unknown_route() -> None:
    rubrics, _ = load_rubrics(ROOT / "prompts" / "rubric_config.yaml")
    assert resolve_rubric(rubrics, "ospf") is not None
    assert resolve_rubric(rubrics, "no_such_route") is None


def test_missing_rubric_reports_unknown_and_empty_routes() -> None:
    rubrics, _ = load_rubrics(ROOT / "prompts" / "rubric_config.yaml")
    unknown = missing_rubric(rubrics, ["ospf", "ghost_route"])
    assert unknown == ["ghost_route"]
    empty = {"empty": {"factual": "", "terminology": ""}}
    assert missing_rubric(empty, ["empty"]) == ["empty"]


def test_prompt_injects_route_specific_rubric() -> None:
    rubrics, _ = load_rubrics(ROOT / "prompts" / "rubric_config.yaml")
    items = [{"route": "ospf", "prompt": "what is a neighborhood?", "answer": "ospf route"}]
    prompt = _prompt("PROTOCOL", items, rubrics)
    assert "ospf" in prompt
    assert "PROTOCOL" in prompt
    assert "what is a neighborhood?" in prompt


def test_prompt_refuses_when_a_route_has_no_rubric() -> None:
    items = [{"route": "ghost_route", "prompt": "q", "answer": "a"}]
    with pytest.raises(JudgeError, match="no rubric"):
        _prompt("PROTOCOL", items, {})


def test_route_scoped_aggregate_ignores_other_in_progress_routes(tmp_path: Path) -> None:
    generations = tmp_path / "generations.jsonl"
    judgments = tmp_path / "judgments.jsonl"
    generations.write_text(
        '{"route":"bgp","sample_id":"1","status":"ok"}\n'
        '{"route":"ospf","sample_id":"1","status":"ok"}\n', encoding="utf-8"
    )
    judgments.write_text(
        '{"route":"bgp","sample_id":"1","factual":5,"completeness":5,"terminology":5,"hallucinations":5,"usefulness":5,"total":25}\n',
        encoding="utf-8",
    )
    result = aggregate(generations, judgments, {"bgp"})
    assert result["generated_ok"] == result["scored"] == 1
    assert result["unscored_generated"] == 0
    assert result["routes"]["bgp"]["mean_total_25"] == 25
