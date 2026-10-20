"""Per-route anchored rubric configuration for the judging stage.

The judging stage scores each frozen answer against five rubric dimensions
(``factual``, ``completeness``, ``terminology``, ``hallucinations``,
``usefulness``). Most of those dimensions are fully generic: the 5/3/1/0 scale
anchors, the completeness and usefulness guidance, and the guardrail that
``hallucinations`` only penalizes claims that are factually false. What varies
by route is *domain knowledge* — which standards and behaviors count as "factual",
and which terms count as correct terminology.

Keeping the generic anchors in code (the ``TEMPLATE`` constant below) means the
config file only stores the genuinely route-specific context: a short
``factual`` description (standards, key concepts, sample behaviors) and a
``terminology`` vocabulary list. The judge prompt is rendered at runtime by
injecting each route's context into the template, so adding a route is one short
entry, not a full rubric reauthoring.
"""
from __future__ import annotations

from typing import Iterable

import yaml

DEFAULT_ROUTE_KEY = "default_route"
RUBRICS_KEY = "rubrics"
FACTUAL_KEY = "factual"
TERMINOLOGY_KEY = "terminology"


TEMPLATE = """You are judging answers for a networking-language-model benchmark.
Score the following answers using these five dimensions (each 0-5). ``total`` is
their sum (0-25).

1. Factual Correctness (0-5): Does the answer match the domain in
{{FACTUAL_DOMAIN}}?
   - 5: All facts correct, exact values and mechanisms where applicable.
   - 3: Mostly correct with minor inaccuracies.
   - 1: Multiple factual errors.
   - 0: Fundamentally wrong or fabricated.
2. Completeness (0-5): Does it cover all major aspects?
   - 5: Covers all key points; lists all applicable items.
   - 3: Covers most points but misses some.
   - 1: Incomplete; misses major aspects.
   - 0: Minimal coverage.
3. Terminology (0-5): Uses the correct terminology of {{TERMINOLOGY_DOMAIN}}?
   - 5: Precise terms throughout.
   - 3: Mostly correct terminology with minor confusion.
   - 1: Mix of correct and incorrect terms.
   - 0: Mostly generic or wrong terminology.
4. Hallucinations (0-5): Are there fabricated or factually false claims?
   - 5: All claims factually true, grounded in real-world behavior.
   - 3: Minor inaccuracies that do not fundamentally misrepresent the topic.
   - 1: Some fabricated concepts, made-up values, or nonexistent features.
   - 0: Significant hallucinations.
   **Important**: This dimension ONLY penalizes claims that are FACTUALLY
   FALSE. Incomplete / vague / missing context -> penalize in Completeness or
   Terminology. Only deduct for a statement that is technically nonexistent,
   fabricated, or directly contradicts the topic.
5. Usefulness (0-5): Would a network engineer find this actionable?
   - 5: Clear, with examples; useful for troubleshooting/config.
   - 3: Somewhat useful but vague or missing context.
   - 1: Technically correct but not actionable.
   - 0: Misleading or would cause problems if followed.
"""


def load_rubrics(path) -> tuple[dict[str, dict[str, str]], str]:
    """Load rubric config.

    Returns ``(rubrics, default_route)`` where ``rubrics`` maps each route to a
    ``{"factual": ..., "terminology": ...}`` context dict. Raises
    ``RuntimeError`` when the file is missing, unparseable, or shaped in a way
    the judging stage cannot rely on.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise RuntimeError(f"rubric config not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise RuntimeError(f"rubric config could not be parsed: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("rubric config must be a mapping of keys to values")
    rubrics = data.get(RUBRICS_KEY)
    if not isinstance(rubrics, dict):
        raise RuntimeError("rubric config has no rubrics mapping")
    cleaned: dict[str, dict[str, str]] = {}
    for key, value in rubrics.items():
        if not key or not isinstance(value, dict):
            continue
        cleaned[str(key)] = {
            FACTUAL_KEY: str(value.get(FACTUAL_KEY, "") or "").strip(),
            TERMINOLOGY_KEY: str(value.get(TERMINOLOGY_KEY, "") or "").strip(),
        }
    if not cleaned:
        raise RuntimeError("rubric config has no non-empty rubric entries")
    default_route = str(data.get(DEFAULT_ROUTE_KEY) or "")
    return cleaned, default_route


def render_rubric(context: dict[str, str]) -> str:
    """Render the static template with a route's context injected."""
    return (
        TEMPLATE.replace("{{FACTUAL_DOMAIN}}", context.get(FACTUAL_KEY, ""))
        .replace("{{TERMINOLOGY_DOMAIN}}", context.get(TERMINOLOGY_KEY, ""))
    )


def resolve_rubric(rubrics: dict[str, dict[str, str]], route: str) -> str | None:
    """Return the rendered rubric for ``route``, or ``None`` when it is absent."""
    context = rubrics.get(route)
    if not context:
        return None
    return render_rubric(context)


def missing_rubric(rubrics: dict[str, dict[str, str]], routes: Iterable[str]) -> list[str]:
    """Return active ``route`` values with no usable rubric context.

    A route counts as missing when it has no entry, or its entry is empty on
    both ``factual`` and ``terminology`` (an empty entry would render a
    generic-only rubric, which is not a real route-defined rubric).
    """
    missing = []
    for route in routes:
        if not route:
            continue
        context = rubrics.get(route)
        if not context or not (context.get(FACTUAL_KEY) or context.get(TERMINOLOGY_KEY)):
            missing.append(route)
    return sorted(missing)
