"""The error attribution cascade: the measuring instrument of this project.

Given (raw model completion, gold tool calls, tool schemas) this module
returns one primary error type plus every co-occurring error type. Figure 1 of
the paper is a picture of the function `attribute` below.

Design rules we committed to, and why:

* Deterministic. No model-as-judge anywhere in the measurement path. If the
  instrument used an LLM, the instrument itself would shift with scale and the
  headline result would be uninterpretable.
* Schema-aware. Argument-name and argument-type errors are judged against the
  tool schema, not against the gold call, because a model can invent a name
  that happens to be absent from the gold call but valid in the schema.
* One primary label, many secondary labels. Per-instance primary labels give a
  partition of error mass that sums to 100 percent (Figure 2a). The multi-label
  view keeps the information the partition throws away.

Run `python -m src.attribution --selftest` for a dependency-free sanity check.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .parsing import ToolCall, parse_tool_calls
from .taxonomy import CASCADE_ORDER, ErrorType, primary_of

#: JSON-schema type name to accepted python types.
JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "float": (int, float),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
    "any": (object,),
}

#: Characters whose presence means the model was *trying* to emit a call. Used
#: to separate "emitted something broken" (E7) from "answered in prose" (E1).
CALL_ATTEMPT = re.compile(r"(<tool_call>|```|[{\[]|\w+\s*\()")

NUMERIC_TOLERANCE = 1e-6


@dataclass
class Attribution:
    """Attribution result for a single benchmark instance."""

    instance_id: str
    model: str
    correct: bool
    primary: ErrorType | None
    labels: tuple[ErrorType, ...] = ()
    n_gold: int = 0
    n_pred: int = 0
    notes: list[str] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        """Flat dict for pandas / JSONL, one row per instance."""
        row: dict[str, Any] = {
            "instance_id": self.instance_id,
            "model": self.model,
            "correct": self.correct,
            "primary": self.primary.value if self.primary else "",
            "n_gold": self.n_gold,
            "n_pred": self.n_pred,
            "notes": "; ".join(self.notes),
        }
        for error in CASCADE_ORDER:
            row[error.value] = error in self.labels
        return row


def _schema_index(tools: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map tool name to its parameter schema, tolerating both common layouts."""
    index: dict[str, dict[str, Any]] = {}
    for tool in tools or []:
        spec = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = spec.get("name")
        if not isinstance(name, str):
            continue
        params = spec.get("parameters") or spec.get("input_schema") or {}
        index[name] = {
            "properties": params.get("properties", {}) or {},
            "required": list(params.get("required", []) or []),
        }
    return index


def _type_ok(value: Any, declared: Any) -> bool:
    """True when `value` is consistent with a JSON-schema type declaration."""
    if declared is None:
        return True
    names = declared if isinstance(declared, list) else [declared]
    for name in names:
        if not isinstance(name, str):
            continue
        accepted = JSON_TYPES.get(name.lower())
        if accepted is None:
            return True  # unknown type name: do not penalise the model
        if name.lower() in {"integer", "number", "float"} and isinstance(value, bool):
            continue  # bool is an int subclass in python; do not let it pass
        if isinstance(value, accepted):
            return True
    return False


def _values_equal(predicted: Any, gold: Any) -> bool:
    """Loose equality: case and whitespace insensitive, numeric tolerance."""
    if isinstance(predicted, bool) or isinstance(gold, bool):
        return bool(predicted) == bool(gold)
    if isinstance(predicted, (int, float)) and isinstance(gold, (int, float)):
        return math.isclose(predicted, gold, rel_tol=NUMERIC_TOLERANCE, abs_tol=NUMERIC_TOLERANCE)
    if isinstance(predicted, str) and isinstance(gold, str):
        return predicted.strip().lower() == gold.strip().lower()
    if isinstance(predicted, (list, tuple)) and isinstance(gold, (list, tuple)):
        if len(predicted) != len(gold):
            return False
        return all(_values_equal(p, g) for p, g in zip(predicted, gold))
    if isinstance(predicted, dict) and isinstance(gold, dict):
        if set(predicted) != set(gold):
            return False
        return all(_values_equal(predicted[k], gold[k]) for k in predicted)
    # Benchmarks sometimes store a gold value as a one-element list of options.
    if isinstance(gold, (list, tuple)) and len(gold) == 1:
        return _values_equal(predicted, gold[0])
    return predicted == gold


def _align(predicted: list[ToolCall], gold: list[ToolCall]) -> list[tuple[ToolCall, ToolCall]]:
    """Greedily pair predicted calls to gold calls by (lowercased) name."""
    unused = list(predicted)
    pairs: list[tuple[ToolCall, ToolCall]] = []
    for gold_call in gold:
        match = next(
            (p for p in unused if p.name.lower() == gold_call.name.lower()), None
        )
        if match is not None:
            unused.remove(match)
            pairs.append((match, gold_call))
    return pairs


def attribute(
    raw_completion: str,
    gold: Sequence[ToolCall],
    tools: Sequence[dict[str, Any]],
    instance_id: str = "",
    model: str = "",
) -> Attribution:
    """Attribute a single prediction to error types.

    `gold` may be empty, which encodes a relevance-detection instance where the
    correct behaviour is to call nothing. Calling anything there is recorded as
    a wrong-function-name error (the model invoked a tool that should not have
    been invoked) and noted so the case can be split out during analysis.
    """
    labels: set[ErrorType] = set()
    notes: list[str] = []
    predicted = parse_tool_calls(raw_completion)
    schemas = _schema_index(tools)
    gold = list(gold)

    if not gold:
        if predicted:
            labels.add(ErrorType.WRONG_FUNC_NAME)
            notes.append("relevance_detection: called a tool when none was needed")
        return Attribution(
            instance_id=instance_id,
            model=model,
            correct=not predicted,
            primary=primary_of(labels),
            labels=tuple(e for e in CASCADE_ORDER if e in labels),
            n_gold=0,
            n_pred=len(predicted),
            notes=notes,
        )

    if not predicted:
        attempted = bool(CALL_ATTEMPT.search(raw_completion or ""))
        labels.add(ErrorType.INVALID_FORMAT if attempted else ErrorType.INSUFFICIENT_CALLS)
        notes.append(
            "unparseable call syntax" if attempted else "answered in prose, no call emitted"
        )
        return Attribution(
            instance_id=instance_id,
            model=model,
            correct=False,
            primary=primary_of(labels),
            labels=tuple(e for e in CASCADE_ORDER if e in labels),
            n_gold=len(gold),
            n_pred=0,
            notes=notes,
        )

    # E5: the same call emitted more than once.
    signatures = Counter(call.signature() for call in predicted)
    duplicates = [sig for sig, count in signatures.items() if count > 1]
    if duplicates:
        labels.add(ErrorType.REPEATED_CALL)
        notes.append(f"{len(duplicates)} duplicated call signature(s)")

    # E6: a called function that does not exist in the provided schema set. If
    # no schemas were supplied we fall back to the gold function names.
    known = set(schemas) or {g.name for g in gold}
    known_lower = {name.lower() for name in known}
    unknown = [c.name for c in predicted if c.name.lower() not in known_lower]
    if unknown:
        labels.add(ErrorType.WRONG_FUNC_NAME)
        notes.append("unknown function(s): " + ", ".join(sorted(set(unknown))))

    # Argument-level checks on the aligned pairs.
    for pred_call, gold_call in _align(predicted, gold):
        schema = schemas.get(pred_call.name) or schemas.get(gold_call.name) or {}
        properties: dict[str, Any] = schema.get("properties", {})
        required: list[str] = schema.get("required", [])

        # E3: hallucinated argument names, or required arguments left out.
        if properties:
            extra = [k for k in pred_call.arguments if k not in properties]
        else:
            extra = [k for k in pred_call.arguments if k not in gold_call.arguments]
        missing = [k for k in required if k not in pred_call.arguments]
        missing += [
            k
            for k in gold_call.arguments
            if k not in pred_call.arguments and k not in required
        ]
        if extra or missing:
            labels.add(ErrorType.WRONG_ARG_NAME)
            if extra:
                notes.append(f"{pred_call.name}: unexpected arg(s) {sorted(set(extra))}")
            if missing:
                notes.append(f"{pred_call.name}: missing arg(s) {sorted(set(missing))}")

        for key, value in pred_call.arguments.items():
            declared = (properties.get(key) or {}).get("type") if properties else None
            # E4: right name, wrong python/JSON type ("3" instead of 3).
            if declared is not None and not _type_ok(value, declared):
                labels.add(ErrorType.WRONG_ARG_TYPE)
                notes.append(
                    f"{pred_call.name}.{key}: expected {declared}, got {type(value).__name__}"
                )
                continue
            # E2: right name and type, wrong content.
            if key in gold_call.arguments and not _values_equal(
                value, gold_call.arguments[key]
            ):
                labels.add(ErrorType.WRONG_ARG_VALUE)
                notes.append(f"{pred_call.name}.{key}: value mismatch")

    # E1: fewer usable calls than the task requires.
    matched = len(_align(predicted, gold))
    if matched < len(gold):
        labels.add(ErrorType.INSUFFICIENT_CALLS)
        notes.append(f"matched {matched} of {len(gold)} required call(s)")

    return Attribution(
        instance_id=instance_id,
        model=model,
        correct=not labels,
        primary=primary_of(labels),
        labels=tuple(e for e in CASCADE_ORDER if e in labels),
        n_gold=len(gold),
        n_pred=len(predicted),
        notes=notes,
    )


def error_mass_distribution(results: Iterable[Attribution]) -> dict[str, float]:
    """Share of primary error labels per type, as percentages summing to 100.

    This is the quantity plotted in Figure 2a. Note it is a share of *errors*,
    not of instances: a model can lower its total error rate while raising a
    type's share, which is exactly the migration effect we are after.
    """
    counts = Counter(r.primary for r in results if r.primary is not None)
    total = sum(counts.values())
    if total == 0:
        return {e.value: 0.0 for e in CASCADE_ORDER}
    return {e.value: 100.0 * counts.get(e, 0) / total for e in CASCADE_ORDER}


# --------------------------------------------------------------------------- #
# Self-test: seven hand-written cases, one per error type. Extend this first
# whenever you change the cascade; it is the cheapest regression test you have.
# --------------------------------------------------------------------------- #

_TOOLS = [
    {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
            },
            "required": ["city"],
        },
    },
    {
        "name": "book_flight",
        "description": "Book a flight.",
        "parameters": {
            "type": "object",
            "properties": {"origin": {"type": "string"}, "dest": {"type": "string"}},
            "required": ["origin", "dest"],
        },
    },
]

_CASES: list[tuple[str, str, list[ToolCall], ErrorType | None]] = [
    (
        "correct",
        '{"name": "get_weather", "arguments": {"city": "Chennai", "days": 3}}',
        [ToolCall("get_weather", {"city": "Chennai", "days": 3})],
        None,
    ),
    (
        "E7 invalid formatting",
        '{"name": "get_weather", "arguments": {"city": "Chennai"',
        [ToolCall("get_weather", {"city": "Chennai"})],
        ErrorType.INVALID_FORMAT,
    ),
    (
        "E6 wrong function name",
        '{"name": "fetch_weather", "arguments": {"city": "Chennai"}}',
        [ToolCall("get_weather", {"city": "Chennai"})],
        ErrorType.WRONG_FUNC_NAME,
    ),
    (
        "E3 wrong argument name",
        '{"name": "get_weather", "arguments": {"location": "Chennai"}}',
        [ToolCall("get_weather", {"city": "Chennai"})],
        ErrorType.WRONG_ARG_NAME,
    ),
    (
        "E4 wrong argument type",
        '{"name": "get_weather", "arguments": {"city": "Chennai", "days": "three"}}',
        [ToolCall("get_weather", {"city": "Chennai", "days": 3})],
        ErrorType.WRONG_ARG_TYPE,
    ),
    (
        "E2 wrong argument value",
        '{"name": "get_weather", "arguments": {"city": "Mumbai", "days": 3}}',
        [ToolCall("get_weather", {"city": "Chennai", "days": 3})],
        ErrorType.WRONG_ARG_VALUE,
    ),
    (
        "E1 insufficient calls",
        '[{"name": "get_weather", "arguments": {"city": "Chennai"}}]',
        [
            ToolCall("get_weather", {"city": "Chennai"}),
            ToolCall("book_flight", {"origin": "MAA", "dest": "BOM"}),
        ],
        ErrorType.INSUFFICIENT_CALLS,
    ),
    (
        "E5 repeated calls",
        '[{"name": "get_weather", "arguments": {"city": "Chennai"}},'
        ' {"name": "get_weather", "arguments": {"city": "Chennai"}}]',
        [ToolCall("get_weather", {"city": "Chennai"})],
        ErrorType.REPEATED_CALL,
    ),
    (
        "E1 prose answer, no call attempted",
        "I am sorry, I do not have access to live weather data.",
        [ToolCall("get_weather", {"city": "Chennai"})],
        ErrorType.INSUFFICIENT_CALLS,
    ),
]


def selftest() -> int:
    """Assert the cascade assigns the intended primary label on each case."""
    failures = 0
    for name, completion, gold, expected in _CASES:
        got = attribute(completion, gold, _TOOLS, instance_id=name, model="selftest")
        ok = got.primary == expected
        status = "ok  " if ok else "FAIL"
        failures += 0 if ok else 1
        expected_label = expected.value if expected else "correct"
        actual_label = got.primary.value if got.primary else "correct"
        print(f"[{status}] {name:38s} expected={expected_label:26s} got={actual_label}")
    print(f"\n{len(_CASES) - failures}/{len(_CASES)} cases passed")
    if not failures:
        dist = error_mass_distribution(
            [attribute(c, g, _TOOLS) for _, c, g, _ in _CASES]
        )
        print("error mass distribution over the self-test cases:")
        print(json.dumps({k: round(v, 1) for k, v in dist.items()}, indent=2))
    return failures


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        raise SystemExit(1 if selftest() else 0)
    print(__doc__)
    print("Run with --selftest to exercise the cascade.")
