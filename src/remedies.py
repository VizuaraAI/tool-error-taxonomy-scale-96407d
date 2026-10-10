"""The seven type-matched remedies.

Each remedy targets exactly one error type. That one-to-one mapping is the
experiment: if the error map is right, a remedy's gain at a given scale should
track the error mass of its target type at that scale, and should be close to
zero where that type is rare.

A remedy is either a *prompt-side* change (extra instruction or exemplars) or a
*post-hoc repair* applied to the raw completion before attribution. Both kinds
are declared here so the sweep can apply them uniformly.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .parsing import ToolCall, parse_tool_calls
from .taxonomy import ErrorType

PromptHook = Callable[[str, Sequence[dict[str, Any]]], str]
RepairHook = Callable[[str, Sequence[dict[str, Any]]], str]


@dataclass
class Remedy:
    """One intervention, paired to the error type it is supposed to fix."""

    key: str
    target: ErrorType
    description: str
    prompt_suffix: str = ""
    repair: RepairHook | None = None
    needs_constrained_decoding: bool = False


def _properties(tools: Sequence[dict[str, Any]], name: str) -> dict[str, Any]:
    for tool in tools:
        spec = tool.get("function", tool)
        if spec.get("name") == name:
            return (spec.get("parameters") or {}).get("properties", {}) or {}
    return {}


def _tool_names(tools: Sequence[dict[str, Any]]) -> list[str]:
    return [t.get("function", t).get("name", "") for t in tools]


def _dump(calls: Sequence[ToolCall]) -> str:
    return json.dumps(
        [{"name": c.name, "arguments": c.arguments} for c in calls], ensure_ascii=False
    )


# --------------------------------------------------------------------------- #
# Repairs
# --------------------------------------------------------------------------- #

def repair_function_name(raw: str, tools: Sequence[dict[str, Any]]) -> str:
    """E6: snap a near-miss function name onto the closest schema name."""
    names = [n for n in _tool_names(tools) if n]
    calls = parse_tool_calls(raw)
    if not calls or not names:
        return raw
    fixed = []
    for call in calls:
        if call.name in names:
            fixed.append(call)
            continue
        match = difflib.get_close_matches(call.name, names, n=1, cutoff=0.72)
        fixed.append(ToolCall(match[0], call.arguments) if match else call)
    return _dump(fixed)


def _flat(name: str) -> str:
    """Strip case and separators so ``City_Name`` and ``cityName`` collide."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _canonical_name(key: str, allowed: list[str]) -> str:
    """Best schema name for `key`, or `key` itself when nothing is close.

    Three passes, cheapest and safest first: exact, then case and separator
    insensitive, then containment (``city_name`` carries ``city``), then fuzzy.
    Containment is resolved to the longest candidate so that a schema with both
    ``city`` and ``city_code`` does not silently pick the shorter one.
    """
    if key in allowed:
        return key
    flat_key = _flat(key)
    by_flat = {_flat(a): a for a in allowed}
    if flat_key in by_flat:
        return by_flat[flat_key]
    contained = [
        original
        for flat, original in by_flat.items()
        if flat and (flat in flat_key or flat_key in flat)
    ]
    if contained:
        return max(contained, key=lambda name: len(_flat(name)))
    match = difflib.get_close_matches(flat_key, list(by_flat), n=1, cutoff=0.7)
    return by_flat[match[0]] if match else key


def repair_argument_names(raw: str, tools: Sequence[dict[str, Any]]) -> str:
    """E3: canonicalise argument names (case, separators, close spellings).

    Only names are touched. A genuinely different name (``location`` where the
    schema says ``city``) is left alone: that is a real E3 and silently
    rewriting it would launder an error the study is trying to count.
    """
    calls = parse_tool_calls(raw)
    if not calls:
        return raw
    fixed = []
    for call in calls:
        allowed = list(_properties(tools, call.name))
        if not allowed:
            fixed.append(call)
            continue
        args: dict[str, Any] = {}
        for key, value in call.arguments.items():
            args[_canonical_name(key, allowed)] = value
        fixed.append(ToolCall(call.name, args))
    return _dump(fixed)


def repair_argument_types(raw: str, tools: Sequence[dict[str, Any]]) -> str:
    """E4: coerce a value to the type the schema declares, when unambiguous."""
    coercions: dict[str, Callable[[Any], Any]] = {
        "integer": lambda v: int(str(v).strip()),
        "number": lambda v: float(str(v).strip()),
        "string": lambda v: v if isinstance(v, str) else json.dumps(v),
        "boolean": lambda v: str(v).strip().lower() in {"true", "yes", "1"},
        "array": lambda v: v if isinstance(v, list) else [v],
    }
    calls = parse_tool_calls(raw)
    if not calls:
        return raw
    fixed = []
    for call in calls:
        properties = _properties(tools, call.name)
        args = dict(call.arguments)
        for key, value in call.arguments.items():
            declared = (properties.get(key) or {}).get("type")
            coerce = coercions.get(declared) if isinstance(declared, str) else None
            if coerce is None:
                continue
            try:
                args[key] = coerce(value)
            except (TypeError, ValueError):
                pass  # leave it alone; a failed coercion is still a real E4
        fixed.append(ToolCall(call.name, args))
    return _dump(fixed)


def repair_duplicates(raw: str, tools: Sequence[dict[str, Any]]) -> str:
    """E5: drop exact repeat calls, preserving first-occurrence order."""
    calls = parse_tool_calls(raw)
    if not calls:
        return raw
    seen: set[str] = set()
    unique = []
    for call in calls:
        signature = call.signature()
        if signature not in seen:
            seen.add(signature)
            unique.append(call)
    return _dump(unique) if len(unique) != len(calls) else raw


# --------------------------------------------------------------------------- #
# The bank
# --------------------------------------------------------------------------- #

REMEDIES: dict[str, Remedy] = {
    "grammar_constrained_decoding": Remedy(
        key="grammar_constrained_decoding",
        target=ErrorType.INVALID_FORMAT,
        description=(
            "Constrain generation to a JSON schema with XGrammar or Outlines, "
            "which makes an unparseable reply impossible by construction."
        ),
        needs_constrained_decoding=True,
    ),
    "tool_doc_retrieval": Remedy(
        key="tool_doc_retrieval",
        target=ErrorType.WRONG_FUNC_NAME,
        description=(
            "Retrieve the top-k most relevant tool documents for the query and "
            "show only those, shrinking the space of names to confuse."
        ),
        prompt_suffix=(
            "\nOnly these tool names exist. Using any other name is an error."
        ),
        repair=repair_function_name,
    ),
    "schema_name_canonicalisation": Remedy(
        key="schema_name_canonicalisation",
        target=ErrorType.WRONG_ARG_NAME,
        description=(
            "Echo the exact argument names back in the prompt and snap "
            "near-miss names onto the schema after generation."
        ),
        prompt_suffix=(
            "\nUse the argument names exactly as spelled in the schema."
        ),
        repair=repair_argument_names,
    ),
    "type_coercion_repair": Remedy(
        key="type_coercion_repair",
        target=ErrorType.WRONG_ARG_TYPE,
        description="Cast each value to its declared JSON-Schema type when unambiguous.",
        prompt_suffix="\nRespect the declared type of every argument.",
        repair=repair_argument_types,
    ),
    "value_grounding_fewshot": Remedy(
        key="value_grounding_fewshot",
        target=ErrorType.WRONG_ARG_VALUE,
        description=(
            "Add two worked exemplars that copy values verbatim out of the "
            "request rather than paraphrasing them."
        ),
        prompt_suffix=(
            "\nCopy argument values verbatim from the request. Do not paraphrase, "
            "round, translate or invent a value."
        ),
    ),
    "call_set_planning_prompt": Remedy(
        key="call_set_planning_prompt",
        target=ErrorType.INSUFFICIENT_CALLS,
        description=(
            "Ask the model to first list every sub-goal in the request, then "
            "emit one call per sub-goal."
        ),
        prompt_suffix=(
            "\nFirst count the distinct things the request asks for, then emit "
            "exactly that many calls. A request with two asks needs two calls."
        ),
    ),
    "duplicate_call_state_tracker": Remedy(
        key="duplicate_call_state_tracker",
        target=ErrorType.REPEATED_CALL,
        description=(
            "Track the calls already issued and suppress an exact repeat, which "
            "is never useful in a stateless benchmark."
        ),
        prompt_suffix="\nNever issue the same call with the same arguments twice.",
        repair=repair_duplicates,
    ),
}


def get(key: str) -> Remedy:
    if key not in REMEDIES:
        raise KeyError(f"unknown remedy {key!r}; known: {sorted(REMEDIES)}")
    return REMEDIES[key]


def apply_repair(key: str, raw: str, tools: Sequence[dict[str, Any]]) -> str:
    """Run a remedy's post-hoc repair, or return `raw` if it has none."""
    remedy = get(key)
    return remedy.repair(raw, tools) if remedy.repair else raw
