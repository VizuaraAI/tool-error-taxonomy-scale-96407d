"""Regression tests for the measuring instrument.

If the cascade is wrong, every number in the paper is wrong, so this is the
test file to extend first whenever you touch `src/attribution.py`.

    pytest -q
"""

from __future__ import annotations

import pytest

from src.attribution import _CASES, _TOOLS, attribute, error_mass_distribution
from src.parsing import ToolCall, parse_tool_calls
from src.remedies import REMEDIES, apply_repair
from src.taxonomy import CASCADE_ORDER, ErrorType, primary_of


@pytest.mark.parametrize("name,completion,gold,expected", _CASES)
def test_cascade_assigns_expected_primary(name, completion, gold, expected):
    result = attribute(completion, gold, _TOOLS, instance_id=name)
    assert result.primary == expected, f"{name}: notes={result.notes}"


@pytest.mark.parametrize(
    "raw,expected_name",
    [
        ('{"name": "f", "arguments": {"a": 1}}', "f"),
        ('[{"name": "f", "arguments": {}}]', "f"),
        ('<tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call>', "f"),
        ('```json\n{"name": "f", "arguments": {"a": 1}}\n```', "f"),
        ('[f(a=1)]', "f"),
        ('f(a=1, b="x")', "f"),
        ('{"function": {"name": "f", "arguments": "{\\"a\\": 1}"}}', "f"),
        ('Sure! Here you go: {"name": "f", "arguments": {"a": 1}}', "f"),
    ],
)
def test_parser_accepts_every_format_we_have_seen(raw, expected_name):
    calls = parse_tool_calls(raw)
    assert calls, f"failed to parse: {raw!r}"
    assert calls[0].name == expected_name


def test_parser_returns_empty_on_prose():
    assert parse_tool_calls("I cannot help with that.") == []
    assert parse_tool_calls("") == []


def test_unparseable_is_e7_but_prose_is_e1():
    gold = [ToolCall("get_weather", {"city": "Chennai"})]
    broken = attribute('{"name": "get_weather", "arguments": {', gold, _TOOLS)
    prose = attribute("I do not have live weather access.", gold, _TOOLS)
    assert broken.primary is ErrorType.INVALID_FORMAT
    assert prose.primary is ErrorType.INSUFFICIENT_CALLS


def test_relevance_detection_no_call_is_correct():
    result = attribute("No tool is needed here.", [], _TOOLS)
    assert result.correct
    assert result.primary is None


def test_relevance_detection_spurious_call_is_an_error():
    result = attribute('{"name": "get_weather", "arguments": {"city": "X"}}', [], _TOOLS)
    assert not result.correct
    assert result.primary is ErrorType.WRONG_FUNC_NAME


def test_value_comparison_is_case_and_whitespace_insensitive():
    gold = [ToolCall("get_weather", {"city": "Chennai"})]
    result = attribute('{"name": "get_weather", "arguments": {"city": "  chennai "}}',
                       gold, _TOOLS)
    assert result.correct


def test_bool_is_not_accepted_as_an_integer():
    gold = [ToolCall("get_weather", {"city": "Chennai", "days": 3})]
    result = attribute(
        '{"name": "get_weather", "arguments": {"city": "Chennai", "days": true}}',
        gold, _TOOLS,
    )
    assert ErrorType.WRONG_ARG_TYPE in result.labels


def test_primary_follows_cascade_order():
    labels = {ErrorType.WRONG_ARG_VALUE, ErrorType.INVALID_FORMAT}
    assert primary_of(labels) is ErrorType.INVALID_FORMAT
    assert primary_of(set()) is None


def test_error_mass_distribution_sums_to_100():
    results = [attribute(c, g, _TOOLS) for _, c, g, _ in _CASES]
    distribution = error_mass_distribution(results)
    assert set(distribution) == {e.value for e in CASCADE_ORDER}
    assert abs(sum(distribution.values()) - 100.0) < 1e-6


def test_multi_label_view_keeps_secondary_errors():
    gold = [
        ToolCall("get_weather", {"city": "Chennai"}),
        ToolCall("book_flight", {"origin": "MAA", "dest": "BOM"}),
    ]
    # One call, to the wrong city: both insufficient and wrong-value.
    result = attribute(
        '[{"name": "get_weather", "arguments": {"city": "Mumbai"}}]', gold, _TOOLS
    )
    assert ErrorType.WRONG_ARG_VALUE in result.labels
    assert ErrorType.INSUFFICIENT_CALLS in result.labels
    assert result.primary is ErrorType.WRONG_ARG_VALUE  # value is earlier in the cascade


# --------------------------------------------------------------------------- #
# Remedies. Each one must fix its own target type, and only its own.
# --------------------------------------------------------------------------- #

def test_every_error_type_has_exactly_one_remedy():
    targets = [r.target for r in REMEDIES.values()]
    assert len(targets) == len(set(targets)) == len(CASCADE_ORDER)


@pytest.mark.parametrize(
    "drifted",
    ["City_Name", "cityName", "CITY", "city "],
)
def test_name_canonicalisation_absorbs_case_and_separator_drift(drifted):
    gold = [ToolCall("get_weather", {"city": "Chennai"})]
    raw = '{"name": "get_weather", "arguments": {"%s": "Chennai"}}' % drifted
    repaired = apply_repair("schema_name_canonicalisation", raw, _TOOLS)
    assert attribute(repaired, gold, _TOOLS).correct


def test_name_canonicalisation_leaves_a_genuinely_wrong_name_alone():
    """Rewriting `location` to `city` would launder an error we must count."""
    gold = [ToolCall("get_weather", {"city": "Chennai"})]
    raw = '{"name": "get_weather", "arguments": {"location": "Chennai"}}'
    repaired = apply_repair("schema_name_canonicalisation", raw, _TOOLS)
    assert attribute(repaired, gold, _TOOLS).primary is ErrorType.WRONG_ARG_NAME


def test_function_name_repair_snaps_a_near_miss():
    gold = [ToolCall("get_weather", {"city": "Chennai"})]
    raw = '{"name": "get_wether", "arguments": {"city": "Chennai"}}'
    assert attribute(raw, gold, _TOOLS).primary is ErrorType.WRONG_FUNC_NAME
    repaired = apply_repair("tool_doc_retrieval", raw, _TOOLS)
    assert attribute(repaired, gold, _TOOLS).correct


def test_type_coercion_fixes_a_stringified_integer():
    gold = [ToolCall("get_weather", {"city": "Chennai", "days": 3})]
    raw = '{"name": "get_weather", "arguments": {"city": "Chennai", "days": "3"}}'
    assert attribute(raw, gold, _TOOLS).primary is ErrorType.WRONG_ARG_TYPE
    repaired = apply_repair("type_coercion_repair", raw, _TOOLS)
    assert attribute(repaired, gold, _TOOLS).correct


def test_duplicate_tracker_drops_exact_repeats():
    gold = [ToolCall("get_weather", {"city": "Chennai"})]
    raw = ('[{"name": "get_weather", "arguments": {"city": "Chennai"}},'
           ' {"name": "get_weather", "arguments": {"city": "Chennai"}}]')
    assert attribute(raw, gold, _TOOLS).primary is ErrorType.REPEATED_CALL
    repaired = apply_repair("duplicate_call_state_tracker", raw, _TOOLS)
    assert attribute(repaired, gold, _TOOLS).correct
