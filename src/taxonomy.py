"""The seven tool-use error types, and the fixed cascade order used to pick
a single primary label for every failed prediction.

This module is pure standard library on purpose: it is the scientific core of
the project and you should be able to read it, test it and trust it before any
model weights are downloaded.

Cascade order matters. We resolve surface-level errors first (a reply that does
not parse cannot also be judged on its argument values), then schema-level
errors, then semantic errors, then multi-call planning errors. Changing the
order changes every number in the paper, so it lives in exactly one place:
`CASCADE_ORDER` below.
"""

from __future__ import annotations

from enum import Enum


class ErrorType(str, Enum):
    """The seven recurring tool-use error types we attribute."""

    INSUFFICIENT_CALLS = "E1_insufficient_calls"
    WRONG_ARG_VALUE = "E2_wrong_argument_values"
    WRONG_ARG_NAME = "E3_wrong_argument_names"
    WRONG_ARG_TYPE = "E4_wrong_argument_types"
    REPEATED_CALL = "E5_repeated_calls"
    WRONG_FUNC_NAME = "E6_wrong_function_name"
    INVALID_FORMAT = "E7_invalid_formatting"


#: Order in which the attribution cascade tests for errors. The first error
#: type that fires becomes the *primary* label for that instance.
CASCADE_ORDER: tuple[ErrorType, ...] = (
    ErrorType.INVALID_FORMAT,
    ErrorType.WRONG_FUNC_NAME,
    ErrorType.WRONG_ARG_NAME,
    ErrorType.WRONG_ARG_TYPE,
    ErrorType.WRONG_ARG_VALUE,
    ErrorType.INSUFFICIENT_CALLS,
    ErrorType.REPEATED_CALL,
)

#: Two families used throughout the analysis. The central hypothesis of the
#: project is that error mass migrates from SURFACE to SEMANTIC as scale grows.
SURFACE_TYPES: frozenset[ErrorType] = frozenset(
    {
        ErrorType.INVALID_FORMAT,
        ErrorType.WRONG_FUNC_NAME,
        ErrorType.WRONG_ARG_NAME,
    }
)

SEMANTIC_TYPES: frozenset[ErrorType] = frozenset(
    {
        ErrorType.WRONG_ARG_VALUE,
        ErrorType.INSUFFICIENT_CALLS,
        ErrorType.REPEATED_CALL,
    }
)

#: E4 sits between the two families: a type error is a schema violation, but it
#: usually comes from a value the model genuinely believed in. We report it
#: separately rather than forcing it into one family.
BRIDGE_TYPES: frozenset[ErrorType] = frozenset({ErrorType.WRONG_ARG_TYPE})

#: Short human labels for figures and tables.
SHORT_LABELS: dict[ErrorType, str] = {
    ErrorType.INSUFFICIENT_CALLS: "insufficient calls",
    ErrorType.WRONG_ARG_VALUE: "wrong arg values",
    ErrorType.WRONG_ARG_NAME: "wrong arg names",
    ErrorType.WRONG_ARG_TYPE: "wrong arg types",
    ErrorType.REPEATED_CALL: "repeated calls",
    ErrorType.WRONG_FUNC_NAME: "wrong func name",
    ErrorType.INVALID_FORMAT: "invalid formatting",
}

#: The remedy we expect to help each error type. Week 6 of the roadmap turns
#: this mapping into the remedy-by-scale gain matrix (Figure 2b).
REMEDY_FOR: dict[ErrorType, str] = {
    ErrorType.INVALID_FORMAT: "grammar_constrained_decoding",
    ErrorType.WRONG_FUNC_NAME: "tool_doc_retrieval",
    ErrorType.WRONG_ARG_NAME: "schema_name_canonicalisation",
    ErrorType.WRONG_ARG_TYPE: "type_coercion_repair",
    ErrorType.WRONG_ARG_VALUE: "value_grounding_fewshot",
    ErrorType.INSUFFICIENT_CALLS: "call_set_planning_prompt",
    ErrorType.REPEATED_CALL: "duplicate_call_state_tracker",
}


def primary_of(labels: set[ErrorType]) -> ErrorType | None:
    """Return the highest-priority error in `labels` per `CASCADE_ORDER`."""
    for error in CASCADE_ORDER:
        if error in labels:
            return error
    return None


def family_of(error: ErrorType) -> str:
    """Return "surface", "semantic" or "bridge" for an error type."""
    if error in SURFACE_TYPES:
        return "surface"
    if error in SEMANTIC_TYPES:
        return "semantic"
    return "bridge"
