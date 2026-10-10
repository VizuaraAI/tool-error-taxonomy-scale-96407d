"""Turn whatever a small model emitted into a list of structured tool calls.

Small models are inconsistent about call format, and that inconsistency is one
of the things we are measuring (error type E7). So this parser is deliberately
*generous*: it accepts every format we have seen in the wild. If a reply still
cannot be parsed after all of these attempts, that is a genuine E7 and not an
artefact of a brittle parser.

Formats handled:
  1. Qwen / Hermes style   <tool_call>{"name": ..., "arguments": {...}}</tool_call>
  2. Raw JSON object       {"name": "f", "arguments": {"a": 1}}
  3. Raw JSON list         [{"name": "f", "arguments": {}}, ...]
  4. Fenced JSON           ```json ... ```
  5. BFCL AST style        [f(a=1, b="x"), g(c=2)]
  6. Bare python call      f(a=1)

Keep this module standard library only so it stays trivially testable.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

TOOL_CALL_TAG = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE
)
FENCED_BLOCK = re.compile(r"```(?:json|python|tool_code)?\s*(.*?)```", re.DOTALL)
PY_CALL = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\s*\((.*)\)\s*$", re.DOTALL)


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation predicted by the model or taken from the gold set."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def signature(self) -> str:
        """Canonical string used for duplicate detection (E5)."""
        try:
            args = json.dumps(self.arguments, sort_keys=True, default=str)
        except TypeError:
            args = repr(sorted(self.arguments.items(), key=lambda kv: kv[0]))
        return f"{self.name}({args})"


def _normalise_one(obj: Any) -> ToolCall | None:
    """Coerce a single decoded JSON object into a ToolCall, or return None."""
    if not isinstance(obj, dict):
        return None
    # OpenAI-style nesting: {"function": {"name": ..., "arguments": ...}}
    if "function" in obj and isinstance(obj["function"], dict):
        obj = obj["function"]
    name = obj.get("name") or obj.get("tool_name") or obj.get("api_name")
    if not isinstance(name, str) or not name.strip():
        return None
    args = obj.get("arguments")
    if args is None:
        args = obj.get("parameters", obj.get("args", {}))
    # Some models emit arguments as a JSON *string* rather than an object.
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"__unparsed__": args}
    if not isinstance(args, dict):
        args = {"__positional__": args}
    return ToolCall(name=name.strip(), arguments=args)


def _from_json_blob(text: str) -> list[ToolCall]:
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return []
    candidates = decoded if isinstance(decoded, list) else [decoded]
    calls = [_normalise_one(c) for c in candidates]
    return [c for c in calls if c is not None]


def _literal(node: ast.AST) -> Any:
    """Evaluate an argument node, falling back to its source-ish repr."""
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return ast.unparse(node)


def _from_python_calls(text: str) -> list[ToolCall]:
    """Parse BFCL-style `[f(a=1), g(b=2)]` or a bare `f(a=1)`."""
    stripped = text.strip()
    # A bracketed list of calls is not valid Python on its own only when the
    # calls are undefined names, which is fine: we parse, never execute.
    for candidate in (stripped, f"[{stripped}]"):
        try:
            tree = ast.parse(candidate, mode="eval")
        except SyntaxError:
            continue
        nodes: list[ast.Call] = []
        body = tree.body
        if isinstance(body, ast.Call):
            nodes = [body]
        elif isinstance(body, (ast.List, ast.Tuple)):
            nodes = [e for e in body.elts if isinstance(e, ast.Call)]
        if not nodes:
            continue
        calls: list[ToolCall] = []
        for node in nodes:
            func = node.func
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = ast.unparse(func)
            else:
                continue
            args = {kw.arg: _literal(kw.value) for kw in node.keywords if kw.arg}
            for i, positional in enumerate(node.args):
                args[f"__positional_{i}__"] = _literal(positional)
            calls.append(ToolCall(name=name, arguments=args))
        if calls:
            return calls
    return []


def parse_tool_calls(raw: str) -> list[ToolCall]:
    """Best-effort extraction of tool calls from a raw model completion.

    Returns an empty list when nothing call-shaped is present. An empty list is
    meaningful: combined with a non-empty gold set it is how E7 (invalid
    formatting) and E1 (insufficient calls) are told apart downstream.
    """
    if not raw or not raw.strip():
        return []

    tagged = TOOL_CALL_TAG.findall(raw)
    if tagged:
        calls: list[ToolCall] = []
        for block in tagged:
            calls.extend(_from_json_blob(block) or _from_python_calls(block))
        if calls:
            return calls

    for block in FENCED_BLOCK.findall(raw):
        calls = _from_json_blob(block) or _from_python_calls(block)
        if calls:
            return calls

    calls = _from_json_blob(raw)
    if calls:
        return calls

    # Last resort: find the outermost JSON object or array anywhere in the text.
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = raw.find(opener), raw.rfind(closer)
        if 0 <= start < end:
            calls = _from_json_blob(raw[start : end + 1])
            if calls:
                return calls

    return _from_python_calls(raw)


def parse_gold(entry: Any) -> list[ToolCall]:
    """Normalise a benchmark's gold answer field into ToolCall objects."""
    if isinstance(entry, str):
        return parse_tool_calls(entry)
    if isinstance(entry, dict):
        one = _normalise_one(entry)
        return [one] if one else []
    if isinstance(entry, list):
        out: list[ToolCall] = []
        for item in entry:
            out.extend(parse_gold(item))
        return out
    return []
