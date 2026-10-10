"""Benchmark loaders, normalised to one instance schema.

Everything downstream sees the same `Instance` shape no matter which benchmark
it came from. When you add a benchmark, write a loader that produces this
shape and nothing else in the codebase has to change.

No data files live in this repo. Loaders pull from the Hugging Face Hub or a
local path you point them at.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from .parsing import ToolCall, parse_gold

#: Benchmarks we support, with the source the loader reads from.
SOURCES: dict[str, str] = {
    "bfcl": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
    "nestful": "ibm-research/nestful",
    "apibank": "local",  # clone github.com/AlibabaResearch/DAMO-ConvAI
}


@dataclass
class Instance:
    """One evaluation item: a query, the tools on offer, and the gold calls."""

    instance_id: str
    benchmark: str
    query: str
    tools: list[dict[str, Any]] = field(default_factory=list)
    gold: list[ToolCall] = field(default_factory=list)
    category: str = ""

    @property
    def requires_no_call(self) -> bool:
        """True for relevance-detection items where calling anything is wrong."""
        return not self.gold


def _as_tool_list(raw: Any) -> list[dict[str, Any]]:
    """Normalise a benchmark's function/tool field into a list of schemas."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if isinstance(raw, dict):
        raw = [raw]
    return [t for t in raw if isinstance(t, dict)] if isinstance(raw, list) else []


def _first(record: dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        if record.get(key) not in (None, "", [], {}):
            return record[key]
    return default


def from_records(
    records: Sequence[dict[str, Any]], benchmark: str
) -> list[Instance]:
    """Map raw benchmark records onto `Instance`, tolerating field-name drift.

    Benchmarks disagree about what to call things (``function`` vs ``tools``,
    ``ground_truth`` vs ``answer``). Rather than one loader per field layout we
    probe a short list of known aliases, which keeps adding a benchmark cheap.
    """
    out: list[Instance] = []
    for i, rec in enumerate(records):
        query = _first(rec, "question", "query", "input", "user_query")
        if isinstance(query, list):  # BFCL stores turns as nested lists
            flat = [m for turn in query for m in (turn if isinstance(turn, list) else [turn])]
            query = " ".join(
                m.get("content", "") if isinstance(m, dict) else str(m) for m in flat
            ).strip()
        out.append(
            Instance(
                instance_id=str(_first(rec, "id", "idx", "sample_id", default=f"{benchmark}-{i}")),
                benchmark=benchmark,
                query=str(query),
                tools=_as_tool_list(_first(rec, "function", "tools", "functions", "apis")),
                gold=parse_gold(
                    _first(rec, "ground_truth", "answer", "gold", "output", default=[])
                ),
                category=str(_first(rec, "test_category", "category", "type")),
            )
        )
    return out


def load_jsonl(path: str | pathlib.Path, benchmark: str = "custom") -> list[Instance]:
    """Load a local .jsonl dump. Useful for a quick private test set."""
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return from_records(records, benchmark)


def load_hf(name: str, split: str = "train", limit: int | None = None) -> list[Instance]:
    """Load a supported benchmark from the Hugging Face Hub.

    Imported lazily so that `parsing`, `taxonomy` and `attribution` stay usable
    (and testable) without the ML stack installed.
    """
    from datasets import load_dataset  # noqa: PLC0415

    source = SOURCES.get(name)
    if source in (None, "local"):
        raise ValueError(
            f"{name!r} has no Hub mirror configured. Download it and use load_jsonl, "
            f"or add an entry to SOURCES. Known: {sorted(SOURCES)}"
        )
    dataset = load_dataset(source, split=split)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return from_records([dict(r) for r in dataset], name)


def iter_batches(items: Sequence[Instance], size: int) -> Iterator[list[Instance]]:
    """Yield fixed-size batches; the last batch may be short."""
    for start in range(0, len(items), size):
        yield list(items[start : start + size])
