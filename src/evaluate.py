"""Entry point: run one model over one benchmark and write attributed results.

    python -m src.evaluate --model 0.5B --benchmark bfcl --limit 200
    python -m src.evaluate --model 3B --benchmark nestful --remedy call_set_planning_prompt

Output is one JSONL row per instance in `runs/<run-name>.jsonl`, plus a small
`.summary.json`. Those rows are exactly the rows of `experiments/log_template.md`
and the columns of Table 1 in the paper.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import yaml

from .attribution import attribute
from .data import Instance, iter_batches, load_hf, load_jsonl
from .models import ChatModel, GenerationConfig, build_prompt
from .remedies import apply_repair, get as get_remedy
from .taxonomy import CASCADE_ORDER

CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def load_config(path: str | pathlib.Path = CONFIG_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_messages(
    instance: Instance, remedy_key: str | None
) -> list[dict[str, str]]:
    messages = build_prompt(instance.query, instance.tools)
    if remedy_key:
        suffix = get_remedy(remedy_key).prompt_suffix
        if suffix:
            messages[0]["content"] += suffix
    return messages


def run(
    model_name: str,
    instances: list[Instance],
    *,
    remedy: str | None = None,
    batch_size: int = 8,
    gen: GenerationConfig | None = None,
) -> list[dict[str, Any]]:
    """Generate, repair if a remedy asks for it, attribute, and return rows."""
    model = ChatModel(model_name)
    gen = gen or GenerationConfig()
    rows: list[dict[str, Any]] = []

    for batch in iter_batches(instances, batch_size):
        completions = model.generate(
            [build_messages(inst, remedy) for inst in batch], gen
        )
        for inst, raw in zip(batch, completions):
            repaired = apply_repair(remedy, raw, inst.tools) if remedy else raw
            result = attribute(
                repaired, inst.gold, inst.tools,
                instance_id=inst.instance_id, model=model_name,
            )
            row = result.to_row()
            row.update(
                benchmark=inst.benchmark,
                category=inst.category,
                remedy=remedy or "none",
                raw_completion=raw,
            )
            rows.append(row)
        print(f"  {len(rows)}/{len(instances)} attributed", end="\r", flush=True)
    print()
    return rows


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Headline numbers for one run: accuracy, error rate, mass distribution."""
    total = len(rows) or 1
    correct = sum(1 for r in rows if r["correct"])
    counts = {e.value: sum(1 for r in rows if r["primary"] == e.value) for e in CASCADE_ORDER}
    return {
        "n": len(rows),
        "call_accuracy": round(100.0 * correct / total, 2),
        "total_error_rate": round(100.0 * (total - correct) / total, 2),
        "primary_error_rate": {k: round(100.0 * v / total, 2) for k, v in counts.items()},
        "error_mass_share": {
            k: round(v, 2)
            for k, v in error_mass_distribution_from_rows(rows).items()
        },
    }


def error_mass_distribution_from_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Share of each primary label among errors only. Mirrors Figure 2a."""
    errors = [r for r in rows if r["primary"]]
    total = len(errors)
    if total == 0:
        return {e.value: 0.0 for e in CASCADE_ORDER}
    return {
        e.value: 100.0 * sum(1 for r in errors if r["primary"] == e.value) / total
        for e in CASCADE_ORDER
    }


def main(argv: list[str] | None = None) -> int:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="0.5B", help="ladder key or Hub id")
    parser.add_argument("--benchmark", default=config.get("benchmark", "bfcl"))
    parser.add_argument("--jsonl", default=None, help="local .jsonl instead of the Hub")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=config.get("limit", 200))
    parser.add_argument("--remedy", default=None)
    parser.add_argument("--batch-size", type=int, default=config.get("batch_size", 8))
    parser.add_argument("--max-new-tokens", type=int, default=config.get("max_new_tokens", 512))
    parser.add_argument("--temperature", type=float, default=config.get("temperature", 0.0))
    parser.add_argument("--seed", type=int, default=config.get("seed", 0))
    parser.add_argument("--out-dir", default=config.get("out_dir", "runs"))
    args = parser.parse_args(argv)

    instances = (
        load_jsonl(args.jsonl, args.benchmark)
        if args.jsonl
        else load_hf(args.benchmark, split=args.split, limit=args.limit)
    )
    if args.jsonl and args.limit:
        instances = instances[: args.limit]
    print(f"{len(instances)} instances from {args.benchmark}")

    rows = run(
        args.model, instances,
        remedy=args.remedy,
        batch_size=args.batch_size,
        gen=GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed,
        ),
    )

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}_{args.benchmark}_{args.remedy or 'none'}_seed{args.seed}"
    with open(out_dir / f"{stem}.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarise(rows) | {
        "model": args.model, "benchmark": args.benchmark,
        "remedy": args.remedy or "none", "seed": args.seed,
    }
    with open(out_dir / f"{stem}.summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
