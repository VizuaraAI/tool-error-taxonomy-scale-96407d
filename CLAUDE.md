# CLAUDE.md

Instructions for the coding agent working in this repository with Balakrishnan.

## What this project is

An empirical study, not a systems project. The deliverable is a short paper
showing how the *composition* of tool-calling errors changes across model
scale (0.5B, 1.5B, 3B), and that the best remedy changes with it.

Seven error types, following the ToolScan taxonomy (arXiv:2411.13547):
E1 insufficient calls, E2 wrong argument values, E3 wrong argument names,
E4 wrong argument types, E5 repeated calls, E6 wrong function name,
E7 invalid formatting.

Central hypothesis: error mass migrates from the **surface** family
(E7, E6, E3) at 0.5B to the **semantic** family (E2, E1, E5) at 3B.

## Non-negotiables

Read these before proposing a change. They are design decisions, not accidents.

1. **The attribution cascade has no model-as-judge.** Not an LLM grader, not an
   embedding similarity, nothing learned. If the instrument shifted with the
   scale being measured, the headline result would be uninterpretable. If you
   are tempted to add "just use a small LLM to classify this edge case", the
   answer is no; add a deterministic rule to `src/attribution.py` instead.
2. **`CASCADE_ORDER` lives in exactly one place** (`src/taxonomy.py`) and
   changing it changes every number in the paper. Never reorder it as a
   drive-by fix. If a change is genuinely warranted, flag it, change it in one
   commit on its own, and re-run everything.
3. **The parser is deliberately generous** (`src/parsing.py`). A brittle parser
   manufactures E7 errors and would fake the paper's main finding. When a new
   output format shows up, add it to the parser and add a case to
   `tests/test_attribution.py`.
4. **Prose answer is not the same as malformed syntax.** An empty parse with
   call-shaped residue is E7; an empty parse with a fluent prose answer is E1.
   Conflating them is the most common measurement bug in this setting.
5. **The ladder must stay controlled.** Qwen2.5-Instruct 0.5B/1.5B/3B share a
   post-training recipe, which is the whole reason the comparison is clean. Do
   not quietly swap a rung for a different family. Cross-family models go in
   `CROSS_FAMILY` and are reported separately.
6. **No placeholder code.** If something is not implemented, do not stub it
   with `pass` and a TODO that silently returns a wrong answer. Raise.

## Layout and conventions

- `src/taxonomy.py`, `src/parsing.py`, `src/attribution.py`: **standard library
  only**. Keep it that way. It is what makes the science testable in seconds
  with no install.
- Heavy imports (`torch`, `transformers`, `datasets`, `peft`) go **inside
  functions**, so `--help` and the tests work on a bare machine.
- Type hints everywhere; `from __future__ import annotations` at the top.
- Entry points are `python -m src.<module>` with argparse and a `main(argv)`
  that returns an int.
- Results are append-only JSONL in `runs/`. Analysis never mutates them.
- Docstrings explain *why*, not *what*. The what is readable from the code.

## Workflow

```bash
python -m src.attribution --selftest   # fastest feedback loop, no deps
pytest -q                              # full instrument regression
LIMIT=50 ./scripts/run_eval.sh 0.5B bfcl
python -m src.analyze --runs runs --out analysis
```

After any change to `attribution.py`, `parsing.py` or `taxonomy.py`, run the
self-test and pytest before anything else. After any change that could alter a
number, say so explicitly and name which table it affects.

## Experiment hygiene

Every run appends a row to `experiments/log.md` (copy it from
`log_template.md` the first time). The log's tables map one-to-one onto the
paper's tables. A result that is not in the log does not exist.

When reporting a run back to Balakrishnan, give: call accuracy, total error
rate, and the seven-way error mass share. Not just accuracy. The share is the
point of the project.

## Where the work is going

Roughly in order. Do not jump ahead; each step de-risks the next.

1. Instrument is trustworthy: self-test passes, and the cascade agrees with a
   hand-labelled slice of ~100 instances per type.
2. Baselines on all three rungs, one benchmark, one seed. Look at the shape.
3. Add NESTFUL. It is the only benchmark where E1 and E5 can occur at all, so
   without it two of the seven types read as zero for structural reasons.
4. Three seeds, and per-benchmark breakdown.
5. The remedy grid: seven remedies times three rungs. This is the falsifiable
   part. A remedy's gain should track its target type's error mass share.
6. Cross-family replication, which decides whether the story is about scale or
   about post-training data.
7. Optional: the LoRA extension in `src/train_lora.py`.
8. Write up. Target is a 4 to 8 page workshop paper.

## Things that would falsify the project

Say so loudly if you see any of these in the data; they are results, not
failures, and the paper commits to reporting them.

- All seven elasticities come out similar: scale shrinks the error set without
  reshaping it.
- Remedy gains do not track error mass shares: the taxonomy is descriptive but
  not actionable.
- Cross-family models cluster by training recipe rather than by size: the
  independent variable is data, not scale.

## Tone

Balakrishnan is learning to do research, not just to ship code. When you make a
non-obvious choice, say in one line why, and name the alternative you rejected.
When a result looks too good, be suspicious out loud and suggest the check.
