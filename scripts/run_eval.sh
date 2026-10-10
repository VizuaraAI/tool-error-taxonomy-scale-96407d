#!/usr/bin/env bash
# Evaluate one model on one benchmark.
#
#   ./scripts/run_eval.sh 0.5B bfcl
#   ./scripts/run_eval.sh 3B nestful call_set_planning_prompt
#
# Start with a small --limit. The first run downloads the checkpoint, so give
# it a few minutes before assuming it has hung.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${1:-0.5B}"
BENCHMARK="${2:-bfcl}"
REMEDY="${3:-}"
LIMIT="${LIMIT:-200}"
SEED="${SEED:-0}"

ARGS=(--model "$MODEL" --benchmark "$BENCHMARK" --limit "$LIMIT" --seed "$SEED")
if [ -n "$REMEDY" ]; then
  ARGS+=(--remedy "$REMEDY")
fi

echo "==> ${MODEL} on ${BENCHMARK}${REMEDY:+ with remedy ${REMEDY}} (limit ${LIMIT}, seed ${SEED})"
python -m src.evaluate "${ARGS[@]}"
