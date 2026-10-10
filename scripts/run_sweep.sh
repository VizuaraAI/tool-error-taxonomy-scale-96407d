#!/usr/bin/env bash
# The full study: every ladder rung, with and without every remedy, then the
# analysis. This is what fills Table 1, Table 2 and Figure 2.
#
#   ./scripts/run_sweep.sh                 # baselines only, fast
#   WITH_REMEDIES=1 ./scripts/run_sweep.sh # the whole grid, slow
#
# Budget roughly: baselines are 3 runs, the full grid is 3 x 8 = 24 runs per
# benchmark per seed. Do the baselines first and look at the error profile
# before you spend a night on the grid.
set -euo pipefail

cd "$(dirname "$0")/.."

MODELS=("0.5B" "1.5B" "3B")
BENCHMARKS=("${BENCHMARKS:-bfcl}")
REMEDIES=(
  grammar_constrained_decoding
  tool_doc_retrieval
  schema_name_canonicalisation
  type_coercion_repair
  value_grounding_fewshot
  call_set_planning_prompt
  duplicate_call_state_tracker
)
LIMIT="${LIMIT:-200}"
SEED="${SEED:-0}"
WITH_REMEDIES="${WITH_REMEDIES:-0}"

for benchmark in "${BENCHMARKS[@]}"; do
  for model in "${MODELS[@]}"; do
    LIMIT="$LIMIT" SEED="$SEED" ./scripts/run_eval.sh "$model" "$benchmark"
    if [ "$WITH_REMEDIES" = "1" ]; then
      for remedy in "${REMEDIES[@]}"; do
        LIMIT="$LIMIT" SEED="$SEED" ./scripts/run_eval.sh "$model" "$benchmark" "$remedy"
      done
    fi
  done
done

echo
echo "==> analysis"
python -m src.analyze --runs runs --out analysis
echo "Table 1 -> analysis/error_mass.csv and analysis/elasticity.csv"
echo "Table 2 -> analysis/remedy_gain.csv"
