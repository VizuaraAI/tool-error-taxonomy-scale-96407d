#!/usr/bin/env bash
# One-time setup. Creates a virtualenv, installs dependencies, and proves the
# measuring instrument works before any weights are downloaded.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"

if [ ! -d .venv ]; then
  echo "creating .venv"
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo
echo "--- self-test of the attribution cascade (no model needed) ---"
python -m src.attribution --selftest

echo
echo "--- unit tests ---"
python -m pytest -q

echo
echo "Setup complete. Next:"
echo "  source .venv/bin/activate"
echo "  ./scripts/run_eval.sh 0.5B bfcl"
