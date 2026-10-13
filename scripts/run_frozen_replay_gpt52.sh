#!/usr/bin/env bash
# GPT-5.2 only. Prepare offline if needed, validate, then resume paid continuations.
# DRY_RUN=1 bash scripts/run_frozen_replay_gpt52.sh performs no API calls.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PY="${PY:-python3}"
RUN_GROUP="${RUN_GROUP:-frozen_replay_gpt52}"
OUT="result/custom/$RUN_GROUP"
CHECKPOINTS="$OUT/checkpoints"
REPEATS="${REPEATS:-1}"
WORKERS="${WORKERS:-4}"
MAX_COST_USD="${MAX_COST_USD:-120}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "$DRY_RUN" != "1" ]]; then
    "$PY" -c 'import litellm' || { echo "Activate the Python environment with litellm, or set PY." >&2; exit 1; }
    [[ -n "${OPENAI_API_KEY:-}" ]] || { echo "OPENAI_API_KEY is not set. Export it in your shell before running." >&2; exit 1; }
fi

if [[ ! -f "$CHECKPOINTS/manifest.json" ]]; then
    "$PY" scripts/prepare_frozen_replay.py --output-root "$CHECKPOINTS"
fi

ARGS=(--manifest "$CHECKPOINTS/manifest.json" --output "$OUT" --model gpt-5.2
      --repeats "$REPEATS" --workers "$WORKERS" --max-cost-usd "$MAX_COST_USD")
"$PY" -u scripts/run_frozen_replay.py "${ARGS[@]}" --dry-run
if [[ "$DRY_RUN" == "1" ]]; then
    exit 0
fi
mkdir -p "$OUT"
"$PY" -u scripts/run_frozen_replay.py "${ARGS[@]}" --run 2>&1 | tee -a "$OUT/console.log"
