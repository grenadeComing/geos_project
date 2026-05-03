#!/usr/bin/env bash
# Run the numeric fill-in-the-blank custom-agent experiments for one or more models.
#
# Usage:
#   bash scripts/run_fill_in_blank_agents.sh gpt-5.4
#   bash scripts/run_fill_in_blank_agents.sh claude-sonnet-4-6 claude-opus-4-6
#
# Optional overrides:
#   DATASET=datasets/fill_in_blank_182.jsonl
#   RUN_PREFIX=fill_in_blank_182_agent
#   WORKERS=2
#   MAX_STEPS=24
#   TIMEOUT=600
#   START_INDEX=1 END_INDEX=50

set -u -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASET="${DATASET:-datasets/fill_in_blank_182.jsonl}"
RUN_PREFIX="${RUN_PREFIX:-fill_in_blank_182_agent}"
WORKERS="${WORKERS:-2}"
MAX_STEPS="${MAX_STEPS:-24}"
TIMEOUT="${TIMEOUT:-600}"
START_INDEX="${START_INDEX:-}"
END_INDEX="${END_INDEX:-}"

if [[ "$#" -lt 1 ]]; then
    echo "Usage: bash scripts/run_fill_in_blank_agents.sh MODEL [MODEL ...]" >&2
    echo "Example: bash scripts/run_fill_in_blank_agents.sh gpt-5.4" >&2
    exit 2
fi

if [[ ! -f "$DATASET" ]]; then
    echo "ERROR: dataset not found: $DATASET" >&2
    exit 1
fi

model_short() {
    case "$1" in
        claude-opus-4-6) echo "opus46" ;;
        claude-sonnet-4-6) echo "sonnet46" ;;
        gpt-5.1) echo "gpt51" ;;
        gpt-5.2) echo "gpt52" ;;
        gpt-5.4) echo "gpt54" ;;
        gemini/gemini-2.5-flash) echo "gflash" ;;
        gemini/gemini-2.5-pro) echo "gpro" ;;
        gemini/gemini-3-flash-preview|google/gemini-3-flash-preview|gemini-3-flash-preview) echo "gflash3" ;;
        *) echo "$1" | tr -c 'A-Za-z0-9' '_' | sed 's/_$//' ;;
    esac
}

run_one() {
    local model="$1"
    local short="$2"
    local label="$3"
    local mode="$4"
    local max_full_chars="$5"

    local cmd=(python3 scripts/evaluate_custom.py
        --dataset "$DATASET"
        --model "$model"
        --name "${RUN_PREFIX}_${label}_${short}"
        --context-mode "$mode"
        --max-steps "$MAX_STEPS"
        --workers "$WORKERS"
        --timeout "$TIMEOUT"
        --max-full-chars "$max_full_chars"
        --task-filter fill
        --resume)

    if [[ -n "$START_INDEX" ]]; then
        cmd+=(--start-index "$START_INDEX")
    fi
    if [[ -n "$END_INDEX" ]]; then
        cmd+=(--end-index "$END_INDEX")
    fi

    printf '\n== %s / %s ==\n' "$model" "$label"
    "${cmd[@]}"
}

echo "[run_fill_in_blank_agents] dataset=$DATASET workers=$WORKERS max_steps=$MAX_STEPS timeout=$TIMEOUT"
echo "[run_fill_in_blank_agents] models=$*"
if [[ -n "$START_INDEX" || -n "$END_INDEX" ]]; then
    echo "[run_fill_in_blank_agents] range=${START_INDEX:-START}-${END_INDEX:-END}"
fi

FAILED=0
for model in "$@"; do
    short="$(model_short "$model")"
    run_one "$model" "$short" "toc" "toc" "100000" || FAILED=$((FAILED + 1))
    run_one "$model" "$short" "raw100k" "full" "100000" || FAILED=$((FAILED + 1))
    run_one "$model" "$short" "raw500k" "full" "500000" || FAILED=$((FAILED + 1))
done

exit "$FAILED"
