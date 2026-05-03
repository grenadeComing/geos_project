#!/usr/bin/env bash
# Run the controlled custom-agent experiments for the NeurIPS E&D paper.
#
# This script intentionally runs only the custom PHREEQC agent:
#   1. TOC-guided tool return
#   2. raw/truncated-output tool return
#
# It does not rerun no-tool baselines, and it does not run summary/RAG.
#
# Usage:
#   bash scripts/run_neurips_custom_agents.sh
#   MODELS="claude-sonnet-4-6 gpt-5.2" bash scripts/run_neurips_custom_agents.sh
#   MODES="toc" bash scripts/run_neurips_custom_agents.sh
#   WORKERS=2 bash scripts/run_neurips_custom_agents.sh
#   START_INDEX=114 END_INDEX=200 MODES="full" bash scripts/run_neurips_custom_agents.sh
#   MAX_FULL_CHARS=250000 MODELS="gpt-5.4" MODES="full" bash scripts/run_neurips_custom_agents.sh

set -u -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MAX_FULL_CHARS_WAS_SET="${MAX_FULL_CHARS+x}"
DATASET="${DATASET:-datasets/neurips_dataset_200_clean.jsonl}"
WORKERS="${WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-24}"
TIMEOUT="${TIMEOUT:-600}"
MAX_FULL_CHARS="${MAX_FULL_CHARS:-100000}"
DRY_RUN="${DRY_RUN:-0}"
RUN_PREFIX="${RUN_PREFIX:-neurips200_clean_agent}"
MODELS="${MODELS:-claude-opus-4-6 claude-sonnet-4-6 gpt-5.1 gpt-5.2}"
MODES="${MODES:-toc full}"
START_INDEX="${START_INDEX:-}"
END_INDEX="${END_INDEX:-}"

model_short() {
    case "$1" in
        claude-opus-4-6) echo "opus46" ;;
        claude-sonnet-4-6) echo "sonnet46" ;;
        gpt-5.1) echo "gpt51" ;;
        gpt-5.2) echo "gpt52" ;;
        gemini/gemini-2.5-flash) echo "gflash" ;;
        gemini/gemini-2.5-pro) echo "gpro" ;;
        gemini/gemini-3-flash-preview|google/gemini-3-flash-preview|gemini-3-flash-preview) echo "gflash3" ;;
        *) echo "$1" | tr -c 'A-Za-z0-9' '_' ;;
    esac
}

raw_label() {
    if [[ "$MAX_FULL_CHARS" =~ ^[0-9]+$ ]] && (( MAX_FULL_CHARS % 1000 == 0 )); then
        echo "raw$((MAX_FULL_CHARS / 1000))k"
    else
        echo "raw${MAX_FULL_CHARS}chars"
    fi
}

mode_name() {
    case "$1" in
        toc) echo "toc" ;;
        full) raw_label ;;
        *) echo "$1" ;;
    esac
}

if [[ ! -f "$DATASET" ]]; then
    echo "ERROR: dataset not found: $DATASET" >&2
    exit 1
fi

if [[ -n "${cMAX_FULL_CHARS:-}" && -z "$MAX_FULL_CHARS_WAS_SET" ]]; then
    echo "ERROR: saw cMAX_FULL_CHARS=$cMAX_FULL_CHARS, but MAX_FULL_CHARS is not set." >&2
    echo "This is usually a typo. Use MAX_FULL_CHARS=$cMAX_FULL_CHARS instead." >&2
    exit 1
fi

if [[ "$RUN_PREFIX" == *100k* && "$MAX_FULL_CHARS" != "100000" ]]; then
    echo "ERROR: RUN_PREFIX contains '100k' but MAX_FULL_CHARS=$MAX_FULL_CHARS." >&2
    echo "Use a neutral prefix, for example RUN_PREFIX=\"neurips200_clean_agent\", so the output label carries the raw cap." >&2
    exit 1
fi

echo "[run_neurips_custom_agents] dataset=$DATASET workers=$WORKERS max_steps=$MAX_STEPS timeout=$TIMEOUT"
echo "[run_neurips_custom_agents] models=$MODELS"
echo "[run_neurips_custom_agents] modes=$MODES"
echo "[run_neurips_custom_agents] max_full_chars=$MAX_FULL_CHARS"
if [[ "$DRY_RUN" == "1" ]]; then
    echo "[run_neurips_custom_agents] dry_run=1"
fi
if [[ -n "$START_INDEX" || -n "$END_INDEX" ]]; then
    echo "[run_neurips_custom_agents] range=${START_INDEX:-START}-${END_INDEX:-END}"
fi

RANGE_ARGS=()
if [[ -n "$START_INDEX" ]]; then
    RANGE_ARGS+=(--start-index "$START_INDEX")
fi
if [[ -n "$END_INDEX" ]]; then
    RANGE_ARGS+=(--end-index "$END_INDEX")
fi

FAILED=0
for model in $MODELS; do
    short="$(model_short "$model")"

    echo
    echo "== $model =="

    for mode in $MODES; do
        label="$(mode_name "$mode")"
        cmd=(python3 scripts/evaluate_custom.py \
            --dataset "$DATASET" \
            --model "$model" \
            --name "${RUN_PREFIX}_${label}_${short}" \
            --context-mode "$mode" \
            --max-steps "$MAX_STEPS" \
            --workers "$WORKERS" \
            --timeout "$TIMEOUT" \
            --max-full-chars "$MAX_FULL_CHARS")
        if ((${#RANGE_ARGS[@]})); then
            cmd+=("${RANGE_ARGS[@]}")
        fi
        cmd+=(--resume)
        if [[ "$DRY_RUN" == "1" ]]; then
            printf '%q ' "${cmd[@]}"
            echo
        else
            "${cmd[@]}" || FAILED=$((FAILED + 1))
        fi
    done
done

exit "$FAILED"
