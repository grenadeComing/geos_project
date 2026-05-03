#!/usr/bin/env bash
# Core PHREEQC-MCQ-200 experiment suite for the NeurIPS E&D rewrite.
#
# Runs only the experiments intended for the PHREEQC benchmark paper:
#   1. direct no-tool baseline
#   2. CoT no-tool baseline
#   3. TOC agent
#   4. raw-output agent ablation
#
# It intentionally does not run summary or RAG; those belong to the future
# multi-simulator output-access paper.
#
# Prefer `run_neurips_oneshot.sh` and `run_neurips_custom_agents.sh` for the
# current rerun. This all-in-one script is kept only for reproducibility.
#
# Usage:
#   bash scripts/run_neurips_core.sh
#   MODELS="gpt-5.2 claude-sonnet-4-6" bash scripts/run_neurips_core.sh
#   WORKERS=2 DRY_RUN=1 bash scripts/run_neurips_core.sh

set -u -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASET="${DATASET:-datasets/neurips_dataset_200_clean.jsonl}"
WORKERS="${WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-24}"
TIMEOUT="${TIMEOUT:-600}"
MAX_FULL_CHARS="${MAX_FULL_CHARS:-100000}"
DRY_RUN="${DRY_RUN:-0}"
MODELS="${MODELS:-claude-opus-4-6 claude-sonnet-4-6 gpt-5.1 gpt-5.2}"
RUN_PREFIX="${RUN_PREFIX:-neurips_dataset_200}"

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

run_cmd() {
    echo "$*"
    if [[ "$DRY_RUN" == "1" ]]; then
        return 0
    fi
    "$@"
}

if [[ ! -f "$DATASET" ]]; then
    echo "ERROR: dataset not found: $DATASET" >&2
    exit 1
fi

echo "[run_neurips_core] dataset=$DATASET run_prefix=$RUN_PREFIX workers=$WORKERS max_steps=$MAX_STEPS timeout=$TIMEOUT max_full_chars=$MAX_FULL_CHARS"
echo "[run_neurips_core] models=$MODELS"

FAILED=0
for model in $MODELS; do
    short="$(model_short "$model")"

    echo
    echo "== $model =="

    run_cmd python3 scripts/oneshot.py \
        --dataset "$DATASET" \
        --model "$model" \
        --name "${RUN_PREFIX}_oneshot_${short}" \
        --prompt-mode direct \
        --workers "$WORKERS" || FAILED=$((FAILED + 1))

    run_cmd python3 scripts/oneshot.py \
        --dataset "$DATASET" \
        --model "$model" \
        --name "${RUN_PREFIX}_cot_${short}" \
        --prompt-mode cot \
        --workers "$WORKERS" || FAILED=$((FAILED + 1))

    run_cmd python3 scripts/evaluate_custom.py \
        --dataset "$DATASET" \
        --model "$model" \
        --name "${RUN_PREFIX}_toc_${short}" \
        --context-mode toc \
        --max-steps "$MAX_STEPS" \
        --workers "$WORKERS" \
        --timeout "$TIMEOUT" \
        --resume || FAILED=$((FAILED + 1))

    run_cmd python3 scripts/evaluate_custom.py \
        --dataset "$DATASET" \
        --model "$model" \
        --name "${RUN_PREFIX}_$(raw_label)_${short}" \
        --context-mode full \
        --max-steps "$MAX_STEPS" \
        --workers "$WORKERS" \
        --timeout "$TIMEOUT" \
        --max-full-chars "$MAX_FULL_CHARS" \
        --resume || FAILED=$((FAILED + 1))
done

exit "$FAILED"
