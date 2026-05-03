#!/usr/bin/env bash
# Run direct one-shot and no-tool CoT baselines on the clean NeurIPS dataset.
#
# Usage:
#   bash scripts/run_neurips_oneshot.sh
#   MODELS="gpt-5.2 claude-sonnet-4-6" bash scripts/run_neurips_oneshot.sh
#   WORKERS=8 bash scripts/run_neurips_oneshot.sh
#
# Gemini is intentionally not in the default model list because current API
# latency is too high for routine reruns. Run it explicitly if needed:
#   MODELS="gemini/gemini-2.5-flash" bash scripts/run_neurips_oneshot.sh
#   MODELS="gemini/gemini-3-flash-preview" bash scripts/run_neurips_oneshot.sh

set -u -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASET="${DATASET:-datasets/neurips_dataset_200_clean.jsonl}"
WORKERS="${WORKERS:-4}"
RUN_PREFIX="${RUN_PREFIX:-neurips200_clean}"
MODELS="${MODELS:-claude-opus-4-6 claude-sonnet-4-6 gpt-5.1 gpt-5.2}"

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

if [[ ! -f "$DATASET" ]]; then
    echo "ERROR: dataset not found: $DATASET" >&2
    exit 1
fi

echo "[run_neurips_oneshot] dataset=$DATASET workers=$WORKERS"
echo "[run_neurips_oneshot] models=$MODELS"

FAILED=0
for model in $MODELS; do
    short="$(model_short "$model")"
    echo
    echo "== $model =="

    python3 scripts/oneshot.py \
        --dataset "$DATASET" \
        --model "$model" \
        --name "${RUN_PREFIX}_oneshot_${short}" \
        --prompt-mode direct \
        --workers "$WORKERS" || FAILED=$((FAILED + 1))

    python3 scripts/oneshot.py \
        --dataset "$DATASET" \
        --model "$model" \
        --name "${RUN_PREFIX}_cot_${short}" \
        --prompt-mode cot \
        --workers "$WORKERS" || FAILED=$((FAILED + 1))
done

exit "$FAILED"
