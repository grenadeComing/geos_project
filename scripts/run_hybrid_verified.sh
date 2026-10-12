#!/usr/bin/env bash
# Hybrid output-access runs on the 158 verified items.
#
# Hybrid mode inlines result.out when it fits MAX_FULL_CHARS (as Raw100k does)
# and returns an uncapped section TOC otherwise (as the fixed TOC runs do).
# Each round runs all MODELS in parallel; rounds run one after another.
# All runs land in one folder: result/custom/$RUN_GROUP/<model>_run<i>/
#
# Usage (from the repo root, inside the conda env that has litellm):
#   SMOKE=1 bash scripts/run_hybrid_verified.sh     # 2 items per model, to check setup
#   bash scripts/run_hybrid_verified.sh             # full: 3 rounds x {gpt-5.1, gpt-5.2}
#   RUNS=2 bash scripts/run_hybrid_verified.sh
#   PY=~/miniconda3/envs/app_test/bin/python bash scripts/run_hybrid_verified.sh

set -u -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-python3}"
DATASET="${DATASET:-datasets/neurips_dataset_158_verified.jsonl}"
MODELS="${MODELS:-gpt-5.1 gpt-5.2}"
RUNS="${RUNS:-3}"
WORKERS="${WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-24}"
TIMEOUT="${TIMEOUT:-600}"
MAX_FULL_CHARS="${MAX_FULL_CHARS:-100000}"
SMOKE="${SMOKE:-0}"
RUN_GROUP="${RUN_GROUP:-hybrid158_verified}"
export PHREEQC_BIN="${PHREEQC_BIN:-$(cd "$REPO_ROOT/.." && pwd)/phreeqc-3.8.6-17100/src/phreeqc}"

model_short() {
    case "$1" in
        gpt-5.1) echo "gpt51" ;;
        gpt-5.2) echo "gpt52" ;;
        gpt-5.4) echo "gpt54" ;;
        claude-sonnet-4-6) echo "sonnet46" ;;
        claude-opus-4-6) echo "opus46" ;;
        *) echo "$1" | tr -c 'A-Za-z0-9' '_' ;;
    esac
}

# Fail fast on setup problems, before any API call is made.
[[ -x "$PHREEQC_BIN" ]] || { echo "ERROR: PHREEQC binary not found: $PHREEQC_BIN (set PHREEQC_BIN)" >&2; exit 1; }
[[ -f "$DATASET" ]] || { echo "ERROR: dataset not found: $DATASET" >&2; exit 1; }
"$PY" -c "import litellm" 2>/dev/null || { echo "ERROR: $PY cannot import litellm; activate the right conda env or set PY" >&2; exit 1; }
[[ -n "${OPENAI_API_KEY:-}" ]] || { echo "ERROR: OPENAI_API_KEY is not set" >&2; exit 1; }

RANGE_ARGS=()
if [[ "$SMOKE" == "1" ]]; then
    RUN_GROUP="${RUN_GROUP}_smoke"
    RUNS=1
    RANGE_ARGS=(--end-index 7)   # first two verified items (ids 2 and 7)
fi

LOG_DIR="result/custom/$RUN_GROUP/logs"
mkdir -p "$LOG_DIR"
echo "[hybrid] dataset=$DATASET models=$MODELS runs=$RUNS workers=$WORKERS max_full_chars=$MAX_FULL_CHARS"
echo "[hybrid] phreeqc=$PHREEQC_BIN python=$PY"
echo "[hybrid] results -> result/custom/$RUN_GROUP/   logs -> $LOG_DIR/"

FAILED=0
for run in $(seq 1 "$RUNS"); do
    echo
    echo "== round $run/$RUNS: $MODELS (in parallel) =="
    pids=()
    for model in $MODELS; do
        name="$RUN_GROUP/$(model_short "$model")_run$run"
        "$PY" scripts/evaluate_custom.py \
            --dataset "$DATASET" \
            --model "$model" \
            --name "$name" \
            --context-mode hybrid \
            --max-full-chars "$MAX_FULL_CHARS" \
            --max-steps "$MAX_STEPS" \
            --workers "$WORKERS" \
            --timeout "$TIMEOUT" \
            ${RANGE_ARGS[@]+"${RANGE_ARGS[@]}"} \
            --resume \
            > "$LOG_DIR/$(model_short "$model")_run$run.log" 2>&1 &
        pids+=("$!")
        echo "  started $model -> result/custom/$name (pid $!)"
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || FAILED=$((FAILED + 1))
    done
    for model in $MODELS; do
        s="result/custom/$RUN_GROUP/$(model_short "$model")_run$run/summary.json"
        if [[ -f "$s" ]]; then
            "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));print(f\"  {d['model']}: {d['correct_rows']}/{d['total_rows']} correct, {d['total_input_tokens']/1e6:.2f}M input tokens\")" "$s"
        else
            echo "  $model: no summary.json (see $LOG_DIR/$(model_short "$model")_run$run.log)"
        fi
    done
done

echo
echo "[hybrid] done; failed processes: $FAILED"
exit "$FAILED"
