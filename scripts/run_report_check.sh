#!/usr/bin/env bash
# Causal check of the "what agents read" mechanism on the 158 verified items.
#
#   toc_report  TOC plus an instruction to answer from result.out as printed,
#               not from SELECTED_OUTPUT files (the intervention)
#   toc         plain uncapped TOC run at the same time (contemporaneous control,
#               also rules out model drift relative to the May TOC runs)
#
# Each round runs all MODELS in parallel for one condition; rounds run in sequence.
# All runs land in one folder: result/custom/$RUN_GROUP/<model>_<condition>_run<i>/
#
# Usage (from the repo root, inside the conda env that has litellm):
#   SMOKE=1 bash scripts/run_report_check.sh       # 2 items per model and condition
#   bash scripts/run_report_check.sh               # toc_report x2 + toc x1, for gpt-5.1 and gpt-5.2
#   REPORT_RUNS=3 CONTROL_RUNS=2 bash scripts/run_report_check.sh
#   MODELS="gpt-5.4" CONDITIONS="toc_report:2 full:1" bash scripts/run_report_check.sh
#       CONDITIONS overrides REPORT_RUNS/CONTROL_RUNS: space-separated <mode>:<runs>, mode in
#       toc_report | toc_round | toc | full (full = inline Raw100k, 100k-character cap, as in the inline runs;
#       toc_round = TOC plus a rounding reminder that names no file)

set -u -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-python3}"
DATASET="${DATASET:-datasets/neurips_dataset_158_verified.jsonl}"
MODELS="${MODELS:-gpt-5.1 gpt-5.2}"
REPORT_RUNS="${REPORT_RUNS:-2}"
CONTROL_RUNS="${CONTROL_RUNS:-1}"
CONDITIONS="${CONDITIONS:-toc_report:$REPORT_RUNS toc:$CONTROL_RUNS}"
WORKERS="${WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-24}"
TIMEOUT="${TIMEOUT:-600}"
UNCAPPED_CHARS=10000000   # TOC index built for every output size, as in the fixed TOC runs
INLINE_CHARS=100000       # Raw100k: inline result.out up to 100k characters, as in the inline runs
SMOKE="${SMOKE:-0}"
RUN_GROUP="${RUN_GROUP:-report158_verified}"
export PHREEQC_BIN="${PHREEQC_BIN:-$(cd "$REPO_ROOT/.." && pwd)/phreeqc-3.8.6-17100/src/phreeqc}"

model_short() {
    case "$1" in
        gpt-5.1) echo "gpt51" ;;
        gpt-5.2) echo "gpt52" ;;
        gpt-5.4) echo "gpt54" ;;
        claude-sonnet-4-6) echo "sonnet46" ;;
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
    RANGE_ARGS=(--end-index 7)   # first two verified items (ids 2 and 7)
fi

# Rounds interleave the conditions: <c1> run 1, <c2> run 1, <c1> run 2, ...
COND_MODES=(); COND_RUNS=()
for c in $CONDITIONS; do
    m="${c%%:*}"; n="${c##*:}"
    case "$m" in toc_report|toc_round|toc|full) ;; *) echo "ERROR: unknown condition '$m' in CONDITIONS" >&2; exit 1 ;; esac
    [[ "$n" =~ ^[0-9]+$ ]] || { echo "ERROR: bad run count in '$c'" >&2; exit 1; }
    [[ "$SMOKE" == "1" && "$n" -gt 1 ]] && n=1
    COND_MODES+=("$m"); COND_RUNS+=("$n")
done
ROUNDS=()
max=0
for n in "${COND_RUNS[@]}"; do (( n > max )) && max=$n; done
for i in $(seq 1 "$max"); do
    for k in "${!COND_MODES[@]}"; do
        (( i <= COND_RUNS[k] )) && ROUNDS+=("${COND_MODES[k]}:$i")
    done
done

LOG_DIR="result/custom/$RUN_GROUP/logs"
mkdir -p "$LOG_DIR"
echo "[report-check] dataset=$DATASET models=$MODELS rounds=${ROUNDS[*]} workers=$WORKERS"
echo "[report-check] phreeqc=$PHREEQC_BIN python=$PY"
echo "[report-check] results -> result/custom/$RUN_GROUP/   logs -> $LOG_DIR/"

FAILED=0
for round in "${ROUNDS[@]}"; do
    mode="${round%%:*}"; run="${round##*:}"
    case "$mode" in
        toc_report) label="tocreport"; chars="$UNCAPPED_CHARS" ;;
        toc_round)  label="tocround";  chars="$UNCAPPED_CHARS" ;;
        toc)        label="toc";       chars="$UNCAPPED_CHARS" ;;
        full)       label="raw100k";   chars="$INLINE_CHARS" ;;
    esac
    echo
    echo "== $mode run $run: $MODELS (in parallel) =="
    pids=()
    for model in $MODELS; do
        name="$RUN_GROUP/$(model_short "$model")_${label}_run$run"
        "$PY" scripts/evaluate_custom.py \
            --dataset "$DATASET" \
            --model "$model" \
            --name "$name" \
            --context-mode "$mode" \
            --max-full-chars "$chars" \
            --max-steps "$MAX_STEPS" \
            --workers "$WORKERS" \
            --timeout "$TIMEOUT" \
            ${RANGE_ARGS[@]+"${RANGE_ARGS[@]}"} \
            --resume \
            > "$LOG_DIR/$(model_short "$model")_${label}_run$run.log" 2>&1 &
        pids+=("$!")
        echo "  started $model -> result/custom/$name (pid $!)"
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || FAILED=$((FAILED + 1))
    done
    for model in $MODELS; do
        s="result/custom/$RUN_GROUP/$(model_short "$model")_${label}_run$run/summary.json"
        if [[ -f "$s" ]]; then
            "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));print(f\"  {d['model']}: {d['correct_rows']}/{d['total_rows']} correct, {d['total_input_tokens']/1e6:.2f}M input tokens\")" "$s"
        else
            echo "  $model: no summary.json (see $LOG_DIR/$(model_short "$model")_${label}_run$run.log)"
        fi
    done
done

echo
echo "[report-check] done; failed processes: $FAILED"
exit "$FAILED"
