#!/usr/bin/env bash
# Sweep custom agent over all (model × context_mode) combinations on phreeqc_bench_v2.
#
# Usage:
#   bash scripts/run_custom_all.sh                # run everything (resume-safe)
#   MODELS="claude-sonnet-4-6" bash scripts/run_custom_all.sh   # subset
#   MODES="toc full" bash scripts/run_custom_all.sh             # subset
#   WORKERS=2 bash scripts/run_custom_all.sh                    # throttle
#   DRY_RUN=1 bash scripts/run_custom_all.sh                    # print commands only
#
# Logs stream to result/custom/<run_name>/run.log and each run is --resume-safe,
# so re-running after a crash picks up where it left off.

set -u -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── knobs (env-overridable) ─────────────────────────────────────────────
MODELS="${MODELS:-claude-sonnet-4-6 claude-opus-4-6 gpt-5.2}"
MODES="${MODES:-toc full summary rag}"
DATASET="${DATASET:-datasets/phreeqc_bench_v2.jsonl}"
WORKERS="${WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-24}"
TIMEOUT="${TIMEOUT:-600}"
DRY_RUN="${DRY_RUN:-0}"
LOG_ROOT="${LOG_ROOT:-result/custom}"

# Short label for run-name suffix.
model_short() {
    case "$1" in
        claude-sonnet-4-6) echo "sonnet";;
        claude-opus-4-6)   echo "opus";;
        gpt-5.2)           echo "gpt52";;
        gpt-5-2|gpt52)     echo "gpt52";;
        gemini/*)          echo "gemini";;
        *)                 echo "$1" | tr -c 'A-Za-z0-9' '_' ;;
    esac
}

# ── preflight ───────────────────────────────────────────────────────────
if [[ ! -f "$DATASET" ]]; then
    echo "ERROR: dataset not found: $DATASET" >&2
    exit 1
fi
if [[ ! -f scripts/evaluate_custom.py ]]; then
    echo "ERROR: scripts/evaluate_custom.py not found — run from repo root." >&2
    exit 1
fi
echo "[run_custom_all] repo=$REPO_ROOT"
echo "[run_custom_all] dataset=$DATASET  workers=$WORKERS  max_steps=$MAX_STEPS  timeout=$TIMEOUT"
echo "[run_custom_all] models: $MODELS"
echo "[run_custom_all] modes:  $MODES"
echo

# ── sweep ───────────────────────────────────────────────────────────────
TOTAL=0; FAILED=0; PASSED=0; SKIPPED=0
for model in $MODELS; do
    mshort="$(model_short "$model")"
    for mode in $MODES; do
        TOTAL=$((TOTAL+1))
        name="custom_${mode}_${mshort}"
        run_dir="${LOG_ROOT}/${name}"
        log_file="${run_dir}/run.log"

        echo "────────────────────────────────────────────────────────────"
        echo "[$(date '+%H:%M:%S')] ($TOTAL) $name  (model=$model mode=$mode)"

        cmd=(python3 scripts/evaluate_custom.py
             --model "$model"
             --context-mode "$mode"
             --name "$name"
             --dataset "$DATASET"
             --workers "$WORKERS"
             --max-steps "$MAX_STEPS"
             --timeout "$TIMEOUT"
             --resume)

        if [[ "$DRY_RUN" == "1" ]]; then
            echo "  DRY: ${cmd[*]}"
            continue
        fi

        mkdir -p "$run_dir"
        # Use tee so we see progress AND keep a log.
        if "${cmd[@]}" 2>&1 | tee "$log_file"; then
            PASSED=$((PASSED+1))
            # Pull final accuracy out of summary.json if present.
            summary_json="${run_dir}/summary.json"
            if [[ -f "$summary_json" ]]; then
                acc=$(python3 -c "import json,sys; print(json.load(open('$summary_json'))['accuracy'])" 2>/dev/null || echo "?")
                echo "[$(date '+%H:%M:%S')] ✓ $name  accuracy=$acc"
            else
                echo "[$(date '+%H:%M:%S')] ✓ $name  (no summary.json found)"
            fi
        else
            FAILED=$((FAILED+1))
            echo "[$(date '+%H:%M:%S')] ✗ $name  (see $log_file)"
        fi
    done
done

echo
echo "════════════════════════════════════════════════════════════"
echo "[run_custom_all] done: total=$TOTAL  ok=$PASSED  failed=$FAILED"
if [[ "$DRY_RUN" != "1" && $PASSED -gt 0 ]]; then
    echo
    echo "Quick leaderboard:"
    for model in $MODELS; do
        mshort="$(model_short "$model")"
        for mode in $MODES; do
            name="custom_${mode}_${mshort}"
            s="${LOG_ROOT}/${name}/summary.json"
            [[ -f "$s" ]] || continue
            acc=$(python3 -c "import json; d=json.load(open('$s')); print(f\"{d['accuracy']*100:5.1f}%  in={d.get('total_input_tokens',0):>8}  out={d.get('total_output_tokens',0):>6}\")" 2>/dev/null || echo "?")
            printf "  %-32s %s\n" "$name" "$acc"
        done
    done
fi

exit $FAILED
