#!/usr/bin/env bash
set -euo pipefail

cd /Users/kezhang/Desktop/projects/geos/geos_github

DATASET="datasets/phreeqc_adaptive_neutral_gpt55_100_v1_balanced.jsonl"
WORKERS="${WORKERS:-4}"

python -u scripts/oneshot.py \
  --dataset "$DATASET" \
  --model gpt-5.1 \
  --name adaptive_neutral_gpt55_100_balanced_oneshot_gpt51 \
  --workers "$WORKERS" \
  --prompt-mode direct \
  --task-filter mcq

python -u scripts/oneshot.py \
  --dataset "$DATASET" \
  --model gpt-5.2 \
  --name adaptive_neutral_gpt55_100_balanced_oneshot_gpt52 \
  --workers "$WORKERS" \
  --prompt-mode direct \
  --task-filter mcq

python -u scripts/oneshot.py \
  --dataset "$DATASET" \
  --model claude-sonnet-4-6 \
  --name adaptive_neutral_gpt55_100_balanced_oneshot_sonnet46 \
  --workers "$WORKERS" \
  --prompt-mode direct \
  --task-filter mcq

python -u scripts/oneshot.py \
  --dataset "$DATASET" \
  --model claude-opus-4-6 \
  --name adaptive_neutral_gpt55_100_balanced_oneshot_opus46 \
  --workers "$WORKERS" \
  --prompt-mode direct \
  --task-filter mcq
