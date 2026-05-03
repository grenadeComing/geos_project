#!/usr/bin/env bash
# Full sweep on phreeqc_bench_v2 for remaining model × strategy cells.
# Already done: GPT-5.1 × {toc, full, summary, rag}, GPT-5.2 × {toc, full, summary, rag}.
# This run: Sonnet, Opus, Gemini 2.5 Pro, Gemini 2.5 Flash × {toc, full, summary, rag}.
# Ordered cheapest → most expensive so you can Ctrl-C early without burning the Opus budget.
# Usage:  bash scripts/run_custom_all.sh

set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Gemini 2.5 Flash — cheapest, run first
python scripts/evaluate_custom.py --model gemini/gemini-2.5-flash --context-mode toc     --name custom_toc_gflash
python scripts/evaluate_custom.py --model gemini/gemini-2.5-flash --context-mode full    --name custom_raw_gflash
python scripts/evaluate_custom.py --model gemini/gemini-2.5-flash --context-mode summary --name custom_summary_gflash
python scripts/evaluate_custom.py --model gemini/gemini-2.5-flash --context-mode rag     --name custom_rag_gflash

# Gemini 2.5 Pro
python scripts/evaluate_custom.py --model gemini/gemini-2.5-pro --context-mode toc     --name custom_toc_gpro
python scripts/evaluate_custom.py --model gemini/gemini-2.5-pro --context-mode full    --name custom_raw_gpro
python scripts/evaluate_custom.py --model gemini/gemini-2.5-pro --context-mode summary --name custom_summary_gpro
python scripts/evaluate_custom.py --model gemini/gemini-2.5-pro --context-mode rag     --name custom_rag_gpro

# Claude Sonnet 4.6
python scripts/evaluate_custom.py --model claude-sonnet-4-6 --context-mode toc     --name custom_toc_sonnet
python scripts/evaluate_custom.py --model claude-sonnet-4-6 --context-mode full    --name custom_raw_sonnet
python scripts/evaluate_custom.py --model claude-sonnet-4-6 --context-mode summary --name custom_summary_sonnet
python scripts/evaluate_custom.py --model claude-sonnet-4-6 --context-mode rag     --name custom_rag_sonnet

# Claude Opus 4.6 — most expensive, run last
python scripts/evaluate_custom.py --model claude-opus-4-6 --context-mode toc     --name custom_toc_opus
python scripts/evaluate_custom.py --model claude-opus-4-6 --context-mode full    --name custom_raw_opus
python scripts/evaluate_custom.py --model claude-opus-4-6 --context-mode summary --name custom_summary_opus
python scripts/evaluate_custom.py --model claude-opus-4-6 --context-mode rag     --name custom_rag_opus
