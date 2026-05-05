# NeurIPS E&D Draft

This directory is the working NeurIPS Evaluations & Datasets rewrite.

## Paper identity

Main framing:

> PHREEQC-MCQ-200 is an evaluation benchmark for LLM agents that operate
> deterministic, large-output scientific simulators.

PHREEQC is the simulator testbed / use case, not the paper's primary identity.
TOC-guided output access is a supporting output-management mechanism, not a
standalone claim that TOC is the best possible protocol.

## What belongs in this paper

- PHREEQC-MCQ-200 dataset construction and validation.
- One-shot, CoT, and TOC-agent results on the frozen 200-question clean benchmark.
- Gain/loss/retention analysis.
- TOC-vs-raw ablation as a minimal output-management check.
- Latency, token usage, unparseable answer counts, per-option accuracy.
- Failure analysis centered on agent/tool failure modes.

## What does not belong in this paper

- Summary-vs-RAG-vs-TOC as a major result.
- Claims that TOC is universally superior.
- Cross-simulator generalization claims.

Summary/RAG are better reserved for a future multi-simulator output-access
protocol paper.

## Before compiling

Download the official NeurIPS 2026 style file into this directory:

```text
neurips_2026.sty
```

Use the E&D option:

```tex
\usepackage[eandd]{neurips_2026}
```

The official 2026 instructions say submissions must use the current-year style
file and include the NeurIPS checklist. The checklist is filled in this draft,
but should be rechecked after the final artifact package is frozen.

## Updating results

All result values are centralized in:

```text
results_macros.tex
```

After rerunning or completing experiments on
`datasets/neurips_dataset_200_clean.jsonl`, update the corresponding macros
there. Use `--` for conditions that were intentionally not run.
Do not put old ACM numbers in the main text.

The analysis outputs used by the draft are in:

```text
../neurips_analysis/
```

Regenerate them from the repository root with:

```bash
node neurips_analysis/analyze_results.js
```

The intended core run scripts are `scripts/run_neurips_oneshot.sh` for direct
and CoT baselines and `scripts/run_neurips_custom_agents.sh` for TOC/raw agent
runs. The paper intentionally excludes summary/RAG from the main result set.
