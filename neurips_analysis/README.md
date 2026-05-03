# NeurIPS Analysis Notes

This folder records the mechanical analyses used by the NeurIPS E&D draft.
It is meant to keep the paper numbers traceable and to separate automatic log
analysis from domain-expert failure analysis.

## IMPORTANT: do NOT re-run `analyze_results.js` casually

The CSVs and `analysis_summary.md` in this folder were generated from the
full original benchmark run. After that run completed, per-question workspace
artifacts (chat logs, `results.jsonl`, `summary.json`) for several Anthropic
and original GPT TOC/Raw runs were lost in a workspace incident; only failure
cases remain in `expert_handoff_neurips/02_MUST_HAVE_failure_analysis/`.

If you re-run the script with the current `result/custom/` contents, the
script will **silently skip** the missing runs and **overwrite** the CSVs
with a reduced subset that does not match the numbers cited in the paper.

To prevent accidental overwrite, the script refuses to run unless you pass
`--force-rerun` or set `ALLOW_ANALYZE_RERUN=1`. Only do this if you have
re-executed the original benchmark and want to regenerate from fresh data.

The CSVs in this folder are the authoritative outputs cited in the paper.

## Inputs

- Dataset: `../datasets/neurips_dataset_200_clean.jsonl`
- Direct/CoT runs: `../result/oneshot/*`
- Agent runs: `../result/custom/*`
- Agent logs and generated PHREEQC artifacts:
  `../result/custom/*/work_space/dataset_q_*/`

## How to rerun

From the repository root:

```bash
node neurips_analysis/analyze_results.js
```

The script overwrites the CSV and Markdown outputs in this folder. It only
reads existing result artifacts; it does not launch new model runs.

## Current outputs

- `analysis_summary.md`: compact human-readable summary.
- `completed_runs.csv`: completed result files, accuracy, ungraded count, and tokens.
- `retention.csv`: kept/gained/lost analysis for Direct -> Agent-TOC.
- `scenario_map.csv`: scenario-stem grouping used for scenario-level metrics.
- `scenario_metrics.csv`: item accuracy, mean scenario accuracy, and bootstrap CIs.
- `per_option.csv`: true-label and predicted-label behavior.
- `toc_raw.csv`: TOC-vs-raw accuracy and token comparison.
- `output_size_summary.csv`: quantiles of observed `result.out` sizes.
- `small_output_size_summary.csv`: distribution restricted to outputs at or
  below the Raw100k cap.
- `output_size_bins.csv`: counts and quantiles for output-size bins.
- `output_size_caps.csv`: how many outputs exceed candidate raw caps and the
  relative capped-character load versus Raw100k.
- `large_output_cap_coverage.csv`: candidate raw caps restricted to the
  `result.out > 100k` tail.
- `answer_value_cap_coverage.csv`: heuristic check for whether the correct
  answer option's numeric value would be hidden in the middle of a head/tail
  truncation at each cap.
- `toc_raw_by_output_size.csv`: TOC-vs-raw accuracy and token comparison split
  by output-size bins.
- `wall_clock.csv`: aggregate elapsed time for all runs and per-item median/p90
  wall-clock time for agent runs with trajectory logs.
- `trajectory.csv`: step counts, PHREEQC executions, read calls, and failed executions.
- `trajectory_failure.csv`: automatic log-derived failure buckets.
- `manual_failure_review_queue.csv`: wrong or ungraded agent cases with artifact paths
  for later expert labeling.

## Definitions

- Scenario stem: first paragraph of the question after whitespace normalization.
  This is a conservative grouping for dependence-aware reporting.
- Scenario CI: nonparametric bootstrap over scenario stems, not over individual
  questions.
- Retention: `kept / (kept + lost)` for a TOC-agent run relative to the matching
  direct no-tool baseline.
- RawNk: raw `result.out` content is returned inline but capped at N thousand
  characters per PHREEQC execution. For example, Raw100k uses 100,000
  characters and Raw250k uses 250,000 characters.
- Agent wall-clock time: seconds from each per-question `run_start` log event
  to the matching `run_end` event. This is not the same as total experiment
  elapsed time because agent runs use multiple workers.
- Run integrity: agent runs with many rows that lack both a `run_end` log event
  and `final_answer.txt` are marked `invalid_*` and excluded from downstream
  metric tables. Isolated cases are kept as operational failures and surfaced as
  warnings.

## Boundary of automatic analysis

The automatic failure categories are not final scientific root-cause labels.
They identify cases such as failed PHREEQC execution, successful run but wrong
answer, step-limit/no-final behavior, or unparseable answer. Domain review is
still needed to decide whether an error came from invalid PHREEQC input,
plausible-but-wrong input, output navigation, numerical extraction, or answer
choice mapping.
