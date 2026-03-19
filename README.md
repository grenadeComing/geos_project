# Result Structure Guide

This README is only for checking experiment outputs.
Code/run details are intentionally omitted.

**Naming reminder**
- `agent` = our customized in-repo agent
- `claude_sdk` = official Claude Agent SDK agent

## Where to look

All outputs are under `result/`.

```text
result/
  agent/
    <run_name>/
      summary.json
      results.jsonl
      work_space/              # only for agent-style runs
  oneshot/
    <run_name>/
      summary.json
      results.jsonl
  claude_sdk/
    <run_name>/
      summary.json
      results.jsonl
      work_space/
```

## Important naming note

- `result/agent/`: runs from our customized in-repo agent implementation
- `result/claude_sdk/`: runs from the official Claude Agent SDK implementation
- `result/oneshot/`: no-agent one-shot baseline runs

## Main files

- `summary.json`: one run's aggregate metrics
- `results.jsonl`: one line per question (detailed prediction record)
- `work_space/`: per-question artifacts for debugging (inputs, outputs, logs)

## How to read `summary.json`

Typical fields:
- `dataset`: dataset file used
- `provider`: model provider
- `model`: model name
- `method`: run type (`agent`, `one_shot_no_tools`, or `claude_sdk`)
- `run_name`: folder name for this run
- `total_rows`: number of evaluated questions
- `correct_rows`: number answered correctly
- `accuracy`: `correct_rows / total_rows`

## How to read `results.jsonl`

Each line is one JSON record for one question, usually with:
- `index`: question number (1-based)
- `truth`: correct option (`A/B/C/D`)
- `prediction`: predicted option (`A/B/C/D`) or `null`
- `is_correct`: `true`, `false`, or `null`
- `error` (optional): runtime/model error message

Interpretation:
- `is_correct: true` -> correct answer
- `is_correct: false` -> wrong answer
- `prediction: null` with `error` -> failed run for that question

## Agent workspace (`work_space/`)

Agent-style runs may include:
- `dataset_q_00001/`, `dataset_q_00002/`, ...
- `final_answer.txt`: final extracted answer in `<<< X >>>` format
- `result.out` and other PHREEQC outputs
- logs and intermediate files

Use this only when you need to inspect why a question failed or was wrong.

